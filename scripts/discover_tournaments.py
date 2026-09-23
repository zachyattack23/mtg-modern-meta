#!/usr/bin/env python3
"""Find Modern tournaments on melee.gg that we have not fetched yet.

melee has no public tournament index -- /Tournament is an error page and the
decklist search needs a sign-in. But Jiliac/MTGODecklistCache scrapes melee
continuously and names each cached file after the tournament, with the melee id
embedded:

    ocher-modern-series-vol-2-449289-2026-09-12.json
                              ^^^^^^ melee tournament id

So one recursive git-tree call against that repo yields every melee tournament
it knows about, and the slug carries the format. That gives us an index without
scraping melee to find what to scrape.

The cache also stores decklists and rounds, but we do not read them for data:
its pairings reference players by display name, whereas melee's own endpoint
gives us decklist ids on both sides of every match. Name-matching would be a
needless source of error when the authoritative join key is one request away.

    python3 scripts/discover_tournaments.py --since 2026-08-01
    python3 scripts/discover_tournaments.py --since 2026-08-01 --min-players 100
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mtgmeta import melee  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_REPO = "Jiliac/MTGODecklistCache"

_FILE_RE = re.compile(r"-(\d{5,8})-(\d{4}-\d{2}-\d{2})\.json$")


def cache_index(years: list[int]) -> list[dict]:
    """Every melee tournament the cache knows about, from its filenames."""
    out: list[dict] = []
    for year in years:
        path = f"master:Tournaments%2Fmelee.gg%2F{year}"
        proc = subprocess.run(
            ["gh", "api", f"repos/{CACHE_REPO}/git/trees/{path}?recursive=1",
             "--jq", '.tree[] | select(.type=="blob") | .path'],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  !! {year}: {proc.stderr.strip()[:120]}", file=sys.stderr)
            continue
        for rel in proc.stdout.splitlines():
            if not rel.endswith(".json"):
                continue
            name = rel.split("/")[-1]
            match = _FILE_RE.search(name)
            if not match:
                continue
            out.append({"id": int(match.group(1)), "date": match.group(2),
                        "slug": name[: match.start()]})
    # The same event can appear under more than one date; keep the earliest.
    seen: dict[int, dict] = {}
    for row in sorted(out, key=lambda r: r["date"]):
        seen.setdefault(row["id"], row)
    return list(seen.values())


def probe(entry: dict) -> dict:
    """Confirm the event on melee and measure how big it actually was.

    A tiny local RCQ and a 1500-player Regional Championship are not the same
    evidence, so size is collected up front rather than discovered after an
    hour of fetching.
    """
    out = dict(entry)
    try:
        meta = melee.get_tournament(entry["id"])
        swiss = [r for r in meta.rounds if r.is_swiss]
        out["name"] = meta.name
        out["date"] = meta.start_date or entry["date"]
        out["swiss_rounds"] = len(swiss)
        if swiss:
            standings = melee.get_standings(entry["id"], swiss[-1].id)
            out["players"] = len(standings)
        else:
            out["players"] = 0
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 - a dead id is an expected outcome
        out["ok"] = False
        out["error"] = str(exc)[:120]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2026])
    ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--until", default="2099-12-31")
    ap.add_argument("--min-players", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the melee round-trip; report slugs only")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "data" / "discovered.json")
    args = ap.parse_args()

    have = {t["id"] for t in json.loads((ROOT / "tournaments.json").read_text())}

    index = cache_index(args.years)
    modern = [r for r in index
              if "modern" in r["slug"].lower()
              and args.since <= r["date"] <= args.until]
    todo = [r for r in modern if r["id"] not in have]

    print(f"{len(index)} melee tournaments in the cache for {args.years}")
    print(f"  Modern, {args.since}..{args.until}: {len(modern)}")
    print(f"  not already in tournaments.json: {len(todo)}")

    if args.no_probe:
        rows = todo
    else:
        print(f"\nprobing melee for size ({args.workers} workers)...")
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(probe, todo))
        bad = [r for r in rows if not r.get("ok")]
        rows = [r for r in rows if r.get("ok")]
        if bad:
            print(f"  {len(bad)} ids did not resolve (private or removed)")

    rows.sort(key=lambda r: -r.get("players", 0))
    keep = [r for r in rows if r.get("players", 0) >= args.min_players]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    total = sum(r.get("players", 0) for r in keep)
    print(f"\n{len(keep)} events at >= {args.min_players} players "
          f"({total:,} player-entries)")
    print(f"\n{'date':<12}{'players':>8}{'rds':>5}  name")
    for r in keep[:40]:
        print(f"{r['date']:<12}{r.get('players', 0):>8}{r.get('swiss_rounds', 0):>5}  "
              f"{(r.get('name') or r['slug'])[:58]}")
    if len(keep) > 40:
        print(f"... and {len(keep) - 40} more in {args.out}")
    print(f"\nwrote {args.out}")
    print("Add the ids you want to tournaments.json, then run fetch_tournaments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
