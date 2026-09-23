#!/usr/bin/env python3
"""Pull tournaments from melee.gg into data/raw/.

Writes one JSON per tournament (metadata + every swiss/playoff match + final
standings) and one JSON per unique decklist under data/cache/decklists/.
Decklists are cached by guid, so re-running only fetches what's new.

    python3 scripts/fetch_tournaments.py 451148 405588 405590
    python3 scripts/fetch_tournaments.py --config tournaments.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mtgmeta import melee  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DECK_CACHE = ROOT / "data" / "cache" / "decklists"


def decklist_ids_from_matches(matches: list[dict]) -> set[str]:
    ids: set[str] = set()
    for match in matches:
        for comp in match.get("Competitors") or []:
            for deck in comp.get("Decklists") or []:
                if deck.get("DecklistId"):
                    ids.add(deck["DecklistId"])
    return ids


def fetch_tournament(tournament_id: int, *, refresh: bool = False) -> dict:
    out_path = RAW / f"tournament_{tournament_id}.json"
    if out_path.exists() and not refresh:
        print(f"[{tournament_id}] cached -> {out_path.name}")
        return json.loads(out_path.read_text())

    meta = melee.get_tournament(tournament_id)
    print(f"[{tournament_id}] {meta.name} ({meta.start_date}) — {len(meta.rounds)} rounds")

    rounds_out = []
    for rnd in meta.rounds:
        matches = melee.get_round_matches(tournament_id, rnd.id)
        print(f"  {rnd.name:<16} {len(matches):>4} matches"
              f"{'' if rnd.is_swiss else '  (playoff)'}")
        rounds_out.append({
            "round_id": rnd.id,
            "round_name": rnd.name,
            "is_swiss": rnd.is_swiss,
            "matches": matches,
        })

    # Standings from the last swiss round give each player's final swiss record.
    swiss = [r for r in meta.rounds if r.is_swiss]
    final_round = swiss[-1] if swiss else meta.rounds[-1]
    standings = melee.get_standings(tournament_id, final_round.id)
    print(f"  standings ({final_round.name}): {len(standings)} players")

    payload = {
        "tournament_id": tournament_id,
        "name": meta.name,
        "start_date": meta.start_date,
        "url": meta.url,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds": rounds_out,
        "standings": standings,
        "standings_round": final_round.name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"[{tournament_id}] wrote {out_path.name} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return payload


def fetch_decklists(deck_ids: set[str], workers: int = 8) -> None:
    """Fetch every uncached decklist.

    Each decklist is its own ~110KB page request, so a large event is a few
    thousand round trips and the job is entirely latency-bound. Running them
    concurrently turns ~35 minutes into ~5; melee serves these as ordinary
    static-ish pages and 8 in flight is well within what a browser opens.
    """
    DECK_CACHE.mkdir(parents=True, exist_ok=True)
    todo = [d for d in sorted(deck_ids) if not (DECK_CACHE / f"{d}.json").exists()]
    print(f"decklists: {len(deck_ids)} referenced, {len(todo)} to fetch "
          f"({workers} workers)", flush=True)
    if not todo:
        return

    done = 0
    failures: list[str] = []
    started = time.time()

    def one(deck_id: str) -> tuple[str, dict | None]:
        try:
            return deck_id, melee.get_decklist(deck_id)
        except melee.MeleeError:
            return deck_id, None

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for deck_id, cards in pool.map(one, todo):
            done += 1
            if cards is None:
                failures.append(deck_id)
            else:
                (DECK_CACHE / f"{deck_id}.json").write_text(
                    json.dumps(cards, ensure_ascii=False))
            if done % 200 == 0 or done == len(todo):
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{len(todo)}  {rate:.1f}/s  "
                      f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:.1f}m",
                      flush=True)

    if failures:
        print(f"decklists: {len(failures)} failed, retrying serially")
        for deck_id in failures:
            try:
                cards = melee.get_decklist(deck_id)
            except melee.MeleeError as exc:
                print(f"  !! {deck_id}: {exc}")
                continue
            (DECK_CACHE / f"{deck_id}.json").write_text(
                json.dumps(cards, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tournament_ids", nargs="*", type=int)
    ap.add_argument("--config", type=pathlib.Path,
                    help="JSON file with a list of {id, label} entries")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch tournaments even if cached")
    ap.add_argument("--skip-decklists", action="store_true")
    args = ap.parse_args()

    ids = list(args.tournament_ids)
    if args.config:
        ids += [entry["id"] for entry in json.loads(args.config.read_text())]
    if not ids:
        ap.error("give at least one tournament id (or --config)")

    all_deck_ids: set[str] = set()
    for tid in ids:
        payload = fetch_tournament(tid, refresh=args.refresh)
        for rnd in payload["rounds"]:
            all_deck_ids |= decklist_ids_from_matches(rnd["matches"])

    if not args.skip_decklists:
        fetch_decklists(all_deck_ids)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
