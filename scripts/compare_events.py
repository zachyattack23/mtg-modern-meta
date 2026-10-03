#!/usr/bin/env python3
"""Compare win rates in a set of new events against a committed baseline.

The question this answers is "did deck X's win rate move at the latest
events?" without waiting for (or trusting) a full re-fit. It classifies the
new events' decklists with the same rules, computes plain unweighted win rates
per deck (mirrors excluded, draws dropped, same as stats.deck_win_rates), and
sets them beside the baseline dashboard's numbers with a two-proportion z-test.

    python3 scripts/compare_events.py 411350 459699
    python3 scripts/compare_events.py 411350 459699 --merged   # family level
    python3 scripts/compare_events.py 411350 459699 --baseline old_dashboard.json

A snapshot of an event still in progress is fine: matches without a result are
skipped, and the output says how many rounds were played.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mtgmeta import archetype, stats  # noqa: E402
from build_dataset import load_overrides  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DECK_CACHE = ROOT / "data" / "cache" / "decklists"


def classify_event(payload: dict, ruleset: archetype.Ruleset,
                   overrides: dict[str, str], merged: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for rnd in payload["rounds"]:
        for match in rnd["matches"]:
            for comp in match.get("Competitors") or []:
                for deck in comp.get("Decklists") or []:
                    did = deck.get("DecklistId")
                    if not did or did in out:
                        continue
                    if did in overrides:
                        out[did] = overrides[did]
                        continue
                    cache = DECK_CACHE / f"{did}.json"
                    if not cache.exists():
                        continue
                    cards = json.loads(cache.read_text())
                    label, parent = ruleset.classify(
                        archetype.cards_to_counts(cards["main"]),
                        archetype.cards_to_counts(cards["side"]))
                    out[did] = (parent or label) if merged else label
    return out


def two_prop_z(w1: float, n1: float, w2: float, n2: float) -> float | None:
    if not n1 or not n2:
        return None
    p1, p2, p = w1 / n1, w2 / n2, (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tournament_ids", nargs="+", type=int)
    ap.add_argument("--baseline", type=pathlib.Path,
                    default=ROOT / "data" / "processed" / "dashboard.json")
    ap.add_argument("--merged", action="store_true",
                    help="compare at deck-family level (Broodscale, Blink...)")
    ap.add_argument("--top", type=int, default=15,
                    help="how many of the baseline's most-played decks to show")
    ap.add_argument("--rules", type=pathlib.Path, default=ROOT / "rules.json")
    ap.add_argument("--overrides", type=pathlib.Path,
                    default=ROOT / "archetype_overrides.csv")
    ap.add_argument("--json", type=pathlib.Path, help="also write rows here")
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text())
    view = baseline["merged" if args.merged else "fine"]
    base = {d["deck"]: d for d in view["decks"]}
    base_ids = {t["id"] for t in view["tournaments"]}

    ruleset = archetype.Ruleset.load(args.rules)
    overrides = load_overrides(args.overrides)

    per_event: dict[int, dict] = {}
    pooled: list[stats.MatchRecord] = []
    entries: collections.Counter = collections.Counter()
    for tid in args.tournament_ids:
        path = RAW / f"tournament_{tid}.json"
        if not path.exists():
            print(f"!! {tid}: not fetched (run fetch_tournaments.py first)")
            continue
        payload = json.loads(path.read_text())
        if tid in base_ids:
            print(f"!! {tid} is already in the baseline; the comparison "
                  f"would double count it")
        mapping = classify_event(payload, ruleset, overrides, args.merged)
        recs = stats.extract_matches(payload, mapping, swiss_only=True,
                                     half_life_days=None)
        pooled.extend(recs)
        entries.update(mapping.values())
        per_event[tid] = {
            "name": payload["name"], "date": payload["start_date"],
            "players": len(payload["standings"]),
            "rounds_played": payload.get("rounds_played"),
            "swiss_rounds": payload.get("swiss_rounds"),
            "complete": payload.get("complete", True),
            "matches": len(recs),
            "rates": stats.deck_win_rates(recs),
            "entries": collections.Counter(mapping.values()),
        }

    if not pooled:
        return 1

    new_rates = stats.deck_win_rates(pooled)
    total_entries = sum(entries.values())

    print(f"\nBaseline: {len(base_ids)} events, {view['total_matches']} matches "
          f"(recency-weighted win rates)")
    for tid, ev in per_event.items():
        status = ("complete" if ev["complete"] else
                  f"in progress, {ev['rounds_played']}/{ev['swiss_rounds']} rounds")
        print(f"New: {tid} {ev['name'][:50]} ({ev['date']}) "
              f"{ev['players']} players, {ev['matches']} matches, {status}")

    ranked = sorted(base.values(), key=lambda d: -d["entries"])
    ranked = [d for d in ranked if d["deck"] != archetype.UNCLASSIFIED][:args.top]

    rows = []
    hdr = (f"{'deck':<24}{'base%':>6}{'new%':>6} | {'base WR':>8}{'new WR':>8}"
           f"{'delta':>7}{'z':>6} | {'new W-L':>9}  per event")
    print("\n" + hdr)
    print("-" * len(hdr))
    for d in ranked:
        deck = d["deck"]
        nr = new_rates.get(deck, {})
        w, l = nr.get("wins", 0), nr.get("losses", 0)
        new_wr = w / (w + l) if (w + l) else None
        base_wr = d["win_rate"]
        base_dec = d["wins"] + d["losses"]
        z = two_prop_z(w, w + l, d["raw_win_rate"] * base_dec, base_dec) \
            if new_wr is not None else None
        lo, hi = stats.wilson(w, w + l) if (w + l) else (None, None)
        per = []
        for tid, ev in per_event.items():
            r = ev["rates"].get(deck, {})
            ew, el = r.get("wins", 0), r.get("losses", 0)
            per.append(f"{tid}: {ew}-{el}"
                       f"{'' if not (ew + el) else f' ({ew / (ew + el):.0%})'}")
        row = {
            "deck": deck,
            "baseline_share": d["share"], "new_share": entries[deck] / total_entries,
            "baseline_win_rate": base_wr, "baseline_raw_win_rate": d["raw_win_rate"],
            "baseline_decided": base_dec,
            "new_wins": w, "new_losses": l, "new_draws": nr.get("draws", 0),
            "new_win_rate": new_wr, "new_win_rate_ci": [lo, hi],
            "delta": (new_wr - base_wr) if new_wr is not None else None,
            "z": z,
            "per_event": {tid: ev["rates"].get(deck, {}) for tid, ev in per_event.items()},
        }
        rows.append(row)
        new_s = f"{new_wr:.1%}" if new_wr is not None else "-"
        delta_s = f"{row['delta']:+.1%}" if new_wr is not None else "-"
        z_s = f"{z:+.1f}" if z is not None else "-"
        print(f"{deck[:23]:<24}{d['share']:>6.1%}{row['new_share']:>6.1%} | "
              f"{base_wr:>8.1%}{new_s:>8}{delta_s:>7}{z_s:>6} | "
              f"{w}-{l:<5}  {'; '.join(per)}")

    print("\nnew WR is unweighted, mirrors excluded, draws dropped. z compares "
          "new W-L against the baseline's raw W-L; |z| > 2 is roughly p < .05.\n"
          "A single event's deck sample is small: a 20-point swing on 40 "
          "matches is within noise.")

    if args.json:
        args.json.write_text(json.dumps(
            {"events": {tid: {k: v for k, v in ev.items() if k not in ("rates", "entries")}
                        for tid, ev in per_event.items()},
             "rows": rows}, indent=1, ensure_ascii=False))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
