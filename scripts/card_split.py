#!/usr/bin/env python3
"""Win rate of one deck split by how many copies of one card its lists run.

    python3 scripts/card_split.py "Mono-Green Broodscale" "Sire of Seven Deaths"
    python3 scripts/card_split.py "Mono-Green Broodscale" Trinisphere --events 411350 459699
    python3 scripts/card_split.py "Mono-Green Broodscale" Trinisphere --vs "Izzet Prowess"

Mirrors are excluded and draws dropped, as in stats.deck_win_rates. The
with/without z-test is a two-proportion test on decided matches; it ignores
that matches cluster within pilots, so treat |z| < 2.5 as nothing. The
power note says the smallest difference the two groups could have shown.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mtgmeta import archetype, stats  # noqa: E402
from build_dataset import load_overrides  # noqa: E402
from compare_events import classify_event  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DECK_CACHE = ROOT / "data" / "cache" / "decklists"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("card")
    ap.add_argument("--vs", action="append", default=[],
                    help="also split the record against this opponent (repeatable)")
    ap.add_argument("--events", type=int, nargs="*",
                    help="tournament ids (default: every fetched event of --min-event-players)")
    ap.add_argument("--min-event-players", type=int, default=100)
    ap.add_argument("--merged", action="store_true", help="deck names at family level")
    args = ap.parse_args()

    ruleset = archetype.Ruleset.load(ROOT / "rules.json")
    overrides = load_overrides(ROOT / "archetype_overrides.csv")

    paths = sorted(RAW.glob("tournament_*.json"))
    if args.events:
        paths = [RAW / f"tournament_{t}.json" for t in args.events]

    copies: dict[str, tuple[int, int]] = {}            # decklist -> (main, side)
    lists: collections.Counter = collections.Counter()  # total copies -> n lists
    overall: dict = collections.defaultdict(collections.Counter)
    versus: dict = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    opp_counts: collections.Counter = collections.Counter()
    n_events = 0

    for path in paths:
        payload = json.loads(path.read_text())
        if not args.events and len(payload["standings"]) < args.min_event_players:
            continue
        n_events += 1
        arch = classify_event(payload, ruleset, overrides, args.merged)
        deck_of: dict[int, str] = {}
        for rnd in payload["rounds"]:
            for match in rnd["matches"]:
                for comp in match.get("Competitors") or []:
                    did = (comp.get("Decklists") or [{}])[0].get("DecklistId")
                    players = comp["Team"]["Players"]
                    if did and players:
                        deck_of[players[0]["ID"]] = did
        for did, name in arch.items():
            if name == args.deck and did not in copies:
                cards = json.loads((DECK_CACHE / f"{did}.json").read_text())
                mc = sum(q for q, c in cards["main"] if c == args.card)
                sc = sum(q for q, c in cards["side"] if c == args.card)
                copies[did] = (mc, sc)
                lists[mc + sc] += 1
        fallback = stats.player_decklists(payload)
        for m in stats.extract_matches(payload, arch, half_life_days=None):
            if m.deck_a == m.deck_b:
                continue
            for me, opp, pid, gw, gl in (
                (m.deck_a, m.deck_b, m.player_a, m.game_wins_a, m.game_wins_b),
                (m.deck_b, m.deck_a, m.player_b, m.game_wins_b, m.game_wins_a),
            ):
                if me != args.deck or gw == gl:
                    continue
                did = deck_of.get(pid) or fallback.get(pid)
                if did not in copies:
                    continue
                k = sum(copies[did])
                key = "w" if gw > gl else "l"
                overall[k][key] += 1
                versus[opp][k][key] += 1
                opp_counts[opp] += 1

    n_lists = sum(lists.values())
    if not n_lists:
        print(f"no {args.deck} lists found")
        return 1
    with_n = n_lists - lists[0]
    main_n = sum(1 for mc, sc in copies.values() if mc)
    side_n = sum(1 for mc, sc in copies.values() if sc and not mc)
    print(f"{args.deck}: {n_lists} lists across {n_events} events")
    print(f"  {args.card}: {with_n} lists ({with_n / n_lists:.0%}) run it; "
          f"{main_n} main, {side_n} side only; copies "
          f"{dict(sorted(lists.items()))}")

    def rate(c: collections.Counter) -> str:
        n = c["w"] + c["l"]
        return f"{c['w']:>4}-{c['l']:<4} {c['w'] / n:>6.1%}" if n else f"{'':>4} {'-':<4} {'':>6}"

    def ztest(a: collections.Counter, b: collections.Counter) -> str:
        n1, n2 = a["w"] + a["l"], b["w"] + b["l"]
        if not n1 or not n2:
            return ""
        p1, p2 = a["w"] / n1, b["w"] / n2
        pp = (a["w"] + b["w"]) / (n1 + n2)
        se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        mde = 2.8 * math.sqrt(0.25 * (1 / n1 + 1 / n2))  # 80% power, two-sided .05
        return (f"with minus without {p1 - p2:+.1%}, z={(p1 - p2) / se:+.2f}; "
                f"smallest detectable gap ~{mde:.0%}")

    def split(table: dict, title: str) -> None:
        print(f"\n{title}")
        print(f"  {'copies':>6} {'lists':>6}   {'W-L':>9} {'WR':>6}")
        for k in sorted(lists):
            print(f"  {k:>6} {lists[k]:>6}   {rate(table[k])}")
        with_ = collections.Counter()
        for k in lists:
            if k:
                with_.update(table[k])
        print(f"  {'any':>6} {with_n:>6}   {rate(with_)}")
        print(f"  {ztest(with_, table[0])}")

    split(overall, "All non-mirror matches")
    opps = args.vs or [o for o, _ in opp_counts.most_common(4)]
    for opp in opps:
        split(versus[opp], f"vs {opp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
