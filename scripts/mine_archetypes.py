#!/usr/bin/env python3
"""Mine candidate archetype signatures from the fetched decklists.

Output is a human-reading report, not a finished classifier. The point is to
see which cards actually separate the clusters in *this* metagame before any
rules get written, rather than hand-authoring a remembered card pool.

    python3 scripts/mine_archetypes.py             # report to stdout
    python3 scripts/mine_archetypes.py --json rules_draft.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mtgmeta import archetype  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DECK_CACHE = ROOT / "data" / "cache" / "decklists"


def load_decks() -> list[dict]:
    """Every distinct decklist seen across all tournaments, with its weak label."""
    names: dict[str, str] = {}
    for path in sorted(RAW.glob("tournament_*.json")):
        payload = json.loads(path.read_text())
        for rnd in payload["rounds"]:
            for match in rnd["matches"]:
                for comp in match.get("Competitors") or []:
                    for deck in comp.get("Decklists") or []:
                        did = deck.get("DecklistId")
                        if did:
                            names.setdefault(did, deck.get("DecklistName") or "")

    decks: list[dict] = []
    missing = 0
    for did, name in names.items():
        cache = DECK_CACHE / f"{did}.json"
        if not cache.exists():
            missing += 1
            continue
        cards = json.loads(cache.read_text())
        main = archetype.cards_to_counts(cards["main"])
        side = archetype.cards_to_counts(cards["side"])
        decks.append({
            "id": did,
            "name": name,
            "label": archetype.label_from_name(name),
            "main": main,
            "side": side,
            "cards": set(main),
        })
    if missing:
        print(f"!! {missing} decklists referenced but not cached "
              f"(run fetch_tournaments.py)\n", file=sys.stderr)
    return decks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cluster", type=int, default=5)
    ap.add_argument("--min-inside", type=float, default=0.70)
    ap.add_argument("--max-outside", type=float, default=0.15)
    ap.add_argument("--json", type=pathlib.Path,
                    help="write a draft ruleset for hand-editing")
    args = ap.parse_args()

    decks = load_decks()
    labelled = [d for d in decks if d["label"]]
    print(f"{len(decks)} decklists, {len(labelled)} with a usable player name "
          f"({len(labelled) / max(len(decks), 1):.0%})\n")

    raw_names = collections.Counter(d["name"] for d in decks)
    useless = [n for n in raw_names if not archetype.label_from_name(n)]
    print(f"{len(raw_names)} distinct name strings; "
          f"{len(useless)} carry no archetype information "
          f"(melee colour auto-fill)\n")

    sigs = archetype.mine_signatures(
        decks, min_cluster=args.min_cluster, min_inside=args.min_inside,
        max_outside=args.max_outside)

    print(f"{'=' * 78}\nCANDIDATE SIGNATURES\n{'=' * 78}")
    for sig in sigs:
        print(f"\n{sig.label}  ({sig.n_decks} decks)")
        for card, s_in, s_out, lift in sig.cards:
            print(f"    {card:<40} in={s_in:5.0%} out={s_out:5.0%} lift={lift:7.1f}")

    unlabelled = [d for d in decks if not d["label"]]
    print(f"\n{'=' * 78}\n{len(unlabelled)} decks need card-based classification\n{'=' * 78}")

    if args.json:
        draft = {"rules": [
            {
                "name": sig.label.title(),
                "parent": None,
                "all_of": {sig.cards[0][0]: 1} if sig.cards else {},
                "any_of": [c for c, *_ in sig.cards[1:5]],
                "any_min": 1,
                "none_of": [],
                "main_only": True,
                "priority": 100,
                "notes": f"mined from {sig.n_decks} weak-labelled decks",
            }
            for sig in sigs
        ]}
        args.json.write_text(json.dumps(draft, indent=2, ensure_ascii=False))
        print(f"\ndraft ruleset -> {args.json} (review and edit before use)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
