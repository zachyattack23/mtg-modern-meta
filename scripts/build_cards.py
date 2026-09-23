#!/usr/bin/env python3
"""Flex-slot map and card-effect tests, per archetype.

Two outputs with very different confidence levels, kept visibly separate:

  * the flex-slot map -- which slots pilots of the same deck disagree on, and
    how. Descriptive, and the part worth trusting.
  * the effect tests -- whether a choice correlates with winning. Underpowered
    by construction, FDR-corrected, and reported with the minimum detectable
    effect beside every row so a null reads correctly.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mtgmeta import archetype, cards, stats  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DECK_CACHE = ROOT / "data" / "cache" / "decklists"


def load_ignore(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return cards.DEFAULT_IGNORE
    doc = json.loads(path.read_text())
    out: set[str] = set()
    for key, value in doc.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            out |= set(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", type=pathlib.Path, default=ROOT / "rules.json")
    ap.add_argument("--ignore", type=pathlib.Path, default=ROOT / "ignore_cards.json")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "data" / "processed" / "cards.json")
    ap.add_argument("--min-decks", type=int, default=25,
                    help="archetypes with fewer pilots are skipped entirely")
    args = ap.parse_args()

    ruleset = archetype.Ruleset.load(args.rules)
    ignore = load_ignore(args.ignore)

    # Classify, then attach each pilot's record to their actual 75.
    deck_cards: dict[str, dict] = {}
    archetype_of: dict[str, str] = {}
    deck_owner: dict[str, int] = {}

    for path in sorted(RAW.glob("tournament_*.json")):
        payload = json.loads(path.read_text())
        for rnd in payload["rounds"]:
            for match in rnd["matches"]:
                for comp in match.get("Competitors") or []:
                    players = comp["Team"]["Players"]
                    for deck in comp.get("Decklists") or []:
                        did = deck.get("DecklistId")
                        if not did or did in archetype_of:
                            continue
                        cache = DECK_CACHE / f"{did}.json"
                        if not cache.exists():
                            continue
                        raw = json.loads(cache.read_text())
                        entry = {"main": archetype.cards_to_counts(raw["main"]),
                                 "side": archetype.cards_to_counts(raw["side"])}
                        deck_cards[did] = entry
                        archetype_of[did] = ruleset.classify(
                            entry["main"], entry["side"])[0]
                        if players:
                            deck_owner[did] = players[0]["ID"]

    # Per-pilot win/loss, using the same non-mirror convention as the ceiling stat.
    matches: list[stats.MatchRecord] = []
    for path in sorted(RAW.glob("tournament_*.json")):
        payload = json.loads(path.read_text())
        matches.extend(stats.extract_matches(payload, archetype_of, swiss_only=True))
    records = stats.pilot_records(matches)

    by_archetype: dict[str, list[tuple[dict, int, int]]] = collections.defaultdict(list)
    for did, arch in archetype_of.items():
        if arch == archetype.UNCLASSIFIED:
            continue
        owner = deck_owner.get(did)
        if owner is None:
            continue
        rec = records.get(arch, {}).get(owner)
        if not rec:
            continue
        by_archetype[arch].append((deck_cards[did], rec[0], rec[1]))

    out: dict[str, dict] = {}
    all_effects: list[cards.CardEffect] = []
    for arch, entries in by_archetype.items():
        if len(entries) < args.min_decks:
            continue
        slots, effects = cards.analyse_archetype(arch, entries, ignore=ignore)
        all_effects.extend(effects)
        out[arch] = {
            "n_decks": len(entries),
            "slots": [{
                "card": s.card, "zone": s.zone, "kind": s.kind,
                "inclusion": s.inclusion_rate, "mean_copies": s.mean_copies,
                "entropy": s.entropy,
                "distribution": {str(k): v for k, v in s.distribution.items()},
            } for s in slots],
        }

    # One FDR correction across every test in the whole study, not per deck.
    cards.benjamini_hochberg(all_effects)
    for eff in all_effects:
        out[eff.archetype].setdefault("effects", []).append({
            "card": eff.card, "zone": eff.zone, "split": eff.split,
            "n_pilots_a": eff.n_pilots_a, "n_pilots_b": eff.n_pilots_b,
            "matches_a": eff.matches_a, "matches_b": eff.matches_b,
            "rate_a": eff.rate_a, "rate_b": eff.rate_b, "delta": eff.delta,
            "ci_a": list(eff.ci_a), "ci_b": list(eff.ci_b),
            "p": eff.p_value, "q": eff.q_value,
            "mde": eff.min_detectable_effect, "powered": eff.powered,
        })

    survivors = [e for e in all_effects if e.q_value < 0.10]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "archetypes": out,
        "summary": {
            "n_tests": len(all_effects),
            "n_survive_fdr": len(survivors),
            "n_powered": sum(1 for e in all_effects if e.powered),
            "median_mde": float(sorted(
                e.min_detectable_effect for e in all_effects)[len(all_effects) // 2])
            if all_effects else None,
        },
    }, ensure_ascii=False))

    print(f"{len(out)} archetypes analysed, {len(all_effects)} card tests")
    print(f"  median minimum-detectable-effect: "
          f"{sorted(e.min_detectable_effect for e in all_effects)[len(all_effects)//2]:.1%}"
          if all_effects else "")
    print(f"  tests with enough power to see a 4-point effect: "
          f"{sum(1 for e in all_effects if e.powered)}")
    print(f"  survive FDR q<0.10: {len(survivors)}")
    for eff in sorted(survivors, key=lambda e: e.q_value)[:15]:
        print(f"    {eff.archetype:<20} {eff.card[:32]:<34} {eff.split:<12} "
              f"{eff.rate_a:.1%} vs {eff.rate_b:.1%} "
              f"(d={eff.delta:+.1%}, q={eff.q_value:.3f})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
