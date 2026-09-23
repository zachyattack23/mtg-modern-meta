#!/usr/bin/env python3
"""Classify every deck, compute the stats, and emit the dashboard payload.

    python3 scripts/build_dataset.py --rules rules.json --out data/processed/dashboard.json

Classification order is: overrides file, then card rules, then Unclassified.
The Unclassified count is printed as a quality metric -- if it climbs above a
few percent the rules need work, it is not a bucket to shrug at.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mtgmeta import archetype, stats  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DECK_CACHE = ROOT / "data" / "cache" / "decklists"


def load_overrides(path: pathlib.Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    with path.open() as fh:
        return {row["decklist_id"]: row["archetype"]
                for row in csv.DictReader(fh)
                if row.get("decklist_id") and row.get("archetype")}


def classify_all(ruleset: archetype.Ruleset, overrides: dict[str, str]
                 ) -> tuple[dict[str, str], dict[str, str | None], dict]:
    """-> (decklist_id -> archetype, archetype -> parent, diagnostics)"""
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

    archetype_of: dict[str, str] = {}
    parents: dict[str, str | None] = {}
    agree = disagree = unlabelled_hits = 0
    disagreements: list[tuple[str, str, str]] = []
    missing = 0

    for did, name in names.items():
        if did in overrides:
            archetype_of[did] = overrides[did]
            parents.setdefault(overrides[did], None)
            continue
        cache = DECK_CACHE / f"{did}.json"
        if not cache.exists():
            missing += 1
            continue
        cards = json.loads(cache.read_text())
        main = archetype.cards_to_counts(cards["main"])
        side = archetype.cards_to_counts(cards["side"])
        label, parent = ruleset.classify(main, side)
        archetype_of[did] = label
        parents[label] = parent

        weak = archetype.label_from_name(name)
        if weak:
            if weak.lower() in label.lower() or label.lower() in weak.lower():
                agree += 1
            else:
                disagree += 1
                if len(disagreements) < 40:
                    disagreements.append((did, weak, label))
        elif label != archetype.UNCLASSIFIED:
            unlabelled_hits += 1

    counts = collections.Counter(archetype_of.values())
    diagnostics = {
        "n_decklists": len(archetype_of),
        "missing_decklists": missing,
        "unclassified": counts.get(archetype.UNCLASSIFIED, 0),
        "unclassified_pct": counts.get(archetype.UNCLASSIFIED, 0)
        / max(len(archetype_of), 1),
        "weak_label_agree": agree,
        "weak_label_disagree": disagree,
        "weak_label_agreement_rate": agree / max(agree + disagree, 1),
        "rescued_from_autofill": unlabelled_hits,
        "disagreements": disagreements,
        "overrides_applied": len(overrides),
    }
    return archetype_of, parents, diagnostics


def build(archetype_of: dict[str, str], *, merge_variants: bool,
          parents: dict[str, str | None], min_pilots: int,
          min_share_for_ranking: float, half_life_days: float | None,
          reference_date: str, min_event_players: int = 0) -> dict:
    mapping = dict(archetype_of)
    if merge_variants:
        mapping = {did: (parents.get(name) or name)
                   for did, name in archetype_of.items()}

    tournaments = []
    matches: list[stats.MatchRecord] = []
    for path in sorted(RAW.glob("tournament_*.json")):
        payload = json.loads(path.read_text())
        # Event tier matters for the ceiling specifically. Pooling 40-player
        # local RCQs with a 1500-player Regional Championship widens the pilot
        # pool, which the model reads as a wider latent skill spread and so
        # raises every deck's P90. Part of that is real skill diversity, but it
        # also changes what the number means: a ceiling measured partly against
        # weak local fields is not the ceiling a strong player faces.
        if len(payload["standings"]) < min_event_players:
            continue
        recs = stats.extract_matches(payload, mapping, swiss_only=True,
                                     reference_date=reference_date,
                                     half_life_days=half_life_days)
        matches.extend(recs)
        tournaments.append({
            "id": payload["tournament_id"], "name": payload["name"],
            "date": payload["start_date"], "url": payload["url"],
            "players": len(payload["standings"]), "matches": len(recs),
            "weight": round(stats.recency_weight(
                payload.get("start_date"), reference_date, half_life_days), 4),
        })

    # Normalise before any fitting: see stats.normalize_weights -- uniform
    # down-weighting otherwise inflates kappa and flattens every ceiling.
    stats.normalize_weights(matches)

    # Metagame share is counted by distinct pilot-decks, not by matches, so a
    # deck that kept winning does not look more popular than it was.
    deck_entries = collections.Counter(mapping.values())
    total_decks = sum(deck_entries.values())

    rates = stats.deck_win_rates(matches)
    spreads = {s.deck: s for s in stats.deck_spreads(matches,
                                                     min_pilots=min_pilots)}

    decks = []
    for deck, n_entries in deck_entries.items():
        rate = rates.get(deck, {})
        spread = spreads.get(deck)
        share = n_entries / max(total_decks, 1)
        decided = rate.get("wins", 0) + rate.get("losses", 0)
        lo, hi = stats.wilson(rate.get("wins", 0), decided)
        decks.append({
            "deck": deck,
            "parent": parents.get(deck),
            "entries": n_entries,
            "share": share,
            "matches": rate.get("matches", 0),
            "wins": rate.get("wins", 0), "losses": rate.get("losses", 0),
            "draws": rate.get("draws", 0),
            "win_rate": rate.get("match_win_rate"),
            "raw_win_rate": rate.get("raw_win_rate"),
            "effective_matches": rate.get("effective_matches"),
            "win_rate_ci": [lo, hi],
            "game_win_rate": rate.get("game_win_rate"),
            "n_pilots": spread.n_pilots if spread else None,
            "ceiling": spread.ceiling_p90 if spread else None,
            "ceiling_ci": list(spread.ceiling_ci) if spread else None,
            "floor": spread.floor_p10 if spread else None,
            "mu": spread.mu if spread else None,
            "kappa": spread.kappa if spread else None,
            "spread": spread.spread if spread else None,
            "skill_expression": spread.skill_expression if spread else None,
            "observed_p90": spread.observed_p90 if spread else None,
            "effective_pilot_matches": spread.effective_matches if spread else None,
            "rankable": bool(spread) and share >= min_share_for_ranking,
        })
    decks.sort(key=lambda d: -d["entries"])

    ranked = [d["deck"] for d in decks if d["rankable"]]
    matrix = stats.matchup_matrix(matches, ranked)
    matrix_out = [
        {"deck": a, "opp": b, **{k: v for k, v in cell.items()}}
        for (a, b), cell in matrix.items()
    ]

    # Per-pilot observed rates power the distribution plot in the drilldown.
    per_pilot = stats.pilot_records(matches)
    pilots = {
        deck: sorted(round(w / n, 4) for w, n, raw in recs.values()
                     if raw >= 3 and n > 0)
        for deck, recs in per_pilot.items() if deck in ranked
    }

    field_shares = {d["deck"]: d["share"] for d in decks if d["rankable"]}
    for entry in decks:
        if entry["rankable"]:
            entry["vs_field"] = stats.expected_vs_field(
                matrix, entry["deck"], field_shares)

    return {
        "tournaments": tournaments,
        "decks": decks,
        "matrix": matrix_out,
        "pilots": pilots,
        "total_matches": len(matches),
        "total_decks": total_decks,
        "merge_variants": merge_variants,
        "half_life_days": half_life_days,
        "effective_matches": stats.effective_n(m.weight for m in matches),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", type=pathlib.Path, default=ROOT / "rules.json")
    ap.add_argument("--overrides", type=pathlib.Path,
                    default=ROOT / "archetype_overrides.csv")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "data" / "processed" / "dashboard.json")
    ap.add_argument("--min-pilots", type=int, default=5)
    ap.add_argument("--min-share", type=float, default=0.005,
                    help="below this metagame share a deck is shown but not ranked")
    ap.add_argument("--half-life", type=float, default=stats.DEFAULT_HALF_LIFE_DAYS,
                    help="recency half-life in days; 0 disables decay")
    ap.add_argument("--min-event-players", type=int, default=0,
                    help="ignore events smaller than this (tier filter)")
    ap.add_argument("--reference-date", default=None,
                    help="date decay counts back from (default: newest event)")
    args = ap.parse_args()

    half_life = args.half_life or None
    reference = args.reference_date or max(
        (json.loads(p.read_text()).get("start_date") or "")
        for p in RAW.glob("tournament_*.json")) or None

    ruleset = archetype.Ruleset.load(args.rules)
    overrides = load_overrides(args.overrides)
    archetype_of, parents, diag = classify_all(ruleset, overrides)

    print(f"classified {diag['n_decklists']} decklists")
    print(f"  unclassified        {diag['unclassified']:>5} "
          f"({diag['unclassified_pct']:.1%})")
    print(f"  agree w/ player name {diag['weak_label_agree']:>4} / "
          f"{diag['weak_label_agree'] + diag['weak_label_disagree']} "
          f"({diag['weak_label_agreement_rate']:.1%})")
    print(f"  rescued from autofill {diag['rescued_from_autofill']:>4}")
    if diag["missing_decklists"]:
        print(f"  !! missing decklists {diag['missing_decklists']}")

    payload = {
        "fine": build(archetype_of, merge_variants=False, parents=parents,
                      min_pilots=args.min_pilots,
                      min_share_for_ranking=args.min_share,
                      half_life_days=half_life, reference_date=reference,
                      min_event_players=args.min_event_players),
        "merged": build(archetype_of, merge_variants=True, parents=parents,
                        min_pilots=args.min_pilots,
                        min_share_for_ranking=args.min_share,
                        half_life_days=half_life, reference_date=reference,
                        min_event_players=args.min_event_players),
        "diagnostics": diag,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"\n  recency half-life: "
          f"{half_life or 'off'} days, reference {reference}")
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {payload['merged']['total_matches']} matches, "
          f"{len(payload['merged']['decks'])} archetypes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
