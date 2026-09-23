#!/usr/bin/env python3
"""Separate deck strength from pilot quality.

Every win rate in this project is confounded: pilots choose their decks, so a
deck with a high win rate may be strong, or may simply have attracted strong
players. This fits both effects at once instead of assuming one away.

    P(A beats B) = logistic( s_A - s_B + d_deck(A) - d_deck(B) )

s is a per-player skill term, d a per-deck term, both on the logit scale, both
ridge-penalised (a Gaussian prior) because most players appear in a single
event. The deck effects that survive are what is left after the model has
already credited the pilots.

Where the identification comes from is worth stating plainly, because it is the
weak point. Within one event a player plays one deck, so player and deck are
perfectly collinear there and that event alone can never separate them. Two
things break the tie:

  * players who appear across events on DIFFERENT decks -- the direct evidence,
    and the script reports how many there are;
  * mirror matches, where the deck terms cancel exactly and the outcome is pure
    player skill. These are excluded from every other analysis here and are
    deliberately included in this one.

If the deck-switcher count is small, the deck effects lean on the prior and the
result should be read as directional.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np
from scipy import optimize, sparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mtgmeta import archetype  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DECK_CACHE = ROOT / "data" / "cache" / "decklists"


def load_matches(min_event_players: int, merge_variants: bool):
    ruleset = archetype.Ruleset.load(ROOT / "rules.json")
    arch: dict[str, str] = {}
    for path in sorted(RAW.glob("tournament_*.json")):
        payload = json.loads(path.read_text())
        for rnd in payload["rounds"]:
            for match in rnd["matches"]:
                for comp in match.get("Competitors") or []:
                    for deck in comp.get("Decklists") or []:
                        did = deck.get("DecklistId")
                        if not did or did in arch:
                            continue
                        cache = DECK_CACHE / f"{did}.json"
                        if not cache.exists():
                            continue
                        raw = json.loads(cache.read_text())
                        main = archetype.cards_to_counts(raw["main"])
                        side = archetype.cards_to_counts(raw["side"])
                        name, parent = ruleset.classify(main, side)
                        arch[did] = (parent or name) if merge_variants else name

    rows = []
    for path in sorted(RAW.glob("tournament_*.json")):
        payload = json.loads(path.read_text())
        if len(payload["standings"]) < min_event_players:
            continue
        event = payload["tournament_id"]
        for rnd in payload["rounds"]:
            if not rnd["is_swiss"]:
                continue
            for match in rnd["matches"]:
                comps = match.get("Competitors") or []
                if len(comps) != 2:
                    continue
                ids = [(c.get("Decklists") or [{}])[0].get("DecklistId") for c in comps]
                players = [c["Team"]["Players"] for c in comps]
                if any(i not in arch for i in ids) or not all(players):
                    continue
                gw = [c.get("GameWins") or 0 for c in comps]
                if gw[0] == gw[1]:
                    continue
                names = [(p[0].get("Username") or "").strip().lower() for p in players]
                if not all(names) or any(n == "n/a" for n in names):
                    continue
                win = 0 if gw[0] > gw[1] else 1
                rows.append((names[win], names[1 - win],
                             arch[ids[win]], arch[ids[1 - win]], event))
    return rows


def fit(rows, sd_player: float, sd_deck: float):
    players = sorted({p for r in rows for p in r[:2]})
    decks = sorted({d for r in rows for d in r[2:4]})
    pi = {p: i for i, p in enumerate(players)}
    di = {d: i for i, d in enumerate(decks)}
    nP, nD, n = len(players), len(decks), len(rows)

    # One row per match: +1 winner, -1 loser, for players and for decks.
    ri, ci, vi = [], [], []
    for k, (wp, lp, wd, ld, _) in enumerate(rows):
        ri += [k, k]; ci += [pi[wp], pi[lp]]; vi += [1.0, -1.0]
        if wd != ld:
            ri += [k, k]; ci += [nP + di[wd], nP + di[ld]]; vi += [1.0, -1.0]
    X = sparse.csr_matrix((vi, (ri, ci)), shape=(n, nP + nD))

    lam_p = 1.0 / (2 * sd_player ** 2)
    lam_d = 1.0 / (2 * sd_deck ** 2)
    pen = np.concatenate([np.full(nP, lam_p), np.full(nD, lam_d)])

    def obj(theta):
        z = X @ theta
        # Every row is an observed win, so the likelihood is -log sigmoid(z).
        nll = float(np.sum(np.logaddexp(0.0, -z))) + float(np.sum(pen * theta ** 2))
        g = X.T @ (-1.0 / (1.0 + np.exp(z))) + 2 * pen * theta
        return nll, np.asarray(g).ravel()

    res = optimize.minimize(obj, np.zeros(nP + nD), jac=True, method="L-BFGS-B",
                            options={"maxiter": 800, "maxfun": 2000})
    return players, decks, res.x[:nP], res.x[nP:], res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-event-players", type=int, default=100)
    ap.add_argument("--sd-player", type=float, default=0.25,
                    help="prior SD on player skill, logit scale")
    ap.add_argument("--sd-deck", type=float, default=0.50)
    ap.add_argument("--split-variants", action="store_true")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "data" / "processed" / "player_skill.json")
    args = ap.parse_args()

    rows = load_matches(args.min_event_players, not args.split_variants)
    print(f"{len(rows)} decided matches (mirrors included -- they identify skill)")

    # Identification diagnostics.
    seen = collections.defaultdict(set)
    events = collections.defaultdict(set)
    for wp, lp, wd, ld, ev in rows:
        seen[wp].add(wd); seen[lp].add(ld)
        events[wp].add(ev); events[lp].add(ev)
    multi_event = sum(1 for v in events.values() if len(v) >= 2)
    switchers = sum(1 for v in seen.values() if len(v) >= 2)
    mirrors = sum(1 for r in rows if r[2] == r[3])
    print(f"  players: {len(seen)}   in 2+ events: {multi_event}   "
          f"on 2+ DIFFERENT decks: {switchers}")
    print(f"  mirror matches (pure skill signal): {mirrors}")
    if switchers < 100:
        print("  !! few deck-switchers: deck effects will lean on the prior")

    players, decks, s, d, res = fit(rows, args.sd_player, args.sd_deck)
    print(f"  converged={res.success} after {res.nit} iterations\n")

    # Raw (uncontrolled) win rate per deck, same match set.
    raw = collections.defaultdict(lambda: [0, 0])
    for wp, lp, wd, ld, _ in rows:
        if wd == ld:
            continue
        raw[wd][0] += 1; raw[wd][1] += 1
        raw[ld][1] += 1

    # Average opponent-adjusted pilot quality per deck.
    skill = dict(zip(players, s))
    pilots = collections.defaultdict(list)
    for wp, lp, wd, ld, _ in rows:
        pilots[wd].append(skill[wp]); pilots[ld].append(skill[lp])

    deck_eff = dict(zip(decks, d))
    centre = float(np.mean([deck_eff[k] for k in deck_eff]))
    out = []
    for k in decks:
        if raw[k][1] < 300:
            continue
        adj = deck_eff[k] - centre
        out.append({
            "deck": k, "matches": raw[k][1],
            "raw_win_rate": raw[k][0] / raw[k][1],
            "adjusted_win_rate": 1 / (1 + math.exp(-adj)),
            "deck_effect_logit": adj,
            "mean_pilot_skill": float(np.mean(pilots[k])),
        })
    out.sort(key=lambda r: -r["adjusted_win_rate"])

    print(f"{'deck':<26}{'raw':>8}{'skill-adj':>11}{'shift':>8}"
          f"{'pilot quality':>15}")
    for r in out:
        print(f"  {r['deck'][:24]:<26}{r['raw_win_rate']:>8.1%}"
              f"{r['adjusted_win_rate']:>11.1%}"
              f"{(r['adjusted_win_rate'] - r['raw_win_rate']) * 100:>+8.1f}"
              f"{r['mean_pilot_skill']:>+15.3f}")

    sd = float(np.std(s))
    print(f"\n  fitted player-skill SD: {sd:.3f} logit "
          f"(~{(1/(1+math.exp(-sd)) - 0.5) * 100:.1f} win-rate points for a 1-SD player)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "decks": out,
        "player_skill_sd_logit": sd,
        "n_matches": len(rows), "n_players": len(players),
        "n_deck_switchers": switchers, "n_mirrors": mirrors,
        "prior_sd_player": args.sd_player, "prior_sd_deck": args.sd_deck,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
