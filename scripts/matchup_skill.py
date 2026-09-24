#!/usr/bin/env python3
"""How much does pilot skill decide each matchup?

A mirror is the cleanest natural experiment in the dataset: both players hold
the same deck, so the deck term cancels exactly and anything above 50% is the
pilot. Non-mirror matchups mix deck and pilot, so the deck's own edge has to be
conditioned out before the skill signal means anything.

The skill proxy has to come from outside the match being predicted or the test
is circular. Two are offered:

  nonmirror -- the player's record over their NON-MIRROR matches only. This is
            the default. It is independent of every mirror outcome, so it
            avoids the leave-one-out artifact below, and it keeps the whole
            sample because mirrors are only 6% of matches.
  loo    -- the player's record over all their OTHER matches. BIASED and kept
            only as a demonstration: removing the match subtracts a win from
            the winner and a loss from the loser, pushing their records in
            opposite directions. It scores 45.2% pooled, below chance, which is
            impossible for an honest predictor and is how the bug was caught.
  cross  -- the player's record in OTHER TOURNAMENTS entirely. Clean, but only
            exists for players who attended more than one event.

SWISS PAIRING IS A TRAP HERE, and it is the reason this analysis is restricted
to early rounds. Swiss pairs players on equal records. A mirror therefore joins
two players whose records match, so if one has a weak non-mirror record the
difference must have come from winning mirrors. Pairing conditions on a
collider (record), which induces a NEGATIVE association between non-mirror
skill and mirror skill. The gradient is unmistakable: the better pilot wins
55.1% of round-1 mirrors (pairing random), 47.4% by rounds 3-5 and 44.1% from
round 6 (p=0.028) -- below chance, which no honest predictor can be. Only
rounds 1-2 are usable, and that discards about 80% of the mirrors.

Both proxies are noisy -- the median player has 8 matches -- and a noisy
predictor attenuates the measured effect toward 50%. `--calibrate` simulates a
world with known skill SD so an observed accuracy can be read against what the
design could have produced.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from player_skill import load_matches  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def build_proxies(rows):
    """-> totals overall, per-event, and over non-mirror matches only."""
    tot = collections.defaultdict(lambda: [0, 0])          # wins, matches
    per_ev = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    nonmir = collections.defaultdict(lambda: [0, 0])
    for wp, lp, wd, ld, ev in rows:
        tot[wp][0] += 1; tot[wp][1] += 1
        tot[lp][1] += 1
        per_ev[wp][ev][0] += 1; per_ev[wp][ev][1] += 1
        per_ev[lp][ev][1] += 1
        if wd != ld:
            nonmir[wp][0] += 1; nonmir[wp][1] += 1
            nonmir[lp][1] += 1
    return tot, per_ev, nonmir


def skill_for(player, won, event, tot, per_ev, nonmir, mode, min_n):
    """Skill estimate that excludes the match being predicted."""
    if mode == "nonmirror":
        w, n = nonmir[player]
    elif mode == "cross":
        w = tot[player][0] - per_ev[player][event][0]
        n = tot[player][1] - per_ev[player][event][1]
    else:                                                   # leave-one-out (biased)
        w = tot[player][0] - (1 if won else 0)
        n = tot[player][1] - 1
    if n < min_n:
        return None
    return w / n


def analyse(rows, mode, min_n, min_matches, gap):
    tot, per_ev, nonmir = build_proxies(rows)
    buckets = collections.defaultdict(lambda: [0, 0])       # correct, usable
    for wp, lp, wd, ld, ev in rows:
        if wd != ld:
            continue                                        # mirrors only here
        sw = skill_for(wp, True, ev, tot, per_ev, nonmir, mode, min_n)
        sl = skill_for(lp, False, ev, tot, per_ev, nonmir, mode, min_n)
        if sw is None or sl is None or abs(sw - sl) < gap:
            continue
        b = buckets[wd]
        b[1] += 1
        if sw > sl:
            b[0] += 1
    out = []
    for deck, (c, n) in buckets.items():
        if n < min_matches:
            continue
        lo, hi = sps.beta.ppf([0.025, 0.975], c + 0.5, n - c + 0.5)
        out.append({"deck": deck, "n": n, "acc": c / n, "lo": float(lo),
                    "hi": float(hi),
                    "p": float(sps.binomtest(c, n, 0.5).pvalue)})
    out.sort(key=lambda r: -r["acc"])
    return out


def calibrate(true_sd_pts, n_matches, matches_per_player, reps=400, seed=0):
    """Accuracy this design would show if skill SD really were `true_sd_pts`."""
    rng = np.random.default_rng(seed)
    sd = true_sd_pts / 100 * 4                              # points -> logit
    hits = tries = 0
    for _ in range(reps):
        # two players, their true skills, and noisy observed records
        a, b = rng.normal(0, sd, 2)
        pa = 1 / (1 + math.exp(-(a - b)))
        winner_is_a = rng.random() < pa
        ra = rng.binomial(matches_per_player, 1 / (1 + math.exp(-a))) / matches_per_player
        rb = rng.binomial(matches_per_player, 1 / (1 + math.exp(-b))) / matches_per_player
        if ra == rb:
            continue
        tries += 1
        if (ra > rb) == winner_is_a:
            hits += 1
    return hits / tries if tries else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nonmirror", "loo", "cross"],
                    default="nonmirror")
    ap.add_argument("--min-event-players", type=int, default=100)
    ap.add_argument("--min-proxy-matches", type=int, default=5)
    ap.add_argument("--min-matches", type=int, default=40)
    ap.add_argument("--gap", type=float, default=0.0,
                    help="only use matches where the skill proxies differ by this much")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    rows = load_matches(args.min_event_players, False)
    res = analyse(rows, args.mode, args.min_proxy_matches, args.min_matches, args.gap)

    print(f"proxy={args.mode}  gap>={args.gap:.0%}  "
          f"(mirrors only: the deck cancels, so >50% is pilot skill)\n")
    print(f"{'mirror':<26}{'n':>6}{'better pilot wins':>20}{'95% CI':>16}{'p':>8}")
    for r in res:
        print(f"  {r['deck'][:24]:<26}{r['n']:>6}{r['acc']:>19.1%}"
              + f"{r['lo']:.0%}-{r['hi']:.0%}".rjust(16)
              + f"{r['p']:>8.3f}")
    if res:
        c = sum(round(r["acc"] * r["n"]) for r in res); n = sum(r["n"] for r in res)
        lo, hi = sps.beta.ppf([0.025, 0.975], c + 0.5, n - c + 0.5)
        print(f"\n  ALL MIRRORS POOLED{'':<8}{n:>6}{c/n:>19.1%}"
              f"{f'{lo:.0%}-{hi:.0%}':>16}{sps.binomtest(c,n,0.5).pvalue:>8.3f}")

    if args.calibrate:
        print("\n=== what this design would show, for a known true skill SD ===")
        print(f"{'true skill SD':>16}{'expected accuracy':>20}")
        for sd in (0, 2, 4.7, 8, 12, 20):
            print(f"{sd:>15.1f}p{calibrate(sd, 0, 8):>19.1%}")
        print("  (8-match proxy, the median here. Noise pulls everything toward 50%.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
