"""Win rates, matchup matrix, and the noise-corrected pilot-skill spread.

The headline question this module answers is not "what does this deck win on
average" but "how well does this deck do in the hands of a good pilot". Those
differ, and the naive version of the second one is almost pure noise: over a
15-round event a single pilot's observed win rate carries a standard error of
about 13 points, so ranking decks by their best pilot mostly ranks them by how
many pilots rolled dice. A 6-pilot deck will beat a 60-pilot deck on that
statistic nearly every time.

So instead of reading the observed spread directly we fit a hierarchical
beta-binomial per deck:

    p_i     ~ Beta(mu_d * kappa_d, (1 - mu_d) * kappa_d)     # pilot's true rate
    w_i     ~ Binomial(m_i, p_i)                             # what we observed

mu_d is the deck's mean true win rate and kappa_d is concentration: large kappa
means every pilot performs the same on the deck, small kappa means the deck's
outcome depends heavily on who is holding it. The ceiling we report is the 90th
percentile of that *latent* Beta, not of the observed rates, so binomial noise
is modelled rather than mistaken for skill.

kappa is poorly identified from a handful of pilots, so each deck's kappa is
shrunk toward a global kappa fitted across every pilot in the dataset.

One honest caveat, stated here because it also belongs in the dashboard: pilot
pools are self-selected, not randomly assigned. A deck whose latent spread is
wide may reward skill, or may simply have attracted a wider mix of players.
This data cannot separate those two.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

# Draws are ~2.5% of matches and are dropped from the binomial denominator
# rather than counted as half a win, which keeps the likelihood integral.
DRAW_POLICY = "drop"


# --------------------------------------------------------------------------
# Match records
# --------------------------------------------------------------------------

@dataclass
class MatchRecord:
    tournament_id: int
    round_name: str
    is_swiss: bool
    player_a: int
    player_b: int
    deck_a: str
    deck_b: str
    game_wins_a: int
    game_wins_b: int
    is_draw: bool

    @property
    def is_mirror(self) -> bool:
        return self.deck_a == self.deck_b


def extract_matches(tournament: dict, archetype_of: dict[str, str],
                    swiss_only: bool = True) -> list[MatchRecord]:
    """Flatten a fetched tournament into two-sided match records.

    Byes and single-competitor rows are dropped: they carry no matchup
    information and would inflate the win rate of whichever decks got them.
    """
    out: list[MatchRecord] = []
    for rnd in tournament["rounds"]:
        if swiss_only and not rnd["is_swiss"]:
            continue
        for match in rnd["matches"]:
            comps = match.get("Competitors") or []
            if len(comps) != 2:
                continue
            sides = []
            for comp in comps:
                players = comp["Team"]["Players"]
                decks = comp.get("Decklists") or []
                if not players or not decks:
                    break
                deck_id = decks[0].get("DecklistId")
                if not deck_id or deck_id not in archetype_of:
                    break
                sides.append((players[0]["ID"], archetype_of[deck_id],
                              comp.get("GameWins") or 0))
            if len(sides) != 2:
                continue

            (pa, da, ga), (pb, db, gb) = sides
            result = (match.get("ResultString") or "").lower()
            if "forfeit" in result or "bye" in result:
                continue
            is_draw = ga == gb

            out.append(MatchRecord(
                tournament_id=tournament["tournament_id"],
                round_name=rnd["round_name"], is_swiss=rnd["is_swiss"],
                player_a=pa, player_b=pb, deck_a=da, deck_b=db,
                game_wins_a=ga, game_wins_b=gb, is_draw=is_draw,
            ))
    return out


# --------------------------------------------------------------------------
# Aggregate win rates and the matchup matrix
# --------------------------------------------------------------------------

def deck_win_rates(matches: list[MatchRecord],
                   exclude_mirrors: bool = True) -> dict[str, dict]:
    """Match- and game-level win rates per deck.

    Mirrors are excluded by default. A mirror is 50% by construction, so
    including them drags every deck toward 0.5 in proportion to its own
    metagame share -- which penalises exactly the popular decks whose numbers
    we trust most.
    """
    agg: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"w": 0, "l": 0, "d": 0, "gw": 0, "gl": 0})

    for m in matches:
        if exclude_mirrors and m.is_mirror:
            continue
        for deck, opp_deck, gw, gl in (
            (m.deck_a, m.deck_b, m.game_wins_a, m.game_wins_b),
            (m.deck_b, m.deck_a, m.game_wins_b, m.game_wins_a),
        ):
            row = agg[deck]
            row["gw"] += gw
            row["gl"] += gl
            if m.is_draw:
                row["d"] += 1
            elif gw > gl:
                row["w"] += 1
            else:
                row["l"] += 1

    out: dict[str, dict] = {}
    for deck, row in agg.items():
        decided = row["w"] + row["l"]
        games = row["gw"] + row["gl"]
        out[deck] = {
            "matches": decided + row["d"],
            "wins": row["w"], "losses": row["l"], "draws": row["d"],
            "match_win_rate": row["w"] / decided if decided else float("nan"),
            "game_wins": row["gw"], "game_losses": row["gl"],
            "game_win_rate": row["gw"] / games if games else float("nan"),
        }
    return out


def matchup_matrix(matches: list[MatchRecord],
                   decks: list[str]) -> dict[tuple[str, str], dict]:
    """Head-to-head record for every ordered deck pair, including mirrors."""
    cells: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(
        lambda: {"w": 0, "l": 0, "d": 0, "gw": 0, "gl": 0})
    wanted = set(decks)

    for m in matches:
        if m.deck_a not in wanted or m.deck_b not in wanted:
            continue
        for deck, opp, gw, gl in (
            (m.deck_a, m.deck_b, m.game_wins_a, m.game_wins_b),
            (m.deck_b, m.deck_a, m.game_wins_b, m.game_wins_a),
        ):
            cell = cells[(deck, opp)]
            cell["gw"] += gw
            cell["gl"] += gl
            if m.is_draw:
                cell["d"] += 1
            elif gw > gl:
                cell["w"] += 1
            else:
                cell["l"] += 1

    out: dict[tuple[str, str], dict] = {}
    for key, cell in cells.items():
        decided = cell["w"] + cell["l"]
        games = cell["gw"] + cell["gl"]
        # Wilson interval, because most cells are small and a bare ratio
        # invites reading 3-0 as a 100% matchup.
        lo, hi = wilson(cell["w"], decided)
        out[key] = {
            "matches": decided + cell["d"],
            "wins": cell["w"], "losses": cell["l"], "draws": cell["d"],
            "win_rate": cell["w"] / decided if decided else float("nan"),
            "ci_low": lo, "ci_high": hi,
            "game_win_rate": cell["gw"] / games if games else float("nan"),
            "games": games,
        }
    return out


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; degrades gracefully at n = 0 and at 0%/100%."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


# --------------------------------------------------------------------------
# Hierarchical beta-binomial
# --------------------------------------------------------------------------

def _neg_log_lik(mu: float, kappa: float, wins: np.ndarray,
                 trials: np.ndarray) -> float:
    """Negative beta-binomial marginal log-likelihood (constants dropped)."""
    if not (1e-6 < mu < 1 - 1e-6) or kappa <= 1e-6:
        return np.inf
    a = mu * kappa
    b = (1 - mu) * kappa
    ll = (special.betaln(wins + a, trials - wins + b) - special.betaln(a, b))
    return -float(np.sum(ll))


def fit_beta_binomial(wins: np.ndarray, trials: np.ndarray,
                      kappa_prior: tuple[float, float] | None = None
                      ) -> tuple[float, float]:
    """Fit (mu, kappa) by maximum marginal likelihood.

    `kappa_prior` is (log_kappa_mean, log_kappa_sd); when given, kappa is
    shrunk toward it, which is what keeps a five-pilot deck from reporting an
    absurdly wide or absurdly narrow latent spread.
    """
    wins = np.asarray(wins, dtype=float)
    trials = np.asarray(trials, dtype=float)
    keep = trials > 0
    wins, trials = wins[keep], trials[keep]
    if len(wins) == 0:
        return (float("nan"), float("nan"))

    def objective(params: np.ndarray) -> float:
        logit_mu, log_kappa = params
        mu = 1 / (1 + math.exp(-logit_mu))
        kappa = math.exp(log_kappa)
        val = _neg_log_lik(mu, kappa, wins, trials)
        if kappa_prior is not None and np.isfinite(val):
            m, s = kappa_prior
            val += 0.5 * ((log_kappa - m) / s) ** 2
        return val

    p_hat = float(np.clip(wins.sum() / max(trials.sum(), 1), 0.02, 0.98))
    start = np.array([math.log(p_hat / (1 - p_hat)),
                      kappa_prior[0] if kappa_prior else math.log(20.0)])
    res = optimize.minimize(objective, start, method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-8,
                                     "maxiter": 2000})
    logit_mu, log_kappa = res.x
    return (1 / (1 + math.exp(-logit_mu)), math.exp(log_kappa))


@dataclass
class DeckSpread:
    deck: str
    n_pilots: int
    n_matches: int
    observed_mean: float
    observed_p90: float          # naive ceiling, shown only for contrast
    mu: float                    # shrunk mean of the latent pilot distribution
    kappa: float
    ceiling_p90: float           # the headline: P90 of the latent Beta
    floor_p10: float
    spread: float                # ceiling_p90 - floor_p10
    mu_ci: tuple[float, float] = (float("nan"), float("nan"))
    ceiling_ci: tuple[float, float] = (float("nan"), float("nan"))
    skill_expression: float = float("nan")  # spread relative to the field's


def fit_global_kappa(pilot_records: dict[str, list[tuple[int, int]]]
                     ) -> tuple[float, float]:
    """Fit one beta-binomial across every pilot, ignoring deck.

    This captures the field's overall pilot-to-pilot heterogeneity and becomes
    the prior each deck's kappa is shrunk toward.
    """
    wins, trials = [], []
    for records in pilot_records.values():
        for w, n in records:
            wins.append(w)
            trials.append(n)
    return fit_beta_binomial(np.array(wins), np.array(trials))


def pilot_records(matches: list[MatchRecord],
                  exclude_mirrors: bool = True
                  ) -> dict[str, dict[int, tuple[int, int]]]:
    """{deck: {player_id: (wins, decided_matches)}} for the beta-binomial.

    A player who switched decks between events is counted separately per deck,
    which is what we want: the unit of analysis is a pilot-deck pairing.
    """
    agg: dict[str, dict[int, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))

    for m in matches:
        if exclude_mirrors and m.is_mirror:
            continue
        if m.is_draw and DRAW_POLICY == "drop":
            continue
        for deck, player, gw, gl in (
            (m.deck_a, m.player_a, m.game_wins_a, m.game_wins_b),
            (m.deck_b, m.player_b, m.game_wins_b, m.game_wins_a),
        ):
            rec = agg[deck][player]
            rec[1] += 1
            if gw > gl:
                rec[0] += 1

    return {deck: {p: (r[0], r[1]) for p, r in players.items()}
            for deck, players in agg.items()}


def deck_spreads(matches: list[MatchRecord], *, min_pilots: int = 5,
                 min_matches_per_pilot: int = 3, kappa_sd: float = 1.5,
                 n_boot: int = 300, seed: int = 0) -> list[DeckSpread]:
    """Fit the hierarchical model and report each deck's ceiling."""
    rng = np.random.default_rng(seed)
    records = pilot_records(matches)

    # Global prior: how much do pilots vary across the whole field?
    flat = {d: list(v.values()) for d, v in records.items()}
    mu_global, kappa_global = fit_global_kappa(flat)
    log_kappa_prior = (math.log(kappa_global), kappa_sd)

    field_spread = _beta_spread(mu_global, kappa_global)

    out: list[DeckSpread] = []
    for deck, players in records.items():
        usable = [(w, n) for w, n in players.values()
                  if n >= min_matches_per_pilot]
        if len(usable) < min_pilots:
            continue
        wins = np.array([w for w, _ in usable], dtype=float)
        trials = np.array([n for _, n in usable], dtype=float)

        mu, kappa = fit_beta_binomial(wins, trials, kappa_prior=log_kappa_prior)
        a, b = mu * kappa, (1 - mu) * kappa
        ceiling = float(stats.beta.ppf(0.90, a, b))
        floor = float(stats.beta.ppf(0.10, a, b))

        obs = wins / trials
        boot_mu, boot_ceiling = [], []
        idx = np.arange(len(usable))
        for _ in range(n_boot):
            pick = rng.choice(idx, size=len(idx), replace=True)
            bmu, bkappa = fit_beta_binomial(wins[pick], trials[pick],
                                            kappa_prior=log_kappa_prior)
            if not (np.isfinite(bmu) and np.isfinite(bkappa)):
                continue
            boot_mu.append(bmu)
            boot_ceiling.append(
                float(stats.beta.ppf(0.90, bmu * bkappa, (1 - bmu) * bkappa)))

        out.append(DeckSpread(
            deck=deck,
            n_pilots=len(usable),
            n_matches=int(trials.sum()),
            observed_mean=float(wins.sum() / trials.sum()),
            observed_p90=float(np.quantile(obs, 0.90)),
            mu=mu, kappa=kappa,
            ceiling_p90=ceiling, floor_p10=floor, spread=ceiling - floor,
            mu_ci=_pct_ci(boot_mu), ceiling_ci=_pct_ci(boot_ceiling),
            skill_expression=(ceiling - floor) / field_spread
            if field_spread else float("nan"),
        ))

    out.sort(key=lambda d: -d.ceiling_p90)
    return out


def _beta_spread(mu: float, kappa: float) -> float:
    if not (np.isfinite(mu) and np.isfinite(kappa)):
        return float("nan")
    a, b = mu * kappa, (1 - mu) * kappa
    return float(stats.beta.ppf(0.90, a, b) - stats.beta.ppf(0.10, a, b))


def _pct_ci(values: list[float]) -> tuple[float, float]:
    if len(values) < 20:
        return (float("nan"), float("nan"))
    return (float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)))


# --------------------------------------------------------------------------
# Field-weighted expectation
# --------------------------------------------------------------------------

def expected_vs_field(matrix: dict[tuple[str, str], dict],
                      deck: str, field_shares: dict[str, float],
                      min_matches: int = 10,
                      prior_weight: float = 12.0) -> float:
    """Expected win rate for `deck` against a metagame of `field_shares`.

    Sparse cells are shrunk toward 50% with a beta prior of `prior_weight`
    pseudo-matches, so a 2-0 cell cannot swing the total.
    """
    total_weight = 0.0
    total = 0.0
    for opp, share in field_shares.items():
        cell = matrix.get((deck, opp))
        if cell is None:
            continue
        w, n = cell["wins"], cell["wins"] + cell["losses"]
        if n == 0:
            continue
        shrunk = (w + prior_weight * 0.5) / (n + prior_weight)
        total += share * shrunk
        total_weight += share
    return total / total_weight if total_weight else float("nan")
