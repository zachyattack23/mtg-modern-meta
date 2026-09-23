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
import datetime
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

# Draws are ~2.5% of matches and are dropped from the binomial denominator
# rather than counted as half a win, which keeps the likelihood integral.
DRAW_POLICY = "drop"

# Recency. Within one format people keep adapting, so an August match is weaker
# evidence about today than a September one. Weight decays exponentially with
# age; half_life_days=None disables it entirely.
#
# The cost is real: down-weighting is equivalent to throwing away sample size,
# which is the exact resource the ceiling model is short of. Every weighted
# figure therefore carries an effective sample size, (sum w)^2 / sum w^2, so
# the price is visible rather than hidden.
DEFAULT_HALF_LIFE_DAYS = 60.0


def recency_weight(date: str | None, reference: str,
                   half_life_days: float | None) -> float:
    """exp(-ln2 * age / half_life). Returns 1.0 when decay is off."""
    if not half_life_days or not date:
        return 1.0
    try:
        d0 = datetime.date.fromisoformat(date)
        d1 = datetime.date.fromisoformat(reference)
    except ValueError:
        return 1.0
    age = max((d1 - d0).days, 0)
    return float(math.exp(-math.log(2.0) * age / half_life_days))


def normalize_weights(matches: list["MatchRecord"]) -> None:
    """Rescale weights in place so their mean is 1.

    Without this, decay shrinks every weight below 1 and the beta-binomial
    reads the whole dataset as less evidence -- which drives kappa up and
    quietly compresses every deck's spread toward its mean. Halving all weights
    in a test moved kappa from 41 to 140 while leaving mu untouched, so the
    ceilings would have flattened for a purely artificial reason.

    Normalising keeps recency as a statement about *relative* emphasis, which
    is what was intended, and leaves the honest precision loss to be reported
    through `effective_n` and the intervals rather than smuggled into kappa.
    """
    if not matches:
        return
    total = sum(m.weight for m in matches)
    if total <= 0:
        return
    scale = len(matches) / total
    for m in matches:
        m.weight *= scale


def effective_n(weights) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2."""
    w = np.asarray(list(weights), dtype=float)
    if w.size == 0 or not w.any():
        return 0.0
    return float(w.sum() ** 2 / np.square(w).sum())


# --------------------------------------------------------------------------
# Match records
# --------------------------------------------------------------------------

@dataclass
class MatchRecord:
    tournament_id: int
    tournament_date: str | None
    weight: float
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
                    swiss_only: bool = True, *, reference_date: str | None = None,
                    half_life_days: float | None = None) -> list[MatchRecord]:
    """Flatten a fetched tournament into two-sided match records.

    Byes and single-competitor rows are dropped: they carry no matchup
    information and would inflate the win rate of whichever decks got them.
    """
    reference = reference_date or datetime.date.today().isoformat()
    weight = recency_weight(tournament.get("start_date"), reference, half_life_days)
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
                tournament_date=tournament.get("start_date"),
                weight=weight,
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
    """Match- and game-level win rates per deck, weighted and raw.

    Mirrors are excluded by default. A mirror is 50% by construction, so
    including them drags every deck toward 0.5 in proportion to its own
    metagame share -- which penalises exactly the popular decks whose numbers
    we trust most.

    Both the recency-weighted rate and the raw one are returned, because the
    difference between them is itself informative: a deck whose weighted rate
    sits well above its raw rate is on the way up.
    """
    agg: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"w": 0.0, "l": 0.0, "d": 0.0, "gw": 0.0, "gl": 0.0,
                 "rw": 0, "rl": 0, "rd": 0, "rgw": 0, "rgl": 0})
    weights: dict[str, list[float]] = collections.defaultdict(list)

    for m in matches:
        if exclude_mirrors and m.is_mirror:
            continue
        for deck, gw, gl in ((m.deck_a, m.game_wins_a, m.game_wins_b),
                             (m.deck_b, m.game_wins_b, m.game_wins_a)):
            row = agg[deck]
            weights[deck].append(m.weight)
            row["gw"] += gw * m.weight
            row["gl"] += gl * m.weight
            row["rgw"] += gw
            row["rgl"] += gl
            if m.is_draw:
                row["d"] += m.weight
                row["rd"] += 1
            elif gw > gl:
                row["w"] += m.weight
                row["rw"] += 1
            else:
                row["l"] += m.weight
                row["rl"] += 1

    out: dict[str, dict] = {}
    for deck, row in agg.items():
        decided = row["w"] + row["l"]
        raw_decided = row["rw"] + row["rl"]
        games = row["gw"] + row["gl"]
        raw_games = row["rgw"] + row["rgl"]
        out[deck] = {
            "matches": row["rw"] + row["rl"] + row["rd"],
            "wins": row["rw"], "losses": row["rl"], "draws": row["rd"],
            "match_win_rate": row["w"] / decided if decided else float("nan"),
            "raw_win_rate": row["rw"] / raw_decided if raw_decided else float("nan"),
            "game_wins": row["rgw"], "game_losses": row["rgl"],
            "game_win_rate": row["gw"] / games if games else float("nan"),
            "raw_game_win_rate": row["rgw"] / raw_games if raw_games else float("nan"),
            "effective_matches": effective_n(weights[deck]),
        }
    return out


def matchup_matrix(matches: list[MatchRecord],
                   decks: list[str]) -> dict[tuple[str, str], dict]:
    """Head-to-head record for every ordered deck pair, including mirrors.

    The reported rate is recency-weighted; the interval is computed on the
    effective sample size, so down-weighted evidence widens the interval
    instead of quietly counting as much as fresh evidence.
    """
    cells: dict[tuple[str, str], dict[str, float]] = collections.defaultdict(
        lambda: {"w": 0.0, "l": 0.0, "d": 0.0, "rw": 0, "rl": 0, "rd": 0,
                 "gw": 0.0, "gl": 0.0})
    weights: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    wanted = set(decks)

    for m in matches:
        if m.deck_a not in wanted or m.deck_b not in wanted:
            continue
        for deck, opp, gw, gl in (
            (m.deck_a, m.deck_b, m.game_wins_a, m.game_wins_b),
            (m.deck_b, m.deck_a, m.game_wins_b, m.game_wins_a),
        ):
            cell = cells[(deck, opp)]
            weights[(deck, opp)].append(m.weight)
            cell["gw"] += gw * m.weight
            cell["gl"] += gl * m.weight
            if m.is_draw:
                cell["d"] += m.weight
                cell["rd"] += 1
            elif gw > gl:
                cell["w"] += m.weight
                cell["rw"] += 1
            else:
                cell["l"] += m.weight
                cell["rl"] += 1

    out: dict[tuple[str, str], dict] = {}
    for key, cell in cells.items():
        decided = cell["w"] + cell["l"]
        raw_decided = cell["rw"] + cell["rl"]
        rate = cell["w"] / decided if decided else float("nan")
        eff = effective_n(weights[key])
        # Wilson on the effective n, with the weighted rate as the centre.
        eff_decided = eff * (raw_decided / max(raw_decided + cell["rd"], 1))
        lo, hi = wilson(rate * eff_decided, eff_decided)
        games = cell["gw"] + cell["gl"]
        out[key] = {
            "matches": raw_decided + cell["rd"],
            "wins": cell["rw"], "losses": cell["rl"], "draws": cell["rd"],
            "win_rate": rate,
            "raw_win_rate": cell["rw"] / raw_decided if raw_decided else float("nan"),
            "ci_low": lo, "ci_high": hi,
            "effective_matches": eff,
            "game_win_rate": cell["gw"] / games if games else float("nan"),
            "games": int(cell["gw"] + cell["gl"]),
        }
    return out


def wilson(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; degrades gracefully at n = 0 and at 0%/100%."""
    if n <= 0:
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
    skill_expression: float = float("nan")  # spread relative to the MEDIAN DECK's
    effective_matches: float = float("nan")  # Kish n after recency weighting


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
                  ) -> dict[str, dict[int, tuple[float, float, int]]]:
    """{deck: {player_id: (weighted_wins, weighted_matches, raw_matches)}}.

    A player who switched decks between events is counted separately per deck,
    which is what we want: the unit of analysis is a pilot-deck pairing.

    Weighted counts are real-valued, which the beta-binomial marginal
    likelihood handles without modification -- betaln is defined on the reals,
    so a down-weighted pilot simply contributes less evidence. The raw match
    count rides along so that inclusion thresholds stay honest: a pilot is kept
    or dropped on how many games they actually played, not on how old they are.
    """
    agg: dict[str, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0.0, 0]))

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
            rec[1] += m.weight
            rec[2] += 1
            if gw > gl:
                rec[0] += m.weight

    return {deck: {p: (r[0], r[1], int(r[2])) for p, r in players.items()}
            for deck, players in agg.items()}


def deck_spreads(matches: list[MatchRecord], *, min_pilots: int = 5,
                 min_matches_per_pilot: int = 3, kappa_sd: float = 1.5,
                 n_boot: int = 300, seed: int = 0) -> list[DeckSpread]:
    """Fit the hierarchical model and report each deck's ceiling."""
    rng = np.random.default_rng(seed)
    records = pilot_records(matches)

    # Global prior: how much do pilots vary across the whole field?
    flat = {d: [(w, n) for w, n, _ in v.values()] for d, v in records.items()}
    mu_global, kappa_global = fit_global_kappa(flat)
    log_kappa_prior = (math.log(kappa_global), kappa_sd)

    out: list[DeckSpread] = []
    for deck, players in records.items():
        # Threshold on raw matches played, then fit on the weighted counts.
        usable = [(w, n) for w, n, raw in players.values()
                  if raw >= min_matches_per_pilot and n > 0]
        if len(usable) < min_pilots:
            continue
        wins = np.array([w for w, _ in usable], dtype=float)
        trials = np.array([n for _, n in usable], dtype=float)
        raw_total = sum(raw for _, _, raw in players.values()
                        if raw >= min_matches_per_pilot)

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
            n_matches=int(raw_total),
            effective_matches=float(trials.sum()),
            observed_mean=float(wins.sum() / trials.sum()),
            observed_p90=float(np.quantile(obs, 0.90)),
            mu=mu, kappa=kappa,
            ceiling_p90=ceiling, floor_p10=floor, spread=ceiling - floor,
            mu_ci=_pct_ci(boot_mu), ceiling_ci=_pct_ci(boot_ceiling),
        ))

    # Normalise skill expression against the MEDIAN DECK, not against a global
    # beta-binomial fitted over every pilot regardless of deck. That global fit
    # absorbs between-deck variation as well as within-deck pilot variation, so
    # its span (13.2 pts on this data) exceeds the median deck's (10.7) and 16
    # of 22 decks scored below 1.0 -- a column labelled "spread" where the
    # median deck reads 0.81 tells the reader the opposite of the truth.
    # Against the median deck, 1.0 means exactly "typical".
    spans = [d.spread for d in out if np.isfinite(d.spread)]
    reference = float(np.median(spans)) if spans else float("nan")
    for entry in out:
        entry.skill_expression = (entry.spread / reference
                                  if reference else float("nan"))

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
