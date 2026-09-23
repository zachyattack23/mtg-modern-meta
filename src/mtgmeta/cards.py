"""Within-archetype card choice: which slots are contested, and do they matter.

Most of a competitive decklist is not a decision -- it is the same twenty-odd
cards at four copies in every build. The interesting part is the tail: the
slots where pilots of the *same* archetype disagree, both on whether to play a
card at all and on how many copies.

Both halves of that decision live in one distribution: the copy count across
pilots, counting non-inclusion as zero. A core four-of puts all its mass on 4
and has zero entropy. A card half the field plays as a two-of spreads mass over
{0, 2} and scores high. So normalised entropy over {0..4} ranks contested slots
without needing separate "inclusion" and "count" heuristics.

Measuring whether a choice *wins* is much weaker than describing it, and this
module is deliberate about saying so:

  * Power. A card played by a quarter of an archetype's pilots yields a few
    hundred matches per arm. The minimum detectable effect is ~6-7 win rate
    points; real card choices are probably worth 1-2. `min_detectable_effect`
    is reported next to every test so a null can be read as "no effect" or
    "could not have seen one".
  * Multiplicity. Tens of archetypes times tens of flex cards is a thousand-odd
    tests, so p-values go through Benjamini-Hochberg before anything is called
    a finding.
  * Confounding. Stronger players converge on the correct build, so a winning
    card may be the card good players pick rather than a card that causes wins.
    Nothing in this data separates those.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

MAX_COPIES = 4


# --------------------------------------------------------------------------
# Cards that are noise rather than decisions
# --------------------------------------------------------------------------

FETCHLANDS = {
    # Onslaught / Zendikar cycles -- within an archetype these are chosen by
    # mana requirements and are functionally interchangeable.
    "Flooded Strand", "Polluted Delta", "Bloodstained Mire",
    "Wooded Foothills", "Windswept Heath", "Scalding Tarn",
    "Verdant Catacombs", "Marsh Flats", "Arid Mesa", "Misty Rainforest",
    "Prismatic Vista", "Fabled Passage", "Evolving Wilds", "Terramorphic Expanse",
}

BASICS = {
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest",
}

DEFAULT_IGNORE = FETCHLANDS | BASICS


# --------------------------------------------------------------------------
# Flex slots
# --------------------------------------------------------------------------

@dataclass
class FlexSlot:
    card: str
    zone: str                       # "main" or "side"
    n_decks: int                    # decks in the archetype
    inclusion_rate: float           # share playing >= 1
    mean_copies: float              # over decks that play it
    distribution: dict[int, int]    # copies -> deck count, including 0
    entropy: float                  # normalised, 0 = unanimous, 1 = maximal
    kind: str                       # "core" | "inclusion" | "count" | "fringe"


def _normalised_entropy(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [n / total for n in counts.values() if n > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(MAX_COPIES + 1)


def flex_slots(decks: list[dict], zone: str = "main", *,
               ignore: set[str] | None = None,
               min_inclusion: float = 0.08,
               min_entropy: float = 0.12) -> list[FlexSlot]:
    """Rank the contested slots of one archetype.

    `decks` is a list of {'main': {card: qty}, 'side': {card: qty}}.
    """
    ignore = DEFAULT_IGNORE if ignore is None else ignore
    n = len(decks)
    if n == 0:
        return []

    seen: set[str] = set()
    for deck in decks:
        seen |= set(deck[zone])
    seen -= ignore

    out: list[FlexSlot] = []
    for card in seen:
        dist: dict[int, int] = {c: 0 for c in range(MAX_COPIES + 1)}
        over = 0
        for deck in decks:
            qty = deck[zone].get(card, 0)
            if qty > MAX_COPIES:
                over += 1
                qty = MAX_COPIES
            dist[qty] += 1
        played = n - dist[0]
        inclusion = played / n
        if inclusion < min_inclusion:
            continue

        entropy = _normalised_entropy(dist)
        mean_copies = (sum(c * k for c, k in dist.items() if c > 0)
                       / played if played else 0.0)

        if inclusion > 0.95 and dist[MAX_COPIES] / n > 0.9:
            kind = "core"
        elif inclusion < 0.15:
            kind = "fringe"
        elif inclusion < 0.9:
            kind = "inclusion"
        else:
            kind = "count"

        if entropy < min_entropy and kind == "core":
            continue

        out.append(FlexSlot(card=card, zone=zone, n_decks=n,
                            inclusion_rate=inclusion, mean_copies=mean_copies,
                            distribution=dist, entropy=entropy, kind=kind))

    out.sort(key=lambda s: -s.entropy)
    return out


# --------------------------------------------------------------------------
# Does the choice win?
# --------------------------------------------------------------------------

@dataclass
class CardEffect:
    archetype: str
    card: str
    zone: str
    split: str                    # e.g. "plays it" vs "does not"
    n_pilots_a: int
    n_pilots_b: int
    wins_a: int
    matches_a: int
    wins_b: int
    matches_b: int
    rate_a: float
    rate_b: float
    delta: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    p_value: float
    q_value: float = float("nan")   # BH-adjusted
    min_detectable_effect: float = float("nan")
    powered: bool = False           # is MDE small enough to be meaningful?


def min_detectable_effect(n_a: int, n_b: int, base_rate: float = 0.5,
                          alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest win-rate gap this sample could detect, as an absolute delta.

    Two-sided two-proportion test. Printed beside every result so that a
    non-finding can be read correctly: with a few hundred matches per arm the
    floor sits near 6-7 points, well above any realistic card effect.
    """
    if n_a < 2 or n_b < 2:
        return float("nan")
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(power)
    var = base_rate * (1 - base_rate) * (1 / n_a + 1 / n_b)
    return float((z_a + z_b) * math.sqrt(var))


def card_effect(archetype_name: str, slot: FlexSlot,
                deck_results: list[tuple[dict, int, int]],
                *, threshold: int | None = None,
                meaningful_effect: float = 0.04) -> CardEffect | None:
    """Compare pilots who played the card against those who did not.

    `deck_results` is [(deck, wins, matches), ...] for one archetype. When the
    slot is a count decision rather than an inclusion decision, `threshold`
    splits at >= threshold copies instead of >= 1.
    """
    cut = threshold if threshold is not None else 1
    a = [(w, m) for deck, w, m in deck_results
         if deck[slot.zone].get(slot.card, 0) >= cut and m > 0]
    b = [(w, m) for deck, w, m in deck_results
         if deck[slot.zone].get(slot.card, 0) < cut and m > 0]
    if len(a) < 5 or len(b) < 5:
        return None

    wins_a, matches_a = sum(w for w, _ in a), sum(m for _, m in a)
    wins_b, matches_b = sum(w for w, _ in b), sum(m for _, m in b)
    if matches_a < 30 or matches_b < 30:
        return None

    rate_a = wins_a / matches_a
    rate_b = wins_b / matches_b

    table = [[wins_a, matches_a - wins_a], [wins_b, matches_b - wins_b]]
    try:
        _, p_value = sps.fisher_exact(table)
    except ValueError:
        p_value = float("nan")

    from .stats import wilson
    mde = min_detectable_effect(matches_a, matches_b,
                                base_rate=(wins_a + wins_b) / (matches_a + matches_b))

    label = (f">={cut} copies" if threshold is not None else "plays it")
    return CardEffect(
        archetype=archetype_name, card=slot.card, zone=slot.zone, split=label,
        n_pilots_a=len(a), n_pilots_b=len(b),
        wins_a=wins_a, matches_a=matches_a, wins_b=wins_b, matches_b=matches_b,
        rate_a=rate_a, rate_b=rate_b, delta=rate_a - rate_b,
        ci_a=wilson(wins_a, matches_a), ci_b=wilson(wins_b, matches_b),
        p_value=p_value, min_detectable_effect=mde,
        powered=bool(np.isfinite(mde) and mde <= meaningful_effect),
    )


def benjamini_hochberg(effects: list[CardEffect], alpha: float = 0.10) -> None:
    """Attach BH-adjusted q-values in place.

    Without this the report is a false-discovery generator: a thousand tests at
    p < 0.05 manufactures fifty 'findings' from pure noise.
    """
    valid = [e for e in effects if np.isfinite(e.p_value)]
    if not valid:
        return
    valid.sort(key=lambda e: e.p_value)
    m = len(valid)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        q = min(prev, valid[i].p_value * m / (i + 1))
        valid[i].q_value = q
        prev = q


def analyse_archetype(archetype_name: str,
                      deck_results: list[tuple[dict, int, int]],
                      *, ignore: set[str] | None = None,
                      max_slots: int = 25) -> tuple[list[FlexSlot], list[CardEffect]]:
    """Flex-slot map plus effect tests for one archetype."""
    decks = [deck for deck, _, _ in deck_results]
    slots: list[FlexSlot] = []
    for zone in ("main", "side"):
        slots.extend(flex_slots(decks, zone=zone, ignore=ignore))
    slots.sort(key=lambda s: -s.entropy)
    slots = slots[:max_slots]

    effects: list[CardEffect] = []
    for slot in slots:
        if slot.kind in ("inclusion", "fringe"):
            eff = card_effect(archetype_name, slot, deck_results)
            if eff:
                effects.append(eff)
        elif slot.kind == "count":
            # Split at the median played count so the arms stay balanced.
            played = [c for c, k in slot.distribution.items()
                      for _ in range(k) if c > 0]
            if played:
                cut = int(np.median(played))
                if 1 < cut <= MAX_COPIES:
                    eff = card_effect(archetype_name, slot, deck_results,
                                      threshold=cut)
                    if eff:
                        effects.append(eff)
    return slots, effects
