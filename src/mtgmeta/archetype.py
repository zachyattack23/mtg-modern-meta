"""Archetype classification from decklist contents.

melee.gg's `DecklistName` is free text the player typed, and when they leave it
blank melee auto-fills a colour string. In the China Open that produced 127
distinct name strings for roughly 15 real archetypes, with entries like
"Izzet", "W-U-B-G" and "Colorless" carrying no strategy information at all.
So names cannot be trusted as labels and decks get classified by their cards.

The pipeline is:

  1. `label_from_name`   -- keep the player names that actually say something,
                            discard melee's colour auto-fill.
  2. `mine_signatures`   -- for each weak-label cluster, find cards that are
                            common inside it and rare outside it.
  3. `Ruleset.classify`  -- apply ordered, human-readable card rules to every
                            deck, including the ones with useless names.

Stage 2 exists so the rules come from the data in front of us rather than from
a remembered metagame; the mined signatures get reviewed and frozen into
`rules.json`, which is the artefact a human edits.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Stage 1: which player-typed names are worth anything
# --------------------------------------------------------------------------

# Colour identities melee uses when auto-naming an unnamed decklist.
_COLOR_WORDS = {
    "mono-white", "mono-blue", "mono-black", "mono-red", "mono-green",
    "mono white", "mono blue", "mono black", "mono red", "mono green",
    "white", "blue", "black", "red", "green", "colorless", "colourless",
    "azorius", "dimir", "rakdos", "gruul", "selesnya", "orzhov", "izzet",
    "golgari", "boros", "simic", "esper", "grixis", "jund", "naya", "bant",
    "abzan", "jeskai", "sultai", "mardu", "temur", "wubrg", "5c", "5-color",
    "five-color", "four-color", "4c", "domain",
}

# Words that describe a posture, not an archetype. "Izzet Aggro" does not tell
# us Prowess from Affinity, so it is not usable as a label.
_GENERIC_WORDS = {
    "aggro", "aggro-control", "control", "midrange", "midrange-control",
    "combo", "ramp", "tempo", "deck", "modern", "value", "pile", "good",
    "stuff", "goodstuff", "brew", "list", "the", "and", "my", "v2", "v3",
    "budget", "tuned", "final", "updated", "new", "test", "copy",
}

# "W-U-B-G", "WUBG", "U/R", "B/G/W"
_COLOR_CODE = re.compile(r"^[wubrgc]([-/][wubrgc])*$", re.I)
# Companion / trailing parentheticals: "Boros (Kaheera)"
_PARENS = re.compile(r"\([^)]*\)")
_NONWORD = re.compile(r"[^\w\s'’-]+")


def label_from_name(name: str | None) -> str | None:
    """Return a usable archetype label from a player-typed name, or None.

    None means "melee auto-filled this" or "the player only described colours
    and posture" -- either way there is nothing to learn from it.

        >>> label_from_name("W-U-B-G Goryo's")
        "Goryo's"
        >>> label_from_name("Izzet") is None
        True
        >>> label_from_name("Boros (Kaheera)") is None
        True
    """
    if not name:
        return None

    cleaned = _PARENS.sub(" ", name)
    cleaned = _NONWORD.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    low = cleaned.lower()
    if low in _COLOR_WORDS or _COLOR_CODE.match(low.replace(" ", "")):
        return None

    # Strip leading colour identity ("Mono-Green Eldrazi Broodscale" ->
    # "Eldrazi Broodscale"), then drop generic posture words.
    tokens = cleaned.split()
    keep: list[str] = []
    for tok in tokens:
        tl = tok.lower()
        if tl in _COLOR_WORDS or _COLOR_CODE.match(tl):
            continue
        if tl.startswith("mono-") or tl.startswith("mono"):
            continue
        if tl in _GENERIC_WORDS:
            continue
        keep.append(tok)

    if not keep:
        return None
    return " ".join(keep)


# --------------------------------------------------------------------------
# Stage 2: mine discriminative cards from the weak-label clusters
# --------------------------------------------------------------------------

@dataclass
class Signature:
    label: str
    n_decks: int
    # (card, support inside cluster, support outside cluster, lift)
    cards: list[tuple[str, float, float, float]]


def mine_signatures(decks: list[dict], *, min_cluster: int = 4,
                    min_inside: float = 0.75, max_outside: float = 0.15,
                    top_n: int = 12) -> list[Signature]:
    """Find cards that identify each weak-label cluster.

    `decks` is a list of {'label': str|None, 'cards': set[str]} (maindeck card
    names). A card qualifies when it appears in at least `min_inside` of the
    cluster and at most `max_outside` of everything else.
    """
    clusters: dict[str, list[set[str]]] = collections.defaultdict(list)
    for deck in decks:
        if deck["label"]:
            clusters[deck["label"]].append(deck["cards"])

    # Normalise labels that differ only by case/spacing before counting.
    merged: dict[str, list[set[str]]] = collections.defaultdict(list)
    for label, members in clusters.items():
        merged[label.lower().strip()].extend(members)

    total_decks = len(decks)
    global_counts: collections.Counter[str] = collections.Counter()
    for deck in decks:
        global_counts.update(deck["cards"])

    signatures: list[Signature] = []
    for label, members in merged.items():
        if len(members) < min_cluster:
            continue
        inside_counts: collections.Counter[str] = collections.Counter()
        for cards in members:
            inside_counts.update(cards)

        scored: list[tuple[str, float, float, float]] = []
        for card, n_in in inside_counts.items():
            support_in = n_in / len(members)
            if support_in < min_inside:
                continue
            n_out = global_counts[card] - n_in
            support_out = n_out / max(total_decks - len(members), 1)
            if support_out > max_outside:
                continue
            lift = support_in / max(support_out, 1e-6)
            scored.append((card, support_in, support_out, lift))

        scored.sort(key=lambda row: (-row[3], -row[1]))
        if scored:
            signatures.append(Signature(label=label, n_decks=len(members),
                                        cards=scored[:top_n]))

    signatures.sort(key=lambda s: -s.n_decks)
    return signatures


# --------------------------------------------------------------------------
# Stage 3: ordered rules
# --------------------------------------------------------------------------

@dataclass
class Rule:
    """One archetype test.

    `all_of`     -- every card must be present at >= the given count
    `any_of`     -- at least `any_min` of these cards present
    `any_groups` -- several independent any-of tests, each {cards, min}; ALL
                    groups must pass
    `none_of`    -- none of these may be present (used to split near-twins)

    `any_groups` exists because colour variants need two unrelated tests at
    once: a deck is Jeskai Blink if it plays a blink payoff (one group) AND a
    red source (another group). A single `any_of` list would let a red source
    alone satisfy the rule.

    Counts are maindeck+sideboard unless `main_only` is set, because some
    archetypes are identified by a sideboard package.
    """
    name: str
    parent: str | None = None
    all_of: dict[str, int] = field(default_factory=dict)
    any_of: list[str] = field(default_factory=list)
    any_min: int = 1
    any_groups: list[dict] = field(default_factory=list)
    none_of: list[str] = field(default_factory=list)
    main_only: bool = True
    priority: int = 100
    notes: str = ""

    def matches(self, main: dict[str, int], side: dict[str, int]) -> bool:
        pool = dict(main) if self.main_only else _merge(main, side)
        for card, need in self.all_of.items():
            if pool.get(card, 0) < need:
                return False
        if self.any_of:
            hits = sum(1 for card in self.any_of if pool.get(card, 0) > 0)
            if hits < self.any_min:
                return False
        for group in self.any_groups:
            hits = sum(1 for card in group["cards"] if pool.get(card, 0) > 0)
            if hits < group.get("min", 1):
                return False
        for card in self.none_of:
            if pool.get(card, 0) > 0:
                return False
        return True

    @property
    def specificity(self) -> int:
        """How many card conditions this rule imposes, for tie-breaking."""
        return (len(self.all_of) + len(self.any_of)
                + sum(len(g["cards"]) for g in self.any_groups) + len(self.none_of))


def _merge(main: dict[str, int], side: dict[str, int]) -> dict[str, int]:
    out = dict(main)
    for card, qty in side.items():
        out[card] = out.get(card, 0) + qty
    return out


UNCLASSIFIED = "Unclassified"


@dataclass
class Ruleset:
    rules: list[Rule]

    @classmethod
    def load(cls, path: pathlib.Path) -> "Ruleset":
        raw = json.loads(path.read_text())
        rules = [Rule(**entry) for entry in raw["rules"]]
        # Most specific first; ties broken by how many cards a rule demands so
        # that a narrow combo rule beats a broad colour-ish one.
        rules.sort(key=lambda r: (r.priority, -r.specificity))
        return cls(rules=rules)

    def classify(self, main: dict[str, int],
                 side: dict[str, int]) -> tuple[str, str | None]:
        for rule in self.rules:
            if rule.matches(main, side):
                return rule.name, rule.parent
        return UNCLASSIFIED, None


def cards_to_counts(entries: list[list]) -> dict[str, int]:
    """Decklist JSON stores [[qty, card], ...]; collapse to {card: qty}."""
    out: dict[str, int] = {}
    for qty, card in entries:
        out[card] = out.get(card, 0) + int(qty)
    return out
