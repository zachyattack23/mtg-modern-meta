# Modern metagame: ceilings, matchups, and contested slots

Pulls Modern tournament data from melee.gg and answers a question the usual
metagame report does not: **what does a deck do in the hands of a good pilot**,
as distinct from its average across everyone who sleeved it up.

Dashboard: https://claude.ai/artifact/TEvEvpbmYqkEkFECAqwTGj

## Quick start

```bash
python3 scripts/fetch_tournaments.py --config tournaments.json   # ~5 min cold
python3 scripts/mine_archetypes.py                               # signature report
python3 scripts/build_dataset.py                                 # classify + stats
python3 scripts/build_cards.py                                   # flex slots
python3 scripts/build_web_payload.py                             # web/data.js
```

Add an event by putting its melee tournament id in `tournaments.json` and
re-running. Everything is cached by id, so a re-run only fetches what is new.

## The three things worth knowing

**1. Deck names on melee are worthless, so decks are classified on cards.**
`DecklistName` is free text the player typed, and melee auto-fills a colour
string when they leave it blank. Across these four events that gave 332 distinct
name strings for ~30 real archetypes, including useless entries like "Izzet",
"W-U-B-G" and "Colorless". `rules.json` classifies on card signatures instead.
Those signatures were **mined from the decklists** (`mine_archetypes.py`) rather
than written from memory, because the current format contains cards no prior
knowledge would include. Current state: 4.5% unclassified, 92% agreement with
the player names that do say something, 434 decks rescued from colour auto-fill.

Spot-checking the disagreements, most are the classifier correctly overriding a
stale player name — a list named "Domain Zoo" that is actually Dimir Midrange,
an "Amulet Titan" with no Amulet of Vigor.

**2. The naive ceiling is mostly noise.** Over a 15-round event a pilot's win
rate has a standard error near 13 points, so ranking decks by their best pilot
ranks them by sample size. `stats.py` fits a hierarchical beta-binomial per deck
and reports the P90 of the *latent* pilot-skill distribution. On simulated data
where the truth is known the naive version overstates by 8-12 points; on this
data it overstates by 13-16.

**3. Card-choice effects are unmeasurable at this sample size.** Of 637 card
tests, **zero** had the power to detect a 4-point effect; the median detectable
effect is 13.4%. The flex-slot map (which slots pilots disagree on) is the
trustworthy output. The win-rate splits are a screen for implausibly large
effects, which usually turn out to be build splits rather than card choices.

## Layout

```
src/mtgmeta/melee.py      melee.gg client (undocumented DataTables endpoints)
src/mtgmeta/archetype.py  weak labels, signature mining, card rules
src/mtgmeta/stats.py      win rates, matchup matrix, beta-binomial ceiling
src/mtgmeta/cards.py      flex slots, effect tests, FDR correction
rules.json                the classifier -- edit this
ignore_cards.json         cards excluded from flex analysis (fetches, basics)
archetype_overrides.csv   decklist_id -> archetype, beats the rules
```

## Known limits

Pilots are **self-selected, not randomly assigned**. A deck with a wide fitted
spread may reward skill, or may simply have attracted a wider mix of players.
Nothing in this data separates those, and more events will not fix it. Read a
wide spread as "this deck's result varied a lot by pilot" — true and useful —
rather than "this deck rewards skill", which is a causal claim the design does
not support.
