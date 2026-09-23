#!/usr/bin/env python3
"""Compact the processed JSON into a single payload the dashboard embeds.

Keys are shortened and floats rounded to 4 places; at ~3400 decks and a 33x33
matchup matrix the verbose form is close to a megabyte, most of it repeated key
names and float noise well past any precision these estimates actually carry.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def r(x, places: int = 4):
    if x is None:
        return None
    try:
        if x != x:  # NaN
            return None
    except TypeError:
        return None
    return round(float(x), places)


def pack_view(view: dict) -> dict:
    return {
        "decks": [{
            "n": d["deck"], "p": d["parent"], "e": d["entries"], "s": r(d["share"]),
            "m": d["matches"], "w": r(d["win_rate"]),
            "ci": [r(d["win_rate_ci"][0]), r(d["win_rate_ci"][1])],
            "gw": r(d["game_win_rate"]),
            "np": d["n_pilots"], "c": r(d["ceiling"]),
            "cci": [r(d["ceiling_ci"][0]), r(d["ceiling_ci"][1])] if d["ceiling_ci"] else None,
            "f": r(d["floor"]), "mu": r(d["mu"]), "k": r(d["kappa"], 1),
            "sp": r(d["spread"]), "sx": r(d["skill_expression"], 3),
            "op": r(d["observed_p90"]), "vf": r(d.get("vs_field")),
            "rk": d["rankable"],
        } for d in view["decks"]],
        "matrix": [[c["deck"], c["opp"], c["wins"], c["losses"], c["draws"],
                    r(c["win_rate"]), r(c["ci_low"]), r(c["ci_high"])]
                   for c in view["matrix"]],
        "pilots": {k: [r(x, 3) for x in v] for k, v in view["pilots"].items()},
        "totalMatches": view["total_matches"],
        "totalDecks": view["total_decks"],
    }


def main() -> int:
    dash = json.loads((PROC / "dashboard.json").read_text())
    cards = json.loads((PROC / "cards.json").read_text())

    out = {
        "tournaments": dash["merged"]["tournaments"],
        "fine": pack_view(dash["fine"]),
        "merged": pack_view(dash["merged"]),
        "diagnostics": {k: v for k, v in dash["diagnostics"].items()
                        if k != "disagreements"},
        "cards": {
            arch: {
                "n": info["n_decks"],
                "slots": [{
                    "c": s["card"], "z": s["zone"], "k": s["kind"],
                    "i": r(s["inclusion"], 3), "mc": r(s["mean_copies"], 2),
                    "h": r(s["entropy"], 3),
                    "d": [s["distribution"][str(i)] for i in range(5)],
                } for s in info["slots"]],
                "effects": [{
                    "c": e["card"], "z": e["zone"], "sp": e["split"],
                    "ma": e["matches_a"], "mb": e["matches_b"],
                    "ra": r(e["rate_a"], 3), "rb": r(e["rate_b"], 3),
                    "d": r(e["delta"], 3), "q": r(e["q"], 4),
                    "mde": r(e["mde"], 3), "pw": e["powered"],
                } for e in info.get("effects", [])],
            } for arch, info in cards["archetypes"].items()
        },
        "cardSummary": cards["summary"],
    }

    web = ROOT / "web"
    web.mkdir(exist_ok=True)
    payload = json.dumps(out, ensure_ascii=True, separators=(",", ":"))
    (web / "data.js").write_text(f"window.META_DATA = {payload};\n")
    print(f"wrote web/data.js ({len(payload) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
