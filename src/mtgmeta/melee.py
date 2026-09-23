"""Client for melee.gg's tournament endpoints.

Everything here is the same data melee.gg's own web UI loads over AJAX. Three
endpoints matter:

  GET  /Tournament/View/{id}            -- HTML; round ids live in data-id attrs
  POST /Match/GetRoundMatches/{roundId} -- every match, both decklists, game scores
  POST /Standing/GetRoundStandings      -- final records (roundId in the form body)
  GET  /Decklist/View/{guid}            -- full 75 inside <pre id="decklist-text">

The endpoints are DataTables-backed, so they want the full server-side columns
payload. A bare {draw, start, length} body gets a 500 or an empty result set.
A Referer header is also required or the WAF returns 403.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

BASE = "https://melee.gg"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

STANDINGS_COLUMNS = [
    "Rank", "Player", "Decklists", "MatchRecord", "GameRecord", "Points",
    "OpponentMatchWinPercentage", "TeamGameWinPercentage",
    "OpponentGameWinPercentage", "FinalTiebreaker", "OpponentCount",
]

MATCH_COLUMNS = ["TableNumber", "PodNumber", "Teams", "Decklists", "ResultString"]

# Politeness delay between requests, seconds.
THROTTLE = 0.05


class MeleeError(RuntimeError):
    pass


def _datatables_payload(columns: list[str], start: int, length: int,
                        extra: dict[str, str] | None = None) -> bytes:
    params: list[tuple[str, str]] = [("draw", "1")]
    for i, col in enumerate(columns):
        params += [
            (f"columns[{i}][data]", col),
            (f"columns[{i}][name]", col),
            (f"columns[{i}][searchable]", "true"),
            (f"columns[{i}][orderable]", "true"),
            (f"columns[{i}][search][value]", ""),
            (f"columns[{i}][search][regex]", "false"),
        ]
    params += [
        ("order[0][column]", "0"),
        ("order[0][dir]", "asc"),
        ("start", str(start)),
        ("length", str(length)),
        ("search[value]", ""),
        ("search[regex]", "false"),
    ]
    for k, v in (extra or {}).items():
        params.append((k, v))
    return urllib.parse.urlencode(params).encode()


def _request(url: str, *, data: bytes | None = None, referer: str = BASE,
             retries: int = 3) -> str:
    headers = {
        "User-Agent": _UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
    }
    if data is not None:
        headers.update({
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        })
    else:
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                time.sleep(THROTTLE)
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last = exc
            time.sleep(2 ** attempt)
    raise MeleeError(f"{url} failed after {retries} attempts: {last}")


def _post_json(url: str, data: bytes, referer: str) -> dict:
    body = _request(url, data=data, referer=referer)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MeleeError(f"{url} returned non-JSON: {body[:200]}") from exc
    if payload.get("Error"):
        raise MeleeError(f"{url}: {payload.get('Message')}")
    return payload


def _paged(url: str, columns: list[str], referer: str,
           extra: dict[str, str] | None = None, page: int = 100) -> list[dict]:
    """Walk a DataTables endpoint until we've seen recordsTotal rows."""
    rows: list[dict] = []
    start = 0
    total = None
    while total is None or start < total:
        payload = _post_json(url, _datatables_payload(columns, start, page, extra), referer)
        total = payload["recordsTotal"]
        batch = payload.get("data") or []
        if not batch:
            break
        rows.extend(batch)
        start += page
    return rows


# --------------------------------------------------------------------------
# Tournament metadata
# --------------------------------------------------------------------------

@dataclass
class Round:
    id: int
    name: str

    @property
    def is_swiss(self) -> bool:
        """Playoff rounds are single-elimination and pair only the top cut.

        Their win rates reflect the cut, not the field, so we keep them flagged
        and let the analysis decide whether to include them.
        """
        return bool(re.match(r"^Round \d+$", self.name))


@dataclass
class Tournament:
    id: int
    name: str
    start_date: str | None
    rounds: list[Round] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"{BASE}/Tournament/View/{self.id}"


def get_tournament(tournament_id: int) -> Tournament:
    url = f"{BASE}/Tournament/View/{tournament_id}"
    page = _request(url)

    title_match = re.search(r"<title>(.*?)\s*\|\s*Melee</title>", page, re.S)
    name = html.unescape(title_match.group(1)).strip() if title_match else str(tournament_id)

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}", page)
    start_date = date_match.group(1) if date_match else None

    # Round buttons appear twice (standings + pairings selectors); dedupe by id
    # while preserving the order melee renders them in.
    seen: dict[int, Round] = {}
    for rid, rname in re.findall(
        r'round-selector" data-id="(\d+)" data-name="([^"]+)"', page
    ):
        rid_int = int(rid)
        if rid_int not in seen:
            seen[rid_int] = Round(id=rid_int, name=html.unescape(rname))

    if not seen:
        raise MeleeError(f"no rounds found on {url} (private or not yet started?)")

    return Tournament(id=tournament_id, name=name, start_date=start_date,
                      rounds=list(seen.values()))


# --------------------------------------------------------------------------
# Matches and standings
# --------------------------------------------------------------------------

def get_round_matches(tournament_id: int, round_id: int) -> list[dict]:
    return _paged(
        f"{BASE}/Match/GetRoundMatches/{round_id}",
        MATCH_COLUMNS,
        referer=f"{BASE}/Tournament/View/{tournament_id}",
    )


def get_standings(tournament_id: int, round_id: int) -> list[dict]:
    return _paged(
        f"{BASE}/Standing/GetRoundStandings",
        STANDINGS_COLUMNS,
        referer=f"{BASE}/Tournament/View/{tournament_id}",
        extra={"roundId": str(round_id)},
    )


_DECKLIST_RE = re.compile(r'<pre class="[^"]*" id="decklist-text">(.*?)</pre>', re.S)
_CARD_RE = re.compile(r"^(\d+)\s+(.+)$")


def get_decklist(decklist_id: str) -> dict[str, list[tuple[int, str]]]:
    """Return {'main': [(qty, card), ...], 'side': [...]}."""
    page = _request(f"{BASE}/Decklist/View/{decklist_id}",
                    referer=f"{BASE}/Decklist/View/{decklist_id}")
    match = _DECKLIST_RE.search(page)
    if not match:
        raise MeleeError(f"no decklist-text block for {decklist_id}")

    text = html.unescape(match.group(1))
    section = "main"
    out: dict[str, list[tuple[int, str]]] = {"main": [], "side": []}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low in {"maindeck", "deck", "main deck"}:
            section = "main"
            continue
        if low in {"sideboard", "side board"}:
            section = "side"
            continue
        if low in {"companion", "commander"}:
            # Companion is listed separately but is a sideboard card in practice.
            section = "side"
            continue
        card_match = _CARD_RE.match(line)
        if card_match:
            out[section].append((int(card_match.group(1)), card_match.group(2).strip()))
    return out
