"""Sports info tool — live scores, schedules, recent results.

Backs onto ESPN's undocumented public API at site.api.espn.com. Stable for
years (it powers their own apps), but officially unsupported. If they ever
break it, only `_fetch_scoreboard` and the URL paths change — the tool
schema and Claude-facing behavior stay identical.

This is our first *client-side* tool: Anthropic gives us a `tool_use` block,
we execute the function locally, and feed a string back as a `tool_result`.
Compare with the server-side `web_search` tool where Anthropic does the
execution invisibly. The agentic loop in `src/llm.py` handles the round-trip.

Coverage (any league with `_LEAGUE_PATHS` entry): NFL, NBA, MLB, NHL, MLS,
EPL, Champions League, NCAAF, NCAAB, WNBA, UFC, F1, PGA, ATP, WTA. Adding
more is a one-line entry — ESPN's path conventions are consistent.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Any

import httpx


# --- Anthropic tool definition ---------------------------------------------
# Claude reads `description` to decide when to invoke. Keep it short but
# concrete; describe coverage so Claude doesn't try this for cricket or
# rugby (we don't ship those leagues yet).
SPORTS_TOOL = {
    "name": "get_sports_info",
    "description": (
        "Get live scores, schedules, and recent results for major sports leagues "
        "(NFL, NBA, MLB, NHL, MLS, EPL, Champions League, NCAA football and "
        "basketball, WNBA, UFC, F1, PGA, ATP, WTA). Use this for any sports "
        "current-information question — it returns structured live data and is "
        "more reliable than web_search for game scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "league": {
                "type": "string",
                "description": (
                    "League code: nfl, nba, mlb, nhl, mls, epl, ucl, ncaaf, "
                    "ncaab, wnba, ufc, f1, pga, atp, wta."
                ),
            },
            "team": {
                "type": "string",
                "description": (
                    "Optional team name to filter by. Fuzzy match — 'Giants', "
                    "'NY Giants', 'New York Giants' all work."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["live", "today", "yesterday", "upcoming", "date"],
                "description": (
                    "live = in-progress only; today = today's full slate; "
                    "yesterday = yesterday's results; upcoming = next 7 days "
                    "of scheduled games; date = a specific date (also pass `date`)."
                ),
            },
            "date": {
                "type": "string",
                "description": "YYYY-MM-DD. Required only when mode=date.",
            },
        },
        "required": ["league", "mode"],
    },
}


# ESPN's URL path component for each league. Format: <sport>/<league>.
# Soccer leagues use locale-prefixed codes: usa.1 = MLS, eng.1 = Premier
# League, uefa.champions = Champions League. Add new leagues here only.
_LEAGUE_PATHS: dict[str, str] = {
    "nfl": "football/nfl",
    "ncaaf": "football/college-football",
    "nba": "basketball/nba",
    "ncaab": "basketball/mens-college-basketball",
    "wnba": "basketball/wnba",
    "mlb": "baseball/mlb",
    "nhl": "hockey/nhl",
    "mls": "soccer/usa.1",
    "epl": "soccer/eng.1",
    "ucl": "soccer/uefa.champions",
    "ufc": "mma/ufc",
    "f1": "racing/f1",
    "pga": "golf/pga",
    "atp": "tennis/atp",
    "wta": "tennis/wta",
}

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Cap output for voice — TTS reading 30 NCAAF Saturday games is unbearable.
_MAX_GAMES_PER_RESPONSE = 8


def _fetch_scoreboard(league_path: str, date_yyyymmdd: str | None) -> dict[str, Any]:
    """One ESPN scoreboard fetch. Always returns a dict — never raises.
    On any error (network, parse, HTTP status) we log and return {} so the
    caller treats it as "no games" rather than crashing the turn."""
    url = f"{_ESPN_BASE}/{league_path}/scoreboard"
    params = {"dates": date_yyyymmdd} if date_yyyymmdd else {}
    try:
        r = httpx.get(url, params=params, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(
            f"[sports] ESPN fetch failed ({league_path} {date_yyyymmdd}): {exc}",
            file=sys.stderr,
        )
        return {}


def _team_matches(competitor: dict, query: str) -> bool:
    """Fuzzy team match. Substring-in-haystack, case-insensitive, against every
    name field ESPN exposes. Lets Claude pass 'Giants', 'NY Giants', 'NYG',
    'New York Giants' — they all hit."""
    team = competitor.get("team", {}) or {}
    haystack = " ".join(
        s for s in (
            team.get("displayName"),
            team.get("shortDisplayName"),
            team.get("name"),
            team.get("location"),
            team.get("abbreviation"),
            team.get("nickname"),
        ) if s
    ).lower()
    return query in haystack


def _format_game(event: dict) -> str:
    """One-line voice summary of a game. Three states:

      pre   "Giants at Cowboys, Sun 8:30 PM"
      in    "Giants 14, Cowboys 7 — 3rd Qtr 8:42"
      post  "Giants beat Cowboys 24-21"

    Sports without home/away (UFC, golf, tennis, racing) fall back to
    ESPN's pre-formatted event name — those are individual events, not
    head-to-head, so the team-pair format doesn't apply.
    """
    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors", []) or []

    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)

    status = (event.get("status") or {}).get("type") or {}
    state = status.get("state", "")
    detail = status.get("shortDetail", "") or status.get("detail", "")

    # Non-team sports (UFC fights, F1 races, tennis matches, golf tournaments)
    # — ESPN's `name` field is already human-readable, so just use it.
    if not (home and away):
        return f"{event.get('name') or event.get('shortName') or 'Event'}: {detail}"

    home_team = home.get("team") or {}
    away_team = away.get("team") or {}
    home_name = home_team.get("shortDisplayName") or home_team.get("displayName") or "Home"
    away_name = away_team.get("shortDisplayName") or away_team.get("displayName") or "Away"
    home_score = home.get("score", "")
    away_score = away.get("score", "")

    if state == "pre":
        return f"{away_name} at {home_name}, {detail}"

    if state == "in":
        return f"{away_name} {away_score}, {home_name} {home_score} — {detail}"

    if state == "post":
        # Try numeric comparison so we can phrase as "X beat Y N-M".
        try:
            h, a = int(home_score), int(away_score)
            if h == a:
                return f"{away_name} {a}, {home_name} {h} (final, tie)"
            winner_name, winner_score = (home_name, h) if h > a else (away_name, a)
            loser_name, loser_score = (away_name, a) if h > a else (home_name, h)
            return f"{winner_name} beat {loser_name} {winner_score}-{loser_score}"
        except (TypeError, ValueError):
            return f"{away_name} {away_score}, {home_name} {home_score} (final)"

    # Unknown state — return whatever ESPN gives us.
    return f"{away_name} at {home_name}: {detail}"


def _dates_for_mode(mode: str, date_str: str | None) -> list[str]:
    """Map a `mode` to a list of YYYYMMDD strings to query.

    `upcoming` fetches a 7-day window because some sports (NFL) only play
    once a week; today-only would miss the next game most of the time.
    All other modes hit a single date.
    """
    today = datetime.now()
    if mode == "yesterday":
        return [(today - timedelta(days=1)).strftime("%Y%m%d")]
    if mode == "upcoming":
        return [(today + timedelta(days=i)).strftime("%Y%m%d") for i in range(0, 8)]
    if mode == "date" and date_str:
        try:
            return [datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")]
        except ValueError:
            pass  # fall through to today
    # 'live' and 'today' both look at today's scoreboard; we filter by state
    # later.
    return [today.strftime("%Y%m%d")]


def execute_sports_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises.
    Every error path produces a readable message so Claude can paraphrase it
    politely instead of crashing the turn."""
    league = (params.get("league") or "").lower().strip()
    team = (params.get("team") or "").lower().strip() or None
    mode = (params.get("mode") or "today").lower().strip()
    date_str = params.get("date")

    league_path = _LEAGUE_PATHS.get(league)
    if not league_path:
        return (
            f"Unknown league '{league}'. Supported: "
            f"{', '.join(sorted(_LEAGUE_PATHS.keys()))}."
        )

    target_dates = _dates_for_mode(mode, date_str)

    matched: list[str] = []
    for d in target_dates:
        data = _fetch_scoreboard(league_path, d)
        for event in data.get("events", []) or []:
            competition = (event.get("competitions") or [{}])[0]

            if team:
                competitors = competition.get("competitors", []) or []
                if not any(_team_matches(c, team) for c in competitors):
                    continue

            state = (event.get("status") or {}).get("type", {}).get("state", "")
            if mode == "live" and state != "in":
                continue
            if mode == "upcoming" and state != "pre":
                continue

            matched.append(_format_game(event))

    if not matched:
        descriptor = league.upper()
        if team:
            descriptor += f" matching '{team}'"
        return f"No {mode} games found for {descriptor}."

    if len(matched) > _MAX_GAMES_PER_RESPONSE:
        truncated = matched[:_MAX_GAMES_PER_RESPONSE]
        truncated.append(f"(+{len(matched) - _MAX_GAMES_PER_RESPONSE} more)")
        matched = truncated

    return "\n".join(matched)
