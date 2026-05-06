"""Game info tool — release dates, summaries, popular titles, recommendations.

Backs onto the RAWG API (rawg.io/apidocs). Free tier with 20k requests/month;
no OAuth dance, just a `key=` query param. RAWG indexes ~half a million games
across every major platform with metacritic scores, genres, ESRB ratings,
release dates, and a similarity graph for recommendations.

Same defensive contract as src/sports.py and src/weather.py: never raises,
always returns a readable string. RAWG outages or missing API key produce
clear strings Claude can paraphrase to the user.

Modes:
  search   — top 5 matches for a name (release date, platforms, score)
  details  — one game with full description (auto-resolves name → top match)
  popular  — most-added recent games, optionally filtered by platform
  similar  — RAWG's suggested-games endpoint (recommendations)

Why these four modes match the voice query distribution:
  search  → "When does GTA 6 come out?" / "Is Stellar Blade on PC?"
  details → "Tell me about Death Stranding 2."
  popular → "What are the hot Switch games right now?"
  similar → "I loved Hollow Knight, what else would I like?"
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any

import httpx


# --- Anthropic tool definition ---------------------------------------------
GAMES_TOOL = {
    "name": "get_game_info",
    "description": (
        "Get information about video games — release dates, platforms, "
        "summaries, reviews, popular titles, and recommendations. Covers "
        "every major platform (PlayStation, Nintendo, Xbox, PC, mobile). "
        "Use this for any game-related question except the user's personal "
        "library or playtime (we don't have access to those). For news or "
        "reviews about a specific game on a specific outlet, prefer "
        "web_fetch with a URL."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["search", "details", "popular", "similar"],
                "description": (
                    "search = quick lookup with up to 5 matches; "
                    "details = full summary of one game (best when the user "
                    "wants info on a specific title); "
                    "popular = trending games filtered by platform/timeframe; "
                    "similar = recommendations based on a game the user liked."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Game name. Required for search, details, and similar. "
                    "Omit for popular."
                ),
            },
            "platform": {
                "type": "string",
                "description": (
                    "Platform filter. Accepted values: switch, ps5, ps4, ps3, "
                    "ps2, vita, xbox-series, xbox-one, xbox-360, pc, ios, "
                    "android. Use only with mode=popular (other modes ignore it)."
                ),
            },
            "timeframe": {
                "type": "string",
                "enum": ["recent", "year", "all"],
                "description": (
                    "For mode=popular: recent = last 90 days (default); "
                    "year = past year; all = all-time top-rated."
                ),
            },
        },
        "required": ["mode"],
    },
}


_RAWG_BASE = "https://api.rawg.io/api"

# Voice cap — Claude editorializes anyway, but we keep raw payloads short.
_MAX_RESULTS = 8

# Same defensive HTTP policy as weather.py / sports.py: 10s timeout, single
# retry on transport errors only (timeouts / DNS / connection drops). HTTP
# 4xx/5xx is returned immediately — those don't fix themselves.
_HTTP_TIMEOUT_SEC = 10.0
_RETRY_BACKOFF_SEC = 0.5


# RAWG's platform IDs. Voice queries say "Switch" / "PS5" / "PlayStation 5",
# not "187". This map normalizes those into RAWG's numeric IDs. Only commonly-
# spoken platforms — RAWG has hundreds, but most never come up in voice.
_PLATFORM_IDS: dict[str, str] = {
    "switch": "7",
    "nintendo switch": "7",
    # As of 2026-05, RAWG has not yet split Switch 2 from Switch — their
    # platform catalog tops out at id=7 ("Nintendo Switch"), and Switch 2
    # exclusives (Mario Kart World, etc.) are misclassified there too. We
    # map "switch 2" to the same id so voice queries don't 'unknown platform'
    # error; the result set will cover Switch 2 titles correctly but also
    # includes Switch 1 noise. Update to the new id when RAWG adds it.
    "switch 2": "7",
    "switch2": "7",
    "nintendo switch 2": "7",
    "3ds": "8",
    "nintendo 3ds": "8",
    "wii u": "10",
    "wii": "11",
    "ps5": "187",
    "playstation 5": "187",
    "playstation5": "187",
    "ps4": "18",
    "playstation 4": "18",
    "ps3": "16",
    "playstation 3": "16",
    "ps2": "15",
    "playstation 2": "15",
    "vita": "19",
    "ps vita": "19",
    "playstation vita": "19",
    "xbox-series": "186",
    "xbox series": "186",
    "xbox series x": "186",
    "xbox series s": "186",
    "xbox-one": "1",
    "xbox one": "1",
    "xbox-360": "14",
    "xbox 360": "14",
    "xbox": "186",  # default to current gen
    "pc": "4",
    "steam": "4",
    "ios": "3",
    "iphone": "3",
    "ipad": "3",
    "android": "21",
}


def _http_get_with_retry(
    url: str, params: dict[str, Any] | None = None
) -> httpx.Response | None:
    """GET with one retry on transport errors. Returns None on final failure
    or any non-2xx response. Caller checks for None."""
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            r = httpx.get(url, params=params, timeout=_HTTP_TIMEOUT_SEC)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as exc:
            print(f"[games] HTTP {exc.response.status_code} on {url}", file=sys.stderr)
            return None
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt == 1:
                print(
                    f"[games] {type(exc).__name__} on {url} — retrying in "
                    f"{_RETRY_BACKOFF_SEC}s",
                    file=sys.stderr,
                )
                time.sleep(_RETRY_BACKOFF_SEC)
                continue
            break
    print(f"[games] gave up after retry: {last_exc}", file=sys.stderr)
    return None


def _api_key() -> str | None:
    """Read the RAWG key fresh each call — keeps it hot-swappable in dev
    without restarting Jarvis."""
    key = os.getenv("RAWG_API_KEY", "").strip()
    return key or None


def _platform_id(name: str | None) -> str | None:
    """Resolve a spoken platform name to RAWG's numeric ID. None if unknown
    or unset (caller treats None as 'no platform filter')."""
    if not name:
        return None
    return _PLATFORM_IDS.get(name.lower().strip())


def _date_range(timeframe: str) -> str | None:
    """Build RAWG's `dates` query value (YYYY-MM-DD,YYYY-MM-DD).
    None for all-time."""
    today = datetime.now()
    if timeframe == "year":
        start = today - timedelta(days=365)
    elif timeframe == "all":
        return None
    else:
        # default: 'recent' → 90 days
        start = today - timedelta(days=90)
    return f"{start.strftime('%Y-%m-%d')},{today.strftime('%Y-%m-%d')}"


def _platforms_str(game: dict) -> str:
    """Compact platform list: 'PS5, Switch, PC'. Capped at 4 to keep voice
    summaries short — Claude can ask for more if needed."""
    plats = game.get("platforms") or []
    names = [
        (p.get("platform") or {}).get("name")
        for p in plats
        if (p.get("platform") or {}).get("name")
    ]
    if not names:
        return ""
    if len(names) > 4:
        return ", ".join(names[:4]) + f" (+{len(names) - 4} more)"
    return ", ".join(names)


def _release_str(game: dict) -> str:
    """Human-readable release: '2026-09-15' or 'TBA'. RAWG sets `tba: true`
    when the date is announced as 'TBA' upstream; `released` may be None or
    a placeholder date in that case."""
    if game.get("tba"):
        return "TBA"
    rel = game.get("released")
    return rel or "unannounced"


def _format_brief(game: dict) -> str:
    """One-line voice summary: 'Name (Platforms) — released YYYY-MM-DD, Metacritic 87'."""
    name = game.get("name") or "Unknown"
    rel = _release_str(game)
    plats = _platforms_str(game)
    score = game.get("metacritic")
    parts = [name]
    if plats:
        parts.append(f"({plats})")
    parts.append(f"— {rel}")
    if score:
        parts.append(f"Metacritic {score}")
    return " ".join(parts)


def _truncate(text: str, limit: int = 600) -> str:
    """Truncate at a sentence boundary near the limit. Description fields can
    be paragraphs long — Claude summarizes anyway, but smaller payloads keep
    cache-write costs down."""
    if not text or len(text) <= limit:
        return text or ""
    cut = text[:limit]
    last_period = cut.rfind(". ")
    if last_period > limit // 2:
        return cut[: last_period + 1]
    return cut + "…"


def _search_top_id(query: str, key: str) -> int | None:
    """Resolve a name to a RAWG game ID via the search endpoint. Used by
    `details` and `similar` modes, which both need an ID."""
    r = _http_get_with_retry(
        f"{_RAWG_BASE}/games",
        params={"search": query, "page_size": 1, "key": key},
    )
    if r is None:
        return None
    try:
        data = r.json()
    except ValueError as exc:
        print(f"[games] search JSON parse failed: {exc}", file=sys.stderr)
        return None
    results = data.get("results") or []
    if not results:
        return None
    return results[0].get("id")


def _do_search(query: str, key: str) -> str:
    r = _http_get_with_retry(
        f"{_RAWG_BASE}/games",
        params={"search": query, "page_size": 5, "key": key},
    )
    if r is None:
        return f"Game search unavailable for '{query}'."
    try:
        data = r.json()
    except ValueError:
        return f"Game search returned malformed data for '{query}'."
    results = data.get("results") or []
    if not results:
        return f"No games found matching '{query}'."
    return "\n".join(_format_brief(g) for g in results)


def _do_details(query: str, key: str) -> str:
    """Two-hop: search → resolve top ID → fetch full details."""
    game_id = _search_top_id(query, key)
    if game_id is None:
        return f"No game found matching '{query}'."

    r = _http_get_with_retry(f"{_RAWG_BASE}/games/{game_id}", params={"key": key})
    if r is None:
        return f"Game details unavailable for '{query}'."
    try:
        g = r.json()
    except ValueError:
        return f"Game details returned malformed data for '{query}'."

    name = g.get("name") or query
    rel = _release_str(g)
    plats = _platforms_str(g)
    score = g.get("metacritic")
    devs = ", ".join(d.get("name", "") for d in (g.get("developers") or [])[:2])
    pubs = ", ".join(p.get("name", "") for p in (g.get("publishers") or [])[:2])
    genres = ", ".join(gn.get("name", "") for gn in (g.get("genres") or [])[:3])
    esrb = (g.get("esrb_rating") or {}).get("name")
    playtime = g.get("playtime")  # average hours
    desc = _truncate(g.get("description_raw") or "", 600)

    lines = [f"{name}"]
    lines.append(f"Released: {rel}")
    if plats:
        lines.append(f"Platforms: {plats}")
    if score:
        lines.append(f"Metacritic: {score}")
    if genres:
        lines.append(f"Genres: {genres}")
    if devs:
        lines.append(f"Developer: {devs}")
    if pubs and pubs != devs:
        lines.append(f"Publisher: {pubs}")
    if esrb:
        lines.append(f"ESRB: {esrb}")
    if playtime:
        lines.append(f"Average playtime: {playtime} hours")
    if desc:
        lines.append(f"\n{desc}")
    return "\n".join(lines)


def _do_popular(platform: str | None, timeframe: str, key: str) -> str:
    """Most-added recent games. `-added` is RAWG's best 'what's hot now'
    signal — counts users adding the game to their lists, more responsive
    than metacritic (which lags) or rating (which favors all-time classics)."""
    params: dict[str, Any] = {
        "ordering": "-added",
        "page_size": _MAX_RESULTS,
        "key": key,
    }
    pid = _platform_id(platform)
    if pid:
        params["platforms"] = pid
    elif platform:
        # Unknown platform name — let Claude know rather than silently dropping
        return (
            f"Unknown platform '{platform}'. Try: switch, ps5, ps4, "
            f"xbox-series, xbox-one, pc, ios, android."
        )
    dates = _date_range(timeframe)
    if dates:
        params["dates"] = dates

    r = _http_get_with_retry(f"{_RAWG_BASE}/games", params=params)
    if r is None:
        return "Game service unavailable."
    try:
        data = r.json()
    except ValueError:
        return "Game service returned malformed data."
    results = data.get("results") or []
    if not results:
        return "No popular games found for that filter."
    descriptor = platform.lower() if platform else "across all platforms"
    period = {
        "recent": "last 90 days",
        "year": "past year",
        "all": "all-time",
    }.get(timeframe, "last 90 days")
    header = f"Most popular on {descriptor} ({period}):"
    return header + "\n" + "\n".join(_format_brief(g) for g in results)


def _do_similar(query: str, key: str) -> str:
    """Three-hop: search → fetch /games/{id} for genres+tags → query /games
    filtered by those genres+tags, ordered by -added (popularity).

    Why not /games/{id}/suggested: RAWG moved that endpoint behind their paid
    Business API tier (free keys get a consistent 401 there). The genre+tag
    fallback approximates what /suggested does internally — RAWG's own
    similarity is built on tag/genre overlap plus collaborative-filtering
    signals — and the free /games filter still gives us a fresh-data result
    rather than falling back to Claude's training cutoff.
    """
    game_id = _search_top_id(query, key)
    if game_id is None:
        return f"No game found matching '{query}' — can't suggest similar titles."

    detail_r = _http_get_with_retry(
        f"{_RAWG_BASE}/games/{game_id}", params={"key": key}
    )
    if detail_r is None:
        return f"Recommendation service unavailable for '{query}'."
    try:
        g = detail_r.json()
    except ValueError:
        return f"Recommendation service returned malformed data for '{query}'."

    # Top 3 of each — narrow enough to keep recommendations on-theme without
    # making the filter so strict it returns nothing. Genres are coarse
    # (Action, RPG); tags are specific (survival-horror, cyberpunk, atmospheric).
    # Combining both lets RAWG's filter behave as "broadly this kind of game,
    # specifically these flavors."
    genre_slugs = ",".join(
        gn.get("slug", "") for gn in (g.get("genres") or [])[:3] if gn.get("slug")
    )
    tag_slugs = ",".join(
        t.get("slug", "") for t in (g.get("tags") or [])[:3] if t.get("slug")
    )

    if not genre_slugs and not tag_slugs:
        return f"Not enough metadata to find games similar to '{query}'."

    params: dict[str, Any] = {
        "ordering": "-added",
        "page_size": _MAX_RESULTS,
        "key": key,
        "exclude_games": str(game_id),
    }
    if genre_slugs:
        params["genres"] = genre_slugs
    if tag_slugs:
        params["tags"] = tag_slugs

    list_r = _http_get_with_retry(f"{_RAWG_BASE}/games", params=params)
    if list_r is None:
        return f"Recommendation service unavailable for '{query}'."
    try:
        list_data = list_r.json()
    except ValueError:
        return f"Recommendation service returned malformed data for '{query}'."

    results = list_data.get("results") or []
    if not results:
        return f"No similar games found for '{query}'."
    return f"Games similar to '{query}':\n" + "\n".join(
        _format_brief(rg) for rg in results
    )


def execute_games_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises."""
    key = _api_key()
    if not key:
        return (
            "Game lookups require a RAWG API key. Set RAWG_API_KEY in .env "
            "(get a free key at rawg.io/apidocs)."
        )

    mode = (params.get("mode") or "search").lower().strip()
    query = (params.get("query") or "").strip()
    platform = params.get("platform")
    timeframe = (params.get("timeframe") or "recent").lower().strip()
    if timeframe not in ("recent", "year", "all"):
        timeframe = "recent"

    if mode in ("search", "details", "similar") and not query:
        return f"A game name is required for mode '{mode}'."

    if mode == "search":
        return _do_search(query, key)
    if mode == "details":
        return _do_details(query, key)
    if mode == "popular":
        return _do_popular(platform, timeframe, key)
    if mode == "similar":
        return _do_similar(query, key)
    return f"Unknown mode '{mode}'. Use search, details, popular, or similar."
