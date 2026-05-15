"""Movie & TV info tool — release dates, summaries, trending titles, recommendations.

Backs onto The Movie Database (TMDB) API (themoviedb.org). Free tier, no
OAuth dance — just an `api_key=` query param (the v3 auth scheme), exactly
the same low-friction shape as RAWG in src/games.py. TMDB indexes essentially
every film and TV series with overviews, genres, cast/crew, ratings, runtime,
networks/studios, and a real recommendations endpoint.

This module is a deliberate structural twin of src/games.py (the M22 RAWG
tool): same four modes, same never-raises defensive contract, same
read-at-call-time API-key handling, same voice-shaped brief/detail
formatting. If you understand games.py you understand this; the only real
asymmetry is TMDB's movie-vs-TV field split (movies use `title`/
`release_date`, TV uses `name`/`first_air_date`), normalized by helpers here.

Same defensive contract as src/games.py / src/sports.py / src/weather.py:
never raises, always returns a readable string. Missing API key, no match,
network failure, malformed JSON all become voice-friendly error strings
Claude can paraphrase to the user.

Modes:
  search   — top 5 matches for a title (movies + TV, year + rating)
  details  — one title with full summary (auto-resolves name → top match)
  popular  — what's trending now, optionally filtered to movie or TV
  similar  — TMDB's recommendations for a title the user liked

Why these four match the voice query distribution (same logic as M22):
  search  → "Is there a movie called Mickey 17?" / "What year did Heat come out?"
  details → "Tell me about the show Severance."
  popular → "What's trending on TV this week?"
  similar → "I loved Arcane, what else should I watch?"
"""

from __future__ import annotations

import os
import sys
from functools import partial

from src.http_util import http_get_with_retry


# --- Anthropic tool definition ---------------------------------------------
TMDB_TOOL = {
    "name": "get_movie_tv_info",
    "description": (
        "Get information about movies and TV shows — release/air dates, "
        "plot summaries, cast, ratings, runtime, what's trending, and "
        "recommendations. Covers essentially every film and series. Use this "
        "for any movie or TV question EXCEPT the user's personal Plex library "
        "or their own watchlist/ratings (we don't have access to those — for "
        "the user's own library use the Plex tools instead). For a review on "
        "a specific outlet (a particular IGN or Rotten Tomatoes page), prefer "
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
                    "details = full summary of one title (best when the user "
                    "wants info on a specific movie or show); "
                    "popular = what's trending now, optional movie/tv filter; "
                    "similar = recommendations based on a title the user liked."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Movie or show title. Required for search, details, and "
                    "similar. Omit for popular."
                ),
            },
            "media": {
                "type": "string",
                "enum": ["movie", "tv"],
                "description": (
                    "Optional. Narrows search/popular and disambiguates "
                    "details/similar when a title exists as both a movie and "
                    "a show. Omit to let the tool pick the best match across "
                    "both."
                ),
            },
            "timeframe": {
                "type": "string",
                "enum": ["day", "week"],
                "description": (
                    "For mode=popular only: day = trending today; "
                    "week = trending this week (default)."
                ),
            },
        },
        "required": ["mode"],
    },
}


_TMDB_BASE = "https://api.themoviedb.org/3"

# Voice cap — Claude editorializes anyway, but we keep raw payloads short.
# Same value/reasoning as games.py's _MAX_RESULTS.
_MAX_RESULTS = 8

# M43: retry helper lives in src/http_util.py (shared across the data tools).
# partial() pre-binds our log tag so the _http_get_with_retry(...) call sites
# below read identically to games/sports/weather.
_http_get_with_retry = partial(http_get_with_retry, tag="tmdb")


# Spoken media words → TMDB's two media types. Voice says "film", "series",
# "show", not "movie"/"tv". Unknown / unspoken → None, which the modes treat
# as "no filter" (search both, trend across all) rather than erroring — for
# voice, silently widening is friendlier than rejecting a fuzzy word. (games.py
# errors on an unknown *platform* because there the user explicitly named one;
# here media is an optional narrowing the user often didn't say at all.)
_MEDIA_ALIASES: dict[str, str] = {
    "movie": "movie",
    "movies": "movie",
    "film": "movie",
    "films": "movie",
    "tv": "tv",
    "tv show": "tv",
    "tv shows": "tv",
    "show": "tv",
    "shows": "tv",
    "series": "tv",
    "tv series": "tv",
    "television": "tv",
}


def _api_key() -> str | None:
    """Read the TMDB key fresh each call — keeps it hot-swappable in dev
    without restarting Jarvis. Same read-at-call-time decision as games.py's
    RAWG key and M28's collector path: consumed only inside this module, and
    the Config dataclass is frozen=True anyway."""
    key = os.getenv("TMDB_API_KEY", "").strip()
    return key or None


def _media_filter(name: str | None) -> str | None:
    """Resolve a spoken media word to 'movie' / 'tv'. None if unknown or
    unset (caller treats None as 'no media filter')."""
    if not name:
        return None
    return _MEDIA_ALIASES.get(name.lower().strip())


def _title(item: dict) -> str:
    """Movies carry `title`, TV carries `name`. Fall back to the original-
    language variants, then a placeholder."""
    return (
        item.get("title")
        or item.get("name")
        or item.get("original_title")
        or item.get("original_name")
        or "Unknown"
    )


def _date_str(item: dict) -> str:
    """Movies: `release_date`. TV: `first_air_date`. TMDB sends '' (not null)
    for unannounced — normalize both to 'unannounced'."""
    raw = item.get("release_date") or item.get("first_air_date") or ""
    return raw or "unannounced"


def _media_label(item: dict, forced: str | None = None) -> str:
    """'Movie' / 'TV'. /search/multi and /trending tag each result with
    `media_type`; the typed endpoints (/search/movie, /movie/{id}) don't, so
    callers pass `forced` when they already know which endpoint they hit."""
    mt = forced or item.get("media_type")
    if mt == "tv":
        return "TV"
    if mt == "movie":
        return "Movie"
    # /search/movie omits media_type; presence of a TV-only field disambiguates.
    if item.get("first_air_date") is not None and item.get("release_date") is None:
        return "TV"
    return "Movie"


def _rating_str(item: dict) -> str:
    """'7.8/10 (12,345 votes)'. TMDB seeds unrated titles at 0.0/0 votes —
    suppress those so we don't read '0 out of 10' for an unreleased film."""
    avg = item.get("vote_average") or 0
    count = item.get("vote_count") or 0
    if not avg or not count:
        return ""
    return f"{avg:.1f}/10 ({count:,} votes)"


def _runtime_str(minutes: int | None) -> str:
    """155 → '2h 35m', 47 → '47m'. Voice reads 'two hours thirty-five'
    far better than 'one hundred fifty-five minutes'."""
    if not minutes or minutes <= 0:
        return ""
    h, m = divmod(int(minutes), 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


def _truncate(text: str, limit: int = 600) -> str:
    """Truncate at a sentence boundary near the limit. Identical policy to
    games.py._truncate — overviews can be a paragraph; Claude summarizes
    anyway, but smaller payloads keep prompt-cache write costs down."""
    if not text or len(text) <= limit:
        return text or ""
    cut = text[:limit]
    last_period = cut.rfind(". ")
    if last_period > limit // 2:
        return cut[: last_period + 1]
    return cut + "…"


def _format_brief(item: dict, forced_media: str | None = None) -> str:
    """One-line voice summary: 'Dune (Movie) — 2021-10-22, 7.8/10 (18,432 votes)'.
    Genres are intentionally omitted here (search/trending payloads only carry
    `genre_ids`, not names — resolving them costs an extra call per result);
    `details` mode carries the named genres. Same lean-brief tradeoff as
    games.py._format_brief."""
    name = _title(item)
    label = _media_label(item, forced_media)
    date = _date_str(item)
    parts = [f"{name} ({label})", f"— {date}"]
    rating = _rating_str(item)
    if rating:
        parts.append(rating)
    return " ".join(parts)


def _search_results(query: str, key: str, media: str | None) -> list[dict] | None:
    """One search hop. media=None → /search/multi (movies + TV + people);
    media set → the typed endpoint (more precise, no person noise). Returns
    the raw result list, with person results stripped from multi (a
    'what has X been in' feature is deliberately out of scope — see module
    docstring / M43 notes). None on any transport/parse failure."""
    if media in ("movie", "tv"):
        endpoint = f"{_TMDB_BASE}/search/{media}"
    else:
        endpoint = f"{_TMDB_BASE}/search/multi"
    r = _http_get_with_retry(
        endpoint, params={"query": query, "page": 1, "api_key": key}
    )
    if r is None:
        return None
    try:
        data = r.json()
    except ValueError as exc:
        print(f"[tmdb] search JSON parse failed ({query}): {exc}", file=sys.stderr)
        return None
    results = data.get("results") or []
    # /search/multi mixes in people; we only do titles. Typed endpoints are
    # already title-only so this is a no-op there.
    return [
        it for it in results
        if it.get("media_type") in (None, "movie", "tv")
        and (it.get("title") or it.get("name"))
    ]


def _search_top(
    query: str, key: str, media: str | None
) -> tuple[int, str] | None:
    """Resolve a name to (tmdb_id, 'movie'|'tv') via the top search result.
    Used by `details` and `similar`, which both need an ID and which detail
    endpoint to hit. None if nothing matched."""
    results = _search_results(query, key, media)
    if not results:
        return None
    top = results[0]
    tid = top.get("id")
    if tid is None:
        return None
    # Typed search has no media_type; infer it from the filter we used.
    mt = top.get("media_type") or media or "movie"
    if mt not in ("movie", "tv"):
        mt = "movie"
    return int(tid), mt


def _do_search(query: str, key: str, media: str | None) -> str:
    results = _search_results(query, key, media)
    if results is None:
        return f"Movie/TV search unavailable for '{query}'."
    if not results:
        return f"No movies or shows found matching '{query}'."
    return "\n".join(_format_brief(it) for it in results[:5])


def _do_details(query: str, key: str, media: str | None) -> str:
    """Two-hop: search → resolve top (id, media_type) → fetch full details
    with credits appended in the same request."""
    resolved = _search_top(query, key, media)
    if resolved is None:
        return f"No movie or show found matching '{query}'."
    tid, mt = resolved

    r = _http_get_with_retry(
        f"{_TMDB_BASE}/{mt}/{tid}",
        params={"api_key": key, "append_to_response": "credits"},
    )
    if r is None:
        return f"Details unavailable for '{query}'."
    try:
        d = r.json()
    except ValueError:
        return f"Details returned malformed data for '{query}'."

    name = _title(d)
    label = "TV Series" if mt == "tv" else "Movie"
    genres = ", ".join(g.get("name", "") for g in (d.get("genres") or [])[:3])
    rating = _rating_str(d)
    overview = _truncate(d.get("overview") or "", 600)
    credits = d.get("credits") or {}
    cast = ", ".join(
        c.get("name", "") for c in (credits.get("cast") or [])[:3] if c.get("name")
    )

    lines = [f"{name} ({label})"]
    if mt == "tv":
        lines.append(f"First aired: {_date_str(d)}")
        status = d.get("status")
        if status:
            lines.append(f"Status: {status}")
        seasons = d.get("number_of_seasons")
        episodes = d.get("number_of_episodes")
        if seasons:
            ep = f", {episodes} episodes" if episodes else ""
            lines.append(f"Seasons: {seasons}{ep}")
        creators = ", ".join(
            c.get("name", "") for c in (d.get("created_by") or [])[:2] if c.get("name")
        )
        if creators:
            lines.append(f"Created by: {creators}")
        networks = ", ".join(
            n.get("name", "") for n in (d.get("networks") or [])[:2] if n.get("name")
        )
        if networks:
            lines.append(f"Network: {networks}")
    else:
        lines.append(f"Released: {_date_str(d)}")
        rt = _runtime_str(d.get("runtime"))
        if rt:
            lines.append(f"Runtime: {rt}")
        director = ", ".join(
            c.get("name", "")
            for c in (credits.get("crew") or [])
            if c.get("job") == "Director" and c.get("name")
        )
        if director:
            lines.append(f"Director: {director}")
        studios = ", ".join(
            p.get("name", "")
            for p in (d.get("production_companies") or [])[:2]
            if p.get("name")
        )
        if studios:
            lines.append(f"Studio: {studios}")

    if genres:
        lines.append(f"Genres: {genres}")
    if rating:
        lines.append(f"Rating: {rating}")
    if cast:
        lines.append(f"Cast: {cast}")
    if overview:
        lines.append(f"\n{overview}")
    return "\n".join(lines)


def _do_popular(media: str | None, timeframe: str, key: str) -> str:
    """TMDB's /trending endpoint. media=None → 'all' (movies + TV mixed,
    people filtered out); media set → that type only. `week` is the better
    'what's hot' default — `day` is noisier and swings on a single big
    release."""
    scope = media if media in ("movie", "tv") else "all"
    window = timeframe if timeframe in ("day", "week") else "week"
    r = _http_get_with_retry(
        f"{_TMDB_BASE}/trending/{scope}/{window}", params={"api_key": key}
    )
    if r is None:
        return "Trending service unavailable."
    try:
        data = r.json()
    except ValueError:
        return "Trending service returned malformed data."
    results = [
        it for it in (data.get("results") or [])
        if it.get("media_type") in (None, "movie", "tv")
        and (it.get("title") or it.get("name"))
    ]
    if not results:
        return "No trending titles found."
    descriptor = {"movie": "movies", "tv": "TV shows"}.get(scope, "movies & TV")
    period = "today" if window == "day" else "this week"
    header = f"Trending {descriptor} {period}:"
    return header + "\n" + "\n".join(
        _format_brief(it) for it in results[:_MAX_RESULTS]
    )


def _do_similar(query: str, key: str, media: str | None) -> str:
    """Two-hop: search → resolve top (id, media_type) → TMDB's
    /recommendations for that title.

    Why /recommendations and not /similar: TMDB exposes both. `/similar` is a
    coarse keyword/genre overlap; `/recommendations` is TMDB's tuned engine
    (collaborative signals + metadata) and is the one their own UI surfaces.
    Both are free-tier — unlike RAWG's /suggested, which M22 found paywalled
    and had to work around with a 3-hop genre+tag fallback. TMDB gives us the
    good endpoint directly, so `similar` here is a clean 2-hop."""
    resolved = _search_top(query, key, media)
    if resolved is None:
        return f"No movie or show found matching '{query}' — can't suggest similar titles."
    tid, mt = resolved

    r = _http_get_with_retry(
        f"{_TMDB_BASE}/{mt}/{tid}/recommendations", params={"api_key": key}
    )
    if r is None:
        return f"Recommendation service unavailable for '{query}'."
    try:
        data = r.json()
    except ValueError:
        return f"Recommendation service returned malformed data for '{query}'."
    results = [
        it for it in (data.get("results") or [])
        if (it.get("title") or it.get("name"))
    ]
    if not results:
        return f"No recommendations found for '{query}'."
    # /<type>/{id}/recommendations returns only same-type results, so force
    # the label rather than relying on an absent media_type.
    return f"Because you liked '{query}':\n" + "\n".join(
        _format_brief(it, forced_media=mt) for it in results[:_MAX_RESULTS]
    )


def execute_tmdb_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises.
    Same shape as execute_games_tool: validate key + args up front, dispatch
    on mode, every failure path is a readable string."""
    key = _api_key()
    if not key:
        return (
            "Movie and TV lookups require a TMDB API key. Set TMDB_API_KEY in "
            ".env (get a free key at themoviedb.org/settings/api)."
        )

    mode = (params.get("mode") or "search").lower().strip()
    query = (params.get("query") or "").strip()
    media = _media_filter(params.get("media"))
    timeframe = (params.get("timeframe") or "week").lower().strip()
    if timeframe not in ("day", "week"):
        timeframe = "week"

    if mode in ("search", "details", "similar") and not query:
        return f"A movie or show title is required for mode '{mode}'."

    if mode == "search":
        return _do_search(query, key, media)
    if mode == "details":
        return _do_details(query, key, media)
    if mode == "popular":
        return _do_popular(media, timeframe, key)
    if mode == "similar":
        return _do_similar(query, key, media)
    return f"Unknown mode '{mode}'. Use search, details, popular, or similar."
