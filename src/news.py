"""News headlines tool — current headlines by topic, from curated RSS feeds (M47).

The fifth sibling of the M13 client-side data-tool pattern (after
get_sports_info / get_weather / get_game_info / get_movie_tv_info). A
deliberate structural twin of src/tmdb.py: same never-raises defensive
contract, same read-shaped-for-voice formatting, same category-alias front
door.

Design decisions (settled with the user before the build):
  - **RSS, not a news API.** Curated feeds we control: zero API key, zero
    cost, no rate limits, and Jarvis only ever contacts the feed URLs we
    chose (privacy-aligned — same ethos as the local STT / FTS5 calls). The
    trade-off (no arbitrary "search all news for X") is accepted for v1;
    a keyword-search backend is a documented conditional follow-on.
  - **Topic-categorized.** `_FEEDS` is a {category: [urls]} map, so "what's
    the TECH news" is answerable, not just "what's the news". One dict vs.
    one list — trivial extra code for a whole class of voice queries.
  - **feedparser, not stdlib xml.etree.** Real-world feeds are a mess (RSS
    2.0 vs Atom vs RDF, CDATA, a dozen date formats, routinely malformed).
    feedparser is small, pure-Python, and robust to all of it; it never
    raises (sets a `bozo` flag instead), which is exactly the fail-soft
    behavior an always-on assistant needs. stdlib XML would mean hand-
    rolling a fragile parser — unlike M45's sqlite3, the stdlib option here
    is NOT a complete solution.

This is the first new consumer of src/http_util.py (the retry helper
extracted in M43): we fetch the feed bytes through the shared httpx
get-with-retry (consistent timeout / one-retry-on-transport-error / per-tool
log tag) and hand the raw bytes to feedparser. feedparser CAN fetch URLs
itself, but routing through http_util keeps every outbound data-tool call on
one policy.

Defensive contract, identical to src/tmdb.py / src/games.py: every public
entry point never raises and always returns a readable string Claude can
paraphrase. Network failure, a dead feed, an empty category, no matches —
all become voice-friendly strings, never an exception into the listen loop.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from functools import partial

import feedparser

from src.http_util import http_get_with_retry


# --- Anthropic tool definition ---------------------------------------------
NEWS_TOOL = {
    "name": "get_news",
    "description": (
        "Get current news headlines by topic from curated reputable RSS "
        "feeds (BBC, NPR, Ars Technica, The Verge, ESPN, and more). ALWAYS "
        "prefer this over web_search for general 'what's the news', "
        "'what's happening today', or 'what's the latest <topic> news' "
        "questions — it returns fresh structured headlines and is faster "
        "and more reliable than scraping. Categories: top (general "
        "headlines), world, tech, business, science, sports (NBA/NHL/NFL/WWE), "
        "gaming (PlayStation/Nintendo). Use web_search "
        "instead when the user wants to dig into one specific story, names "
        "a specific outlet/URL (then web_fetch), or asks about a niche "
        "topic these general feeds won't cover. NOTE: this reads a fixed "
        "set of feeds — the optional 'topic' only filters those current "
        "headlines, it is NOT a web-wide news search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["top", "world", "tech", "business", "science",
                         "sports", "gaming"],
                "description": (
                    "Which feed bucket. 'top' = general headlines (default "
                    "if the user just asks for 'the news'). 'sports' covers "
                    "NBA / NHL / NFL / WWE; 'gaming' covers PlayStation / "
                    "Nintendo."
                ),
            },
            "topic": {
                "type": "string",
                "description": (
                    "Optional. Case-insensitive keyword filter applied to "
                    "the fetched headlines in the chosen category (e.g. "
                    "'election', 'AI', 'Tesla'). Filters the current feeds "
                    "only — not a web search. Omit to get all top headlines."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Optional. Max headlines to return (default 5, max 10). "
                    "Voice replies want few; raise only if asked for more."
                ),
            },
        },
        "required": [],
    },
}


# Curated feed registry. The user "controls the sources" by editing this map
# — that's the deliberate RSS trade-off. Every URL here is a high-uptime,
# permissive feed that serves a plain httpx client without a custom User-
# Agent (http_util has no header arg; feeds that demand a UA were excluded on
# purpose to keep this dep-light). Add/replace freely; categories are the
# only thing the tool schema is coupled to.
_FEEDS: dict[str, list[str]] = {
    "top": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.npr.org/1001/rss.xml",
    ],
    "world": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.npr.org/1004/rss.xml",
    ],
    "tech": [
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.theverge.com/rss/index.xml",
        "https://hnrss.org/frontpage",
    ],
    "business": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.npr.org/1006/rss.xml",
    ],
    "science": [
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://feeds.npr.org/1007/rss.xml",
    ],
    # M55 — sports + gaming. Added for the "good morning" briefing, which
    # deliberately serves these in place of general/world news. Every URL
    # below was verified live (HTTP 200, parses, has entries) through
    # http_util's fetch path before being committed — two other WWE
    # candidates were dropped for 301/308 redirects (httpx doesn't follow
    # them; the M47 landmine). https-canonical, no-redirect URLs only.
    "sports": [
        "https://www.espn.com/espn/rss/nba/news",
        "https://www.espn.com/espn/rss/nhl/news",
        "https://www.espn.com/espn/rss/nfl/news",
        "https://www.wrestlinginc.com/feed/",
    ],
    "gaming": [
        "https://www.pushsquare.com/feeds/latest",
        "https://www.nintendolife.com/feeds/latest",
    ],
}

# Spoken words → canonical category. Voice says "technology", "world news",
# "the economy", "what's happening" — not the bare enum. Unknown / unspoken
# → 'top' (silently widening is friendlier than rejecting a fuzzy word; same
# call as tmdb._media_filter returning None for "no filter").
_CATEGORY_ALIASES: dict[str, str] = {
    "": "top",
    "top": "top", "headline": "top", "headlines": "top", "news": "top",
    "general": "top", "latest": "top", "today": "top", "breaking": "top",
    "world": "world", "world news": "world", "international": "world",
    "global": "world", "foreign": "world",
    "tech": "tech", "technology": "tech", "tech news": "tech",
    "gadgets": "tech", "computing": "tech",
    "business": "business", "finance": "business", "financial": "business",
    "markets": "business", "market": "business", "economy": "business",
    "economic": "business", "money": "business",
    "science": "science", "sci": "science", "scientific": "science",
    "space": "science", "research": "science", "environment": "science",
    "sports": "sports", "sport": "sports", "nhl": "sports", "hockey": "sports",
    "nfl": "sports", "football": "sports", "wwe": "sports",
    "wrestling": "sports", "nba": "sports", "basketball": "sports",
    "gaming": "gaming", "games": "gaming", "game": "gaming",
    "video games": "gaming", "videogames": "gaming",
    "playstation": "gaming", "nintendo": "gaming",
}

# Voice cap. Claude condenses for speech anyway; a lean payload also keeps
# prompt-cache write cost down (same value/reasoning as tmdb._MAX_RESULTS).
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 10
_PER_FEED_CAP = 12   # entries pulled per feed before the merge/sort/dedupe

_http_get_with_retry = partial(http_get_with_retry, tag="news")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _category(name: str | None) -> str | None:
    """Resolve a spoken category word to a canonical key. None only if the
    user named something we genuinely don't have a bucket for (caller turns
    that into a friendly 'I cover these topics' message rather than guessing
    — distinct from the unspoken case, which aliases to 'top')."""
    if name is None:
        return "top"
    key = _WS_RE.sub(" ", name.strip().lower())
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    # A single descriptive word we don't alias (e.g. "sports" — that's a
    # different tool) → None so the caller can say what it DOES cover.
    return None


def _strip_html(text: str) -> str:
    """Feed summaries are HTML fragments. Voice can't speak markup and Claude
    summarizes anyway, so flatten to text. Cheap regex strip is sufficient
    here — we're not rendering it, just giving Claude a clean gist."""
    if not text:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


def _truncate(text: str, limit: int = 240) -> str:
    """Trim a summary to a sentence boundary near the limit. Same policy as
    tmdb._truncate — Claude editorializes, but a lean payload is cheaper."""
    if not text or len(text) <= limit:
        return text or ""
    cut = text[:limit]
    dot = cut.rfind(". ")
    return cut[: dot + 1] if dot > limit // 2 else cut + "…"


def _entry_dt(entry) -> datetime | None:
    """feedparser normalizes dates to a UTC time.struct_time on
    `published_parsed` (falling back to `updated_parsed`). Some feeds carry
    neither — those sort last rather than erroring."""
    st = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if not st:
        return None
    try:
        return datetime(*st[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _age_str(dt: datetime | None) -> str:
    """'3h ago' / '2d ago' / 'just now'. Voice reads relative age far better
    than an ISO timestamp, and it's the part of 'when' that actually matters
    for news."""
    if dt is None:
        return ""
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 0:
        return "just now"
    if secs < 3600:
        m = int(secs // 60)
        return "just now" if m < 1 else f"{m}m ago"
    if secs < 86_400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86_400)}d ago"


def _source_name(parsed, url: str) -> str:
    """Human feed name from the feed's own <title>, falling back to the
    host. Used so a merged multi-feed list still attributes each headline."""
    title = ""
    feed = getattr(parsed, "feed", None)
    if feed is not None:
        title = (getattr(feed, "title", "") or "").strip()
    if title:
        # "BBC News - Home" / "NPR Topics: News" → "BBC News" / "NPR"
        return re.split(r"\s[-—:|]\s", title)[0].strip()
    m = re.search(r"://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else "source"


def _collect(category: str) -> list[dict]:
    """Fetch every feed in the category, parse, normalize entries. Per-feed
    failures are isolated — one dead feed must not blank the whole category
    (graceful degradation; the listen loop never sees an exception)."""
    items: list[dict] = []
    for url in _FEEDS.get(category, []):
        r = _http_get_with_retry(url)
        if r is None:
            continue  # http_util already logged the why
        # feedparser never raises — it sets .bozo on malformed input but
        # still returns whatever entries it could salvage. That fail-soft
        # behavior is exactly why it was chosen over stdlib XML.
        parsed = feedparser.parse(r.content)
        src = _source_name(parsed, url)
        for e in parsed.entries[:_PER_FEED_CAP]:
            title = _strip_html(getattr(e, "title", "")).strip()
            if not title:
                continue
            items.append(
                {
                    "title": title,
                    "source": src,
                    "dt": _entry_dt(e),
                    "summary": _truncate(
                        _strip_html(
                            getattr(e, "summary", "")
                            or getattr(e, "description", "")
                        )
                    ),
                }
            )
    return items


def _dedupe(items: list[dict]) -> list[dict]:
    """Drop repeat stories (wire copy shows up across feeds). Key on a
    normalized title; first occurrence (already newest after the sort) wins."""
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        key = re.sub(r"[^a-z0-9]+", "", it["title"].lower())[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _format(category: str, items: list[dict], topic: str | None) -> str:
    label = {
        "top": "Top headlines",
        "world": "World news",
        "tech": "Technology news",
        "business": "Business news",
        "science": "Science news",
        "sports": "Sports news",
        "gaming": "Gaming news",
    }.get(category, "Headlines")
    header = f"{label}" + (f" matching '{topic}'" if topic else "") + ":"
    lines = [header]
    for i, it in enumerate(items, 1):
        age = _age_str(it["dt"])
        meta = f"{it['source']}" + (f", {age}" if age else "")
        line = f"{i}. {it['title']} ({meta})"
        if it["summary"]:
            line += f" — {it['summary']}"
        lines.append(line)
    return "\n".join(lines)


def execute_news_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises.
    Same shape as execute_tmdb_tool: resolve the category alias, fetch +
    merge + sort + dedupe, optional topic filter, every failure path is a
    readable string."""
    category = _category(params.get("category"))
    if category is None:
        return (
            "I cover top headlines, world, tech, business, science, sports, "
            "and gaming news. Which would you like?"
        )

    topic = (params.get("topic") or "").strip()

    try:
        limit = int(params.get("limit") or _DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    limit = max(1, min(limit, _MAX_LIMIT))

    items = _collect(category)
    if not items:
        return (
            f"I couldn't reach the {category} news feeds just now, sir — "
            f"they may be temporarily unavailable."
        )

    # Newest first; undated entries sink to the bottom rather than erroring.
    items.sort(key=lambda it: it["dt"] or datetime.min.replace(tzinfo=timezone.utc),
               reverse=True)
    items = _dedupe(items)

    if topic:
        tl = topic.lower()
        filtered = [
            it for it in items
            if tl in it["title"].lower() or tl in it["summary"].lower()
        ]
        if not filtered:
            return (
                f"None of the current {category} headlines mention "
                f"'{topic}', sir. These feeds are a fixed set — for a "
                f"deeper search I'd need the web."
            )
        items = filtered

    return _format(category, items[:limit], topic or None)
