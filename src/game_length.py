"""Game-length tool — "how long does it take to beat X?"

Backs onto How Long To Beat (howlongtobeat.com) via the community
`howlongtobeatpy` package — community-reported completion times for the main
story, main+extras, and 100% completionist runs. No API key (HLTB has no
public API; the package scrapes their search endpoint).

Why a SEPARATE tool from get_game_info (src/games.py) rather than a 5th mode:
the project's pattern is **one tool per data source** — RAWG→get_game_info,
TMDB→get_movie_tv_info, Wolfram→wolfram_query, RSS→get_news. HLTB is a new
source with its own (flaky) dependency, so it earns its own tool + module. The
M66/M73 "mode-vs-sibling" rule governs splits WITHIN a single backend; this is
across backends. (RAWG's get_game_info already returns a crude average
`playtime`; HLTB gives the real main/extra/completionist breakdown — the value
that makes this worth a tool.)

Same defensive contract as src/games.py / src/tmdb.py: never raises, always
returns a readable string. The howlongtobeatpy dependency is **lazy-imported
behind a sentinel** (like the M69 speaker-id / M39 face-auth stacks) so a
missing or broken install degrades to a clear message instead of taking down
the import of llm.py — HLTB scrapers break when the site changes, so isolating
this dep's failure is deliberate.

NOTE on latency: an HLTB search runs ~5 s (their search endpoint is slow). That
is acceptable for an explicit "how long is X" question (Claude streams a
preamble; the tool call is a one-shot lookup), but it's why this is not folded
into a hotter path.

NOTE on threading: howlongtobeatpy's sync `search()` spins its own asyncio loop
internally. The agentic tool loop runs in a synchronous worker thread with no
running event loop (the same thread that calls `asyncio.run(...)` for TTS), so
the sync call is safe here — verified during the M-build de-risk.
"""

from __future__ import annotations

import sys


# --- Anthropic tool definition ---------------------------------------------
GAME_LENGTH_TOOL = {
    "name": "get_game_length",
    "description": (
        "Get how long a video game takes to beat — typical completion times "
        "for the main story, main story plus extras, and a 100% "
        "completionist run, from How Long To Beat (community-reported "
        "playtimes). Use this for 'how long is X', 'how long to beat X', "
        "'how many hours is X'. This is about TIME TO COMPLETE a game; for a "
        "game's release date, platforms, summary, reviews, or "
        "recommendations use get_game_info instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The game's title. Required.",
            },
        },
        "required": ["name"],
    },
}


# --- lazy dependency load ---------------------------------------------------
# Cache the client across calls (instantiating it is cheap, but caching keeps
# the import attempt one-shot). `_unavailable` latches a failed import so we
# don't re-probe a broken/missing package on every call.
_hltb_client = None
_hltb_unavailable = False


def _get_client():
    """Return a cached HowLongToBeat client, or None if the package is
    unavailable. Import is attempted once; a failure latches so subsequent
    calls short-circuit to the friendly 'not installed' message."""
    global _hltb_client, _hltb_unavailable
    if _hltb_client is not None:
        return _hltb_client
    if _hltb_unavailable:
        return None
    try:
        from howlongtobeatpy import HowLongToBeat
    except ImportError as exc:
        print(f"[game_length] howlongtobeatpy not installed: {exc}",
              file=sys.stderr)
        _hltb_unavailable = True
        return None
    # Default minimum similarity (0.4) filters junk matches out of the result
    # list, so empty results genuinely mean "no good match".
    _hltb_client = HowLongToBeat()
    return _hltb_client


def _fmt_hours(value) -> str | None:
    """Format an HLTB hours figure for voice. HLTB returns floats (e.g.
    60.03) and 0.0 when it has no data for that playstyle. Round to the
    nearest half-hour and drop a trailing '.0'; None for missing/zero so the
    caller omits that line."""
    try:
        h = float(value)
    except (TypeError, ValueError):
        return None
    if h <= 0:
        return None
    r = round(h * 2) / 2  # nearest half-hour
    r = int(r) if r == int(r) else r
    return f"{r} hour" if r == 1 else f"{r} hours"


def _format(entry) -> str:
    """Render an HLTBEntry as voice-friendly lines. Suppresses playstyles HLTB
    has no data for; falls back to the all-styles average if the breakdown is
    entirely empty (rare — usually a just-announced game)."""
    name = getattr(entry, "game_name", None) or "that game"
    rows = [
        ("Main story", getattr(entry, "main_story", 0)),
        ("Main + extras", getattr(entry, "main_extra", 0)),
        ("Completionist", getattr(entry, "completionist", 0)),
    ]
    lines = []
    for label, hrs in rows:
        formatted = _fmt_hours(hrs)
        if formatted:
            lines.append(f"{label}: {formatted}")
    if not lines:
        avg = _fmt_hours(getattr(entry, "all_styles", 0))
        if avg:
            lines.append(f"All playstyles: {avg}")
    if not lines:
        return (
            f"How Long To Beat lists '{name}' but has no completion-time "
            "data yet, sir."
        )
    return f"How long to beat {name}:\n" + "\n".join(lines)


def execute_game_length_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises."""
    name = (params.get("name") or "").strip()
    if not name:
        return "A game title is required, sir."

    client = _get_client()
    if client is None:
        return (
            "Game-length lookups need the howlongtobeatpy package, which "
            "isn't installed. Run: pip install howlongtobeatpy."
        )

    try:
        results = client.search(name)
    except Exception as exc:  # noqa: BLE001 — HLTB scrape can fail many ways
        print(f"[game_length] search failed for {name!r}: {exc}",
              file=sys.stderr)
        return f"How Long To Beat lookup failed for '{name}', sir."

    if not results:
        return (
            f"I couldn't find a game called '{name}' on How Long To Beat, sir."
        )

    # Highest-similarity match wins (HLTB returns several candidates ranked by
    # a fuzzy similarity score). We surface the matched name so Claude/the user
    # can catch a wrong-game match.
    best = max(results, key=lambda e: getattr(e, "similarity", 0))
    return _format(best)
