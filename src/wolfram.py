"""WolframAlpha computation tool — precise computational & quantitative
answers (M49).

The sixth sibling of the M13 client-side data-tool pattern (after
get_sports_info / get_weather / get_game_info / get_movie_tv_info /
get_news). Same never-raises defensive contract, same read-shaped-for-voice
output.

Why this tool exists — it covers a real gap, it doesn't duplicate Claude:
  - Claude's mental arithmetic is genuinely unreliable for multi-digit or
    multi-step math, and it has no live quantitative data. WolframAlpha is
    *authoritative* for exact computation, unit/currency conversion, date
    math, equations, calculus, statistics, and scientific/physical/
    astronomical data. This is the same complementary logic as web_search
    vs. training knowledge — use the authoritative source when being
    exactly right matters.

API choice — the Short Answers API (`/v1/result`):
  - It returns ONE plain-text line, which is exactly what a voice assistant
    wants: Claude paraphrases it into Jarvis's voice in one sentence.
  - Rejected alternatives: the Full Results API returns XML "pods" (rich but
    overkill — we'd just be flattening it back to one line); the Spoken
    Results API pre-phrases answers as "The answer is …", which fights the
    Jarvis persona Claude is supposed to supply.

Why this tool does NOT use src/http_util (deliberate, and the interesting
bit): the shared M43 retry helper collapses EVERY non-2xx response to None.
But WolframAlpha's Short Answers API returns **HTTP 501** as a *meaningful*
response — "Wolfram|Alpha did not understand your input" — not a transport
failure. We need to see that 501 distinctly so Claude gets an honest
"rephrase it or answer from your own knowledge" signal, as opposed to a
network problem ("try again later"). A shared helper has a contract; when an
API's semantics don't fit that contract, you make the direct call rather
than force-fit it. The one-retry-on-transport-error policy is still mirrored
from http_util for codebase consistency.

Defensive contract, identical to src/news.py / src/tmdb.py: the public entry
point never raises and always returns a readable string Claude can
paraphrase. Missing AppID, network failure, an uninterpretable query — all
become voice-friendly strings, never an exception into the listen loop.
"""

from __future__ import annotations

import os
import sys
import time

import httpx


# --- Anthropic tool definition ---------------------------------------------
WOLFRAM_TOOL = {
    "name": "wolfram_query",
    "description": (
        "Query WolframAlpha for precise computational and quantitative "
        "answers. Use this — NOT your own reasoning — whenever being "
        "EXACTLY right matters and the question is computational: "
        "multi-digit or multi-step arithmetic, unit and currency "
        "conversions, date/time calculations ('how many days until…', "
        "'what day of the week was…'), solving equations, calculus, "
        "statistics, and scientific / physical / astronomical data and "
        "constants. WolframAlpha is authoritative for exact computation; "
        "your mental arithmetic is not. Do NOT use it for trivial mental "
        "math (2+2, a simple percentage) — answer those directly — nor for "
        "things other tools own (weather → get_weather, sports → "
        "get_sports_info, news → get_news, the user's own setup → "
        "knowledge_search) nor for open-ended or current-events questions "
        "(use web_search). Returns one concise factual line; phrase the "
        "spoken reply in your own voice."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The computational question, in plain language or math "
                    "notation — e.g. 'integral of x^2 from 0 to 3', "
                    "'40 miles in kilometers', 'days until December 25 "
                    "2026', 'speed of light in mph', '17% of 4250'."
                ),
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": (
                    "Optional. Force the answer's unit system. Omit to let "
                    "WolframAlpha choose; set it only when the user clearly "
                    "wants one (e.g. 'in metric', 'give it to me in miles')."
                ),
            },
        },
        "required": ["query"],
    },
}


# Short Answers API. One endpoint, one plain-text line back. See the module
# docstring for why this API and not the Full/Spoken variants.
_BASE = "https://api.wolframalpha.com/v1/result"

# HTTP policy mirrors src/http_util: a generous 10s ceiling (longer than
# WolframAlpha's own internal compute timeout, so httpx never cuts off a
# valid slow computation) and a single 0.5s-backoff retry on transport
# errors only.
_TIMEOUT_SEC = 10.0
_RETRY_BACKOFF_SEC = 0.5

# Voice queries are short; a 500-char cap is generous and defends against an
# absurd input bloating the request.
_MAX_QUERY_LEN = 500


def _app_id() -> str | None:
    """Read the WolframAlpha AppID fresh each call — keeps it hot-swappable
    in dev without restarting Jarvis. Same read-at-call-time decision as
    games.py's RAWG key and tmdb.py's TMDB key (consumed only inside this
    module; the Config dataclass is frozen)."""
    key = os.getenv("WOLFRAM_APP_ID", "").strip()
    return key or None


def _fetch(query: str, units: str | None, app_id: str) -> tuple[str, str]:
    """GET the Short Answers API. Returns (status, text):

      ("ok",      answer)  — HTTP 200, a usable plain-text answer
      ("nounder", "")      — HTTP 501, WolframAlpha couldn't interpret it
      ("badkey",  "")      — HTTP 401/403, the AppID was rejected
      ("error",   "")      — transport failure, or any other status code

    Direct httpx (not src/http_util) so the meaningful 501 is visible — see
    the module docstring. One retry on transport error mirrors http_util.
    """
    params: dict[str, str] = {"appid": app_id, "i": query}
    if units:
        params["units"] = units

    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            r = httpx.get(_BASE, params=params, timeout=_TIMEOUT_SEC)
        except httpx.RequestError as exc:
            # Timeout / connect error / DNS — transient, worth one retry.
            last_exc = exc
            if attempt == 1:
                print(
                    f"[wolfram] {type(exc).__name__} — retrying in "
                    f"{_RETRY_BACKOFF_SEC}s",
                    file=sys.stderr,
                )
                time.sleep(_RETRY_BACKOFF_SEC)
                continue
            break

        if r.status_code == 200:
            return ("ok", r.text.strip())
        if r.status_code == 501:
            # NOT an error — WolframAlpha understood the request format but
            # has no interpretation. A meaningful, expected outcome.
            return ("nounder", "")
        if r.status_code in (401, 403):
            print(
                f"[wolfram] HTTP {r.status_code} — AppID rejected",
                file=sys.stderr,
            )
            return ("badkey", "")
        print(
            f"[wolfram] unexpected HTTP {r.status_code}", file=sys.stderr
        )
        return ("error", "")

    print(f"[wolfram] gave up after retry: {last_exc}", file=sys.stderr)
    return ("error", "")


def execute_wolfram_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises.
    Same shape as execute_news_tool / execute_tmdb_tool: every failure path
    is a readable, voice-friendly string."""
    query = (params.get("query") or "").strip()
    if not query:
        return "No query was provided for WolframAlpha, sir."
    if len(query) > _MAX_QUERY_LEN:
        query = query[:_MAX_QUERY_LEN]

    # The Short Answers API spells imperial as "nonmetric". Expose the
    # friendlier "imperial" in the schema, translate here; anything
    # unrecognized or unset → no units param (let WolframAlpha decide).
    units_in = (params.get("units") or "").strip().lower()
    if units_in == "imperial":
        units = "nonmetric"
    elif units_in == "metric":
        units = "metric"
    else:
        units = ""

    app_id = _app_id()
    if app_id is None:
        return (
            "WolframAlpha queries need an AppID, sir. Set WOLFRAM_APP_ID in "
            ".env — a free non-commercial key from "
            "developer.wolframalpha.com."
        )

    status, answer = _fetch(query, units or None, app_id)

    if status == "ok":
        return answer or (
            f"WolframAlpha returned an empty answer for '{query}', sir."
        )
    if status == "nounder":
        return (
            f"WolframAlpha couldn't interpret '{query}'. Try rephrasing it "
            f"as a more explicit computation — or, if you're confident, "
            f"answer it from your own knowledge."
        )
    if status == "badkey":
        return (
            "WolframAlpha rejected the AppID — check that WOLFRAM_APP_ID in "
            ".env is a valid key from developer.wolframalpha.com."
        )
    return (
        "I couldn't reach WolframAlpha just now, sir — it may be "
        "temporarily unavailable."
    )
