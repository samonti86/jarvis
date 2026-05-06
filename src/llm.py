"""Anthropic SDK client: streaming Claude with prompt caching, web_search,
and client-side tools (get_sports_info as of M13).

Two flavors of tool use coexist here:

  Server-side (web_search): Anthropic runs it. We just declare the tool and
  the model calls it transparently. The text stream pauses 1-2s, then
  resumes — text_stream handles this for us. No client involvement.

  Client-side (get_sports_info): the model emits a `tool_use` block, the
  stream stops with `stop_reason="tool_use"`, and we have to:
    1. execute the tool locally
    2. append the assistant turn (full content) + a user turn (tool_result)
    3. restart the stream so Claude can continue with the data
  This is the "agentic loop" pattern and is the standard Anthropic SDK shape
  for any client-side tool. Once we have it, adding more client-side tools
  (Plex, Home Assistant, etc.) is a one-line dispatch addition.

Caching note: Sonnet 4.6's minimum cacheable prefix is 2048 tokens. The
JARVIS_SYSTEM_PROMPT alone is well below that, but as recent-conversation
summaries get injected (M10), the prompt grows toward the threshold and
caching activates automatically.

Multi-turn: caller passes the full history (alternating user/assistant) plus
the new user message as the last entry. The generator yields text chunks; the
caller is responsible for appending the assistant response to the history.

Long-term memory: caller can pass `summaries` (a list of SummaryRecord from
src.memory). They get formatted with relative timestamps and appended to the
system prompt under a "Recent conversations" section, giving Jarvis context
about what was discussed in earlier (now-sealed) sessions.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Iterator

import anthropic

from src.games import GAMES_TOOL, execute_games_tool
from src.memory import SummaryRecord, format_summaries_for_prompt
from src.plex_mcp import PlexMCPClient
from src.sports import SPORTS_TOOL, execute_sports_tool
from src.weather import WEATHER_TOOL, execute_weather_tool


JARVIS_SYSTEM_PROMPT = """You are Jarvis, a personal voice assistant in the spirit of Tony Stark's J.A.R.V.I.S.

Tone:
- Courteous, dryly witty, understated. Closer to the films' calm Jarvis than a parody.
- Address the user as "sir" only occasionally — not every sentence.
- Never apologize unnecessarily. Never over-explain.

Format (this is voice — replies are spoken aloud through TTS):
- Default to short, conversational responses. Long answers are tedious to listen to.
- Prefer short sentences. They have better prosody when spoken.
- No markdown, bullet points, code fences, or visual formatting — none of it survives TTS.
- Avoid URLs, file paths, and long digit strings. If you must, spell them out naturally.

Language:
- Reply in the same language the user spoke in. English in English; Spanish in Spanish.
- When replying in Spanish, use the formal usted form, and Mexican conventions (not Castilian).
- Match the cultural register: dry-witty British butler in English; formal, courteous gentleman in Spanish.

Conversation:
- You may receive prior turns of the current conversation. Treat them as ongoing context —
  the user can reference earlier exchanges with pronouns or follow-ups ("and what about Madrid?").
- Stay consistent with what you said before unless corrected.

Memory of past sessions:
- A "Recent conversations" section may appear below, summarizing earlier sessions.
- Memory is for what we DISCUSSED, not for what is CURRENTLY TRUE. Use it for context
  ("what did we talk about earlier?", "remember that thing about Docker?") — never as
  a source of truth for time-sensitive facts.
- If a past summary mentions a fact that can change — sports rosters, scores, weather,
  prices, news, "current" anything — treat it as STALE and fetch the fresh answer with
  the appropriate tool. Do NOT parrot the old summary back. The user asking the question
  again is itself a signal they want a current answer, not a memory recall.
- Don't volunteer the summaries unsolicited — only reference them when relevant to the question.
- If a memory isn't there, say so plainly. Don't invent or guess at past discussions.

Live information (you have five tools — pick the right one):
1. get_sports_info — for live scores, schedules, and recent results in major leagues
   (NFL, NBA, MLB, NHL, MLS, EPL, Champions League, NCAA football and basketball, WNBA,
   UFC, F1, PGA, ATP, WTA). ALWAYS prefer this over web_search for any sports query —
   it returns structured live data and is far more reliable than scraped web pages.
2. get_weather — for current weather, today's forecast, or a multi-day forecast for
   any city worldwide. ALWAYS prefer this over web_search for weather queries.
3. get_game_info — for video game release dates, summaries, popular titles, and
   recommendations across PlayStation, Nintendo, Xbox, PC, and mobile. ALWAYS prefer
   this over web_search for general game-info queries. Use mode=details when the user
   asks for a summary; mode=popular for "what's hot on <platform>"; mode=similar when
   the user names a game they liked and wants recommendations. NOTE: this tool covers
   reference data only — it cannot see the user's personal library, trophies, or
   playtime, so don't claim it can.
4. web_fetch — retrieves the full contents of a SPECIFIC URL or PDF. Use this when
   the user names a particular site or document ("check ESPN for Giants news", "what
   does IGN say about Super Mario", "summarize this PDF at <url>"). You may chain
   web_search → web_fetch when you need to find a URL first, then read it in depth.
5. web_search — for general info-finding when no specific source is named: news,
   market prices, recent releases, "who is the current X", anything that changes
   over time.

Tool-use rules:
- For TIME-SENSITIVE categories, ALWAYS prefer the appropriate tool over memory or
  training data. Even if a similar answer is in your "Recent conversations" memory or
  feels familiar from training, fetch again — the world has likely moved on.
- When the user names a specific website or asks about a PDF, prefer web_fetch (or
  web_search → web_fetch if you need to find the right URL on that site first).
- Do NOT call tools for things that don't change: math, geography, definitions,
  established historical facts, well-known general knowledge. Answer those directly.
- Don't pre-announce ("Let me check…") — just call the tool when needed and answer.
- After fetching, summarize in one or two sentences for voice. Don't read raw lists,
  URLs, or citations aloud. For multi-game results, mention the user's team if they
  named one, or a notable highlight; don't recite every game.

Knowledge limits:
- For questions outside the scope of your tools (your own internal state, future
  events, opinions you don't have), say so plainly rather than fabricating."""


# Appended to the system prompt only when a Plex MCP session is live. Kept
# separate so we don't promise tools that aren't there when Plex graceful-
# failed at startup.
_PLEX_PROMPT_ADDENDUM = """

Media library (Plex):
- You also have tools (prefixed by their MCP-server names) to query and control
  the user's personal Plex Media Server. Use them when the user asks about THEIR
  library, what's currently playing on their TVs/clients, or wants to control
  playback ("play X on the living room", "what's on my Plex right now?",
  "show me recently added films").
- Do NOT use Plex tools for general movie/TV info — for that, web_search /
  web_fetch are right. Plex tools are scoped to what the user owns.
- For voice replies, summarize Plex results briefly. Don't read full IDs, file
  paths, or long lists. If a search returns many matches, name a few and offer
  to narrow down."""


# Anthropic's server-side web search tool. Server-side = Anthropic runs it,
# we just declare it. The "_20260209" version includes dynamic filtering.
# Supported on Sonnet 4.6, Opus 4.6, Opus 4.7.
WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
}


# Anthropic's server-side web fetch tool. Pulls the full contents of a
# specific URL (HTML or PDF) and returns the parsed text. Pairs naturally
# with web_search (search to find a URL → fetch to read it in depth).
#
# We deliberately use _20250910 (the older version) rather than _20260209.
# The newer version's "dynamic filtering" feature runs Anthropic's code
# execution sandbox to filter big pages, which threads a `container_id`
# through the conversation — and our agentic loop in stream_response() does
# not track or forward that container_id. Mismatch surfaced in M19 testing
# as: "container_id is required when there are pending tool uses generated
# by code execution with tools." For voice queries (typical pages ≤ a few
# KB), dynamic filtering's token-saving benefit is small; for the rare
# multi-MB doc, the agentic loop's MAX_LOOP_ITERATIONS still bounds cost.
# Revisit if we ever add a "summarize this 200-page document" workflow.
WEB_FETCH_TOOL = {
    "type": "web_fetch_20250910",
    "name": "web_fetch",
}


# Cap on agentic-loop iterations. In practice voice queries finish in 1-2
# tool calls; 5 is generous safety. If we ever hit this we log it.
_MAX_LOOP_ITERATIONS = 5


def _format_today() -> str:
    """Cross-platform 'Sunday, May 4, 2026' (no leading zero on day)."""
    now = datetime.now()
    return now.strftime("%A, %B ") + str(now.day) + now.strftime(", %Y")


def build_system_prompt(
    summaries: list[SummaryRecord] | None = None,
    plex_available: bool = False,
) -> str:
    """Compose the system prompt with the current date and optional memory.

    The current date gives Claude a temporal anchor for reasoning about what's
    stale vs. current. We use date precision (not time) so the cache breakpoint
    invalidates at most once per day, not per turn.

    When `plex_available` is True, append a Plex section advertising the
    media-library tools. We gate this on actual session presence so a
    graceful-fail startup doesn't leave Claude believing in tools it can't call.
    """
    base = f"{JARVIS_SYSTEM_PROMPT}\n\nToday is {_format_today()}."
    if plex_available:
        base += _PLEX_PROMPT_ADDENDUM
    if not summaries:
        return base
    memory_block = format_summaries_for_prompt(summaries)
    return (
        base
        + "\n\nRecent conversations (for your memory — only mention these if relevant):\n"
        + memory_block
    )


def _execute_client_tool(
    name: str,
    tool_input: dict,
    plex_client: PlexMCPClient | None = None,
) -> str:
    """Dispatch a client-side tool call. Returns a string for Claude to consume.
    Errors become readable strings — never raises."""
    try:
        if name == "get_sports_info":
            return execute_sports_tool(tool_input)
        if name == "get_weather":
            return execute_weather_tool(tool_input)
        if name == "get_game_info":
            return execute_games_tool(tool_input)
        if plex_client is not None and name in plex_client.tool_names:
            return plex_client.call_tool(name, tool_input)
        return f"Unknown tool: {name}"
    except Exception as exc:
        # Defensive — existing tool executors swallow their own errors,
        # but a future tool might not. Don't let a tool exception kill the turn.
        print(f"[llm] tool '{name}' raised: {exc}", file=sys.stderr)
        return f"Tool error: {exc}"


def stream_response(
    api_key: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
    summaries: list[SummaryRecord] | None = None,
    plex_client: PlexMCPClient | None = None,
) -> Iterator[str]:
    """Stream Claude's response, handling client-side tool use transparently.

    Yields text chunks suitable for direct TTS feeding. The caller sees a single
    continuous stream of text even when one or more tool calls happen in the
    middle — each tool round-trip is invisible from the caller's perspective.

    `messages` must be the full alternating history ending with a user message.
    `summaries` (optional) prepends recent-session context to the system prompt.
    `plex_client` (optional, M21) — if a live Plex MCP session is provided,
    its tools are surfaced to Claude alongside the built-in tools.
    """
    client = anthropic.Anthropic(api_key=api_key)
    system_text = build_system_prompt(
        summaries, plex_available=plex_client is not None
    )
    system_param = [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    tools = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL, SPORTS_TOOL, WEATHER_TOOL, GAMES_TOOL]
    if plex_client is not None:
        tools.extend(plex_client.tools)

    # Mutable working copy — we append assistant + tool_result turns as the
    # agentic loop runs. The caller's `messages` list is left untouched.
    working = list(messages)

    # Telemetry accumulators across all iterations.
    web_searched = False
    web_fetched = False
    sports_called = False
    weather_called = False
    games_called = False
    paused = False
    plex_tools_called: set[str] = set()
    total_input = total_output = total_cache_read = total_cache_create = 0
    iterations = 0

    while iterations < _MAX_LOOP_ITERATIONS:
        iterations += 1

        with client.messages.stream(
            model=model,
            max_tokens=1024,
            system=system_param,
            messages=working,
            tools=tools,
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()

        # Accumulate token usage. cache_* fields are None on uncached turns.
        u = final.usage
        total_input += u.input_tokens
        total_output += u.output_tokens
        total_cache_read += u.cache_read_input_tokens or 0
        total_cache_create += u.cache_creation_input_tokens or 0

        # Track server-side tool use (web_search, web_fetch) for telemetry
        # only — the SDK has already executed them and the text stream above
        # included Claude's post-tool text.
        for block in final.content:
            if getattr(block, "type", None) != "server_tool_use":
                continue
            sname = getattr(block, "name", None)
            if sname == "web_search":
                web_searched = True
            elif sname == "web_fetch":
                web_fetched = True

        # Server-side loop hit its 10-iteration cap. Rare in voice; we log
        # and bail rather than try to manually resume.
        if final.stop_reason == "pause_turn":
            paused = True
            break

        # Normal end of turn — Claude is done responding.
        if final.stop_reason != "tool_use":
            break

        # Client-side tool use. Append the assistant turn (full content,
        # including any text + tool_use blocks Claude emitted) so the SDK
        # has the canonical record. Then run each tool and feed results back.
        working.append({"role": "assistant", "content": final.content})

        tool_results = []
        for block in final.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = block.name
            if name == "get_sports_info":
                sports_called = True
            elif name == "get_weather":
                weather_called = True
            elif name == "get_game_info":
                games_called = True
            elif plex_client is not None and name in plex_client.tool_names:
                plex_tools_called.add(name)
            result_text = _execute_client_tool(
                name, block.input or {}, plex_client=plex_client
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        working.append({"role": "user", "content": tool_results})
        # Loop continues — next stream iteration sees the tool result and
        # produces Claude's final answer (or another tool call).
    else:
        # while-else: ran out of iterations without hitting break. Log it so
        # we notice if it ever happens in real use.
        print(
            f"[llm] hit MAX_LOOP_ITERATIONS={_MAX_LOOP_ITERATIONS} agentic cap",
            file=sys.stderr,
        )

    extra = ""
    if web_searched:
        extra += " web_search=yes"
    if web_fetched:
        extra += " web_fetch=yes"
    if sports_called:
        extra += " sports_tool=yes"
    if weather_called:
        extra += " weather_tool=yes"
    if games_called:
        extra += " games_tool=yes"
    if plex_tools_called:
        extra += f" mcp_tools={','.join(sorted(plex_tools_called))}"
    if paused:
        extra += " PAUSED_TURN(10-iter cap)"
    if iterations > 1:
        extra += f" iters={iterations}"

    print(
        f"[llm] tokens: input={total_input} output={total_output} "
        f"cache_read={total_cache_read} cache_create={total_cache_create} "
        f"history_msgs={len(messages)} summaries={len(summaries) if summaries else 0}"
        f"{extra}",
        file=sys.stderr,
    )
