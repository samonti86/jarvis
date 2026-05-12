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
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterator

import anthropic

from src.diagnostics_collector import (
    RUN_PC_DIAGNOSTICS_COLLECTOR_TOOL,
    execute_run_pc_diagnostics_collector,
)
from src.file_reader import READ_LOCAL_FILE_TOOL, execute_read_local_file
from src.games import GAMES_TOOL, execute_games_tool
from src.memory import SummaryRecord, format_summaries_for_prompt
from src.pc_diagnostics import PC_DIAGNOSTICS_TOOL, execute_pc_diagnostics_tool
from src.plex_actions import PLEX_ACTION_TOOL, execute_plex_action
from src.plex_laptop import (
    PLEX_LAPTOP_HEALTH_TOOL,
    PLEX_LOGS_SEARCH_TOOL,
    PLEX_LOGS_TAIL_TOOL,
    PlexLaptopClient,
    execute_plex_laptop_health,
    execute_plex_logs_search,
    execute_plex_logs_tail,
)
from src.plex_mcp import PlexMCPClient
from src.sports import SPORTS_TOOL, execute_sports_tool
from src.system_control import SYSTEM_CONTROL_TOOL, execute_system_control_tool
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

Local PC control (you have four tools — pick the right one):
6. pc_diagnostics — read-only LIVE telemetry on THIS Windows PC: CPU, RAM, disk,
   processes, services, network, recent System event-log entries. Use for any
   "how is my PC doing", "what's slowing me down", "any recent errors", "is X
   service running" question. NEVER modifies state — call freely without
   confirmation.
7. system_control — fixed allowlist of safe actions on THIS PC: open_app,
   lock_workstation, volume_set, volume_mute, volume_unmute, screen_off,
   kill_process. Each action is individually scoped — there is NO arbitrary-
   command path.
8. read_local_file — read a text file on THIS PC that the user points you at:
   a config file, a log, a Dockerfile, ~/.ssh/config, an error log. Read-only.
   Whatever you read joins the conversation, so only read what the user asked
   about. Refuses binary files and private-key material.
9. run_pc_diagnostics_collector — a DEEP snapshot: collects host / security /
   package / event-log data into a bundle of text files and returns their
   paths. Use for "run a full diagnostic", "deep system check", "collect
   everything for a support ticket", or follow-up troubleshooting that needs
   more than the live pc_diagnostics snapshot. After it runs, use
   read_local_file on the specific bundle files relevant to the question.
   Slow (60-90s) and writes to disk — confirmation-gated.

Local-PC safety rules:
- For low-impact actions (open_app, lock_workstation, volume_*, screen_off)
  and any read_local_file call: briefly announce what you're about to do (or
  just do it), then call the tool. No confirmation needed for these.
- For kill_process AND run_pc_diagnostics_collector: ALWAYS ask the user to
  confirm in plain language first ("Confirm: terminate chrome.exe?" /
  "Confirm: run a full diagnostics collection? It takes about a minute."),
  wait for an explicit yes, THEN call with confirmed=true. The tools enforce
  this server-side — calling without confirmed=true returns a confirmation-
  required notice rather than acting.
- "Improve performance" / "fix my PC" style requests: start with diagnostics
  (pc_diagnostics for a live look, or run_pc_diagnostics_collector for a deep
  one) and report findings. Suggest remediations in plain language but DO NOT
  apply changes the user didn't explicitly ask for. You are the analyst; the
  user is the decider.
- pc_diagnostics is "right now"; run_pc_diagnostics_collector is "deep
  snapshot for follow-up". For a quick "any errors lately?" the live tool is
  enough — don't kick off a minute-long collection unless the user wants the
  depth or a ticket bundle.

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


# Appended only when engineer mode is on. Unlocks structured depth without
# changing the calm-butler tone — the user is a Technical Support Engineer /
# SRE and values trade-off analysis over brevity in this mode. Calibration
# bullet at the end keeps the model from lecturing on simple questions.
_ENGINEER_PROMPT_ADDENDUM = """

Engineer mode (deeper technical reasoning):
- This conversation is in engineer mode. The user is a Technical Support Engineer / SRE
  who values depth, precision, and trade-off analysis over brevity. The calm, dryly-witty
  tone still holds — engineer mode is depth, not chattiness.
- You may write longer, structured responses with paragraphs, bullet points, ordered
  steps, or code blocks where they help comprehension. Visual structure is allowed here
  even though normal voice mode forbids it.
- Lead with the answer, then back it up with reasoning. Don't bury the lede.
- Explain WHY, not just WHAT. When recommending an approach, surface the trade-offs:
  what alternatives you considered, why this choice over those, what could go wrong.
- For diagnostic / troubleshooting questions: think like a senior engineer pair-partner.
  Form a hypothesis, suggest the next diagnostic step, propose multiple approaches with
  their trade-offs. Don't jump to a single fix when several are viable.
- Push back if the user is about to do something inefficient or risky — name the better
  approach. Be direct, not preachy. Match their technical level (assume senior).
- Connect new concepts to what they already know (Linux, networking, sysadmin,
  containers, cybersecurity) when it helps the explanation land.
- Calibrate to context. A quick fix doesn't need a lecture. New territory, non-obvious
  choices, "why is this happening" questions warrant more depth. When in doubt, lean
  toward more explanation than less.
- For genuinely simple questions ("what's 2+2", "what's the weather"), keep it brief
  even in engineer mode. Depth is a tool, not a default for everything."""


# Appended only when a live SSH client to the Plex laptop is available.
# Same gating discipline as _PLEX_PROMPT_ADDENDUM — never promise tools that
# aren't actually wired up.
_PLEX_LAPTOP_PROMPT_ADDENDUM = """

Remote Plex laptop (over SSH):

Diagnostics (read-only):
- plex_logs_tail — last N lines of Plex Media Server.log on the Plex laptop.
  Use for "what's Plex up to right now?", "is Plex okay?".
- plex_logs_search — regex search of the same log. Use for "any transcoder
  errors today?", "any 401s?", "any streaming failures recently?". The
  pattern is a .NET regex — alternation works ('error|fail|warning').
- plex_laptop_health — CPU/RAM/disk/network on the Plex laptop. Use for
  "how is the Plex box doing?", "is the Plex laptop drowning?", "what's the
  disk space on Plex?".
- For "is THIS PC vs the Plex laptop", remember pc_diagnostics is for THIS
  PC and plex_laptop_health is for the remote one.
- Voice summaries: when reading log lines aloud, paraphrase the gist
  ("a couple of transcoder warnings around 8 PM, otherwise clean") rather
  than reading raw timestamps and stack traces.

Actions (destructive, all confirmation-gated):
- plex_action — restart Plex Media Server, refresh a library, or empty
  Plex's trash for a library. ALL THREE require explicit user confirmation.
- For ANY plex_action call: ask the user to confirm in plain language
  first ("Confirm: restart Plex on the laptop?" / "Confirm: refresh the
  Movies library?"), wait for an explicit yes, THEN call plex_action with
  confirmed=true. The tool itself enforces this — if you call without
  confirmed=true it returns a confirmation-required notice rather than
  firing. Same pattern as kill_process on the local PC.
- For refresh_library and empty_trash you need a library_id. If you don't
  know it, call library_list (Plex MCP) first to enumerate sections.
- If the user asks for an action we don't support yet (rotate logs,
  restart the laptop itself, clear OS-level caches), say so plainly —
  diagnose first, propose remediations in words, but don't pretend to
  have a tool you don't."""


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

# Token budgets. Default mode is voice-shaped — short replies, no thinking.
# Engineer mode unlocks extended thinking + a generous output budget so the
# model can both reason and write the longer structured answer it produced.
_DEFAULT_MAX_TOKENS = 1024
_ENGINEER_THINKING_BUDGET = 5000
_ENGINEER_MAX_TOKENS = 8192   # must exceed thinking_budget; ~3k left for the actual reply


@dataclass
class TelemetryRecord:
    """Per-turn structured telemetry. Same data the existing stderr log line
    carries, exposed as a callable contract so the UI can surface it.

    Latency is wall-clock LLM-and-tools time only — TTS playback is excluded.
    For an SRE skimming the console, "how long did Jarvis spend thinking" is
    the more useful number than "how long did the whole turn take".
    """
    elapsed_sec: float
    iterations: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_create_tokens: int
    tools_used: list[str] = field(default_factory=list)
    paused: bool = False
    thinking_enabled: bool = False  # M26: extended thinking was active for this turn

    @property
    def total_tokens(self) -> int:
        # cache_read_tokens are billed at a discount but still count as
        # "input" semantically; include for the SRE-grade total.
        return self.input_tokens + self.output_tokens + self.cache_read_tokens


def _format_today() -> str:
    """Cross-platform 'Sunday, May 4, 2026' (no leading zero on day)."""
    now = datetime.now()
    return now.strftime("%A, %B ") + str(now.day) + now.strftime(", %Y")


def build_system_prompt(
    summaries: list[SummaryRecord] | None = None,
    plex_available: bool = False,
    plex_laptop_available: bool = False,
    engineer_mode: bool = False,
) -> str:
    """Compose the system prompt with the current date and optional memory.

    The current date gives Claude a temporal anchor for reasoning about what's
    stale vs. current. We use date precision (not time) so the cache breakpoint
    invalidates at most once per day, not per turn.

    Optional addenda are gated on actual tool availability so a graceful-fail
    startup doesn't leave Claude believing in tools it can't call:
    - `plex_available` (M21): advertise the Plex MCP media tools
    - `plex_laptop_available` (M24): advertise the remote-laptop SSH tools
    - `engineer_mode` (M26): unlock structured-depth replies and connect to
      the user's technical expertise. Toggleable per-turn from the tray.

    Note: engineer addendum is placed BEFORE the memory block so it stays in
    the cacheable prefix. Memory varies per turn; tool addenda + engineer
    addendum stay stable across turns within a session, which matters for
    prompt-cache hit rate.
    """
    base = f"{JARVIS_SYSTEM_PROMPT}\n\nToday is {_format_today()}."
    if plex_available:
        base += _PLEX_PROMPT_ADDENDUM
    if plex_laptop_available:
        base += _PLEX_LAPTOP_PROMPT_ADDENDUM
    if engineer_mode:
        base += _ENGINEER_PROMPT_ADDENDUM
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
    plex_laptop_client: PlexLaptopClient | None = None,
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
        if name == "pc_diagnostics":
            return execute_pc_diagnostics_tool(tool_input)
        if name == "system_control":
            return execute_system_control_tool(tool_input)
        if name == "read_local_file":
            return execute_read_local_file(tool_input)
        if name == "run_pc_diagnostics_collector":
            return execute_run_pc_diagnostics_collector(tool_input)
        if plex_laptop_client is not None and name in _PLEX_LAPTOP_TOOL_NAMES:
            return _PLEX_LAPTOP_DISPATCH[name](plex_laptop_client, tool_input)
        if plex_client is not None and name in plex_client.tool_names:
            return plex_client.call_tool(name, tool_input)
        return f"Unknown tool: {name}"
    except Exception as exc:
        # Defensive — existing tool executors swallow their own errors,
        # but a future tool might not. Don't let a tool exception kill the turn.
        print(f"[llm] tool '{name}' raised: {exc}", file=sys.stderr)
        return f"Tool error: {exc}"


# Static dispatch table for the SSH-backed tools (M24 read-only + M27
# actions) — keeps _execute_client_tool's main body unchanged in shape
# with the existing one-liner-per-tool pattern.
_PLEX_LAPTOP_DISPATCH = {
    "plex_logs_tail": execute_plex_logs_tail,
    "plex_logs_search": execute_plex_logs_search,
    "plex_laptop_health": execute_plex_laptop_health,
    "plex_action": execute_plex_action,
}
_PLEX_LAPTOP_TOOL_NAMES = frozenset(_PLEX_LAPTOP_DISPATCH.keys())


def stream_response(
    api_key: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
    summaries: list[SummaryRecord] | None = None,
    plex_client: PlexMCPClient | None = None,
    plex_laptop_client: PlexLaptopClient | None = None,
    on_complete: Callable[[TelemetryRecord], None] | None = None,
    engineer_mode: bool = False,
) -> Iterator[str]:
    """Stream Claude's response, handling client-side tool use transparently.

    Yields text chunks suitable for direct TTS feeding. The caller sees a single
    continuous stream of text even when one or more tool calls happen in the
    middle — each tool round-trip is invisible from the caller's perspective.

    `messages` must be the full alternating history ending with a user message.
    `summaries` (optional) prepends recent-session context to the system prompt.
    `plex_client` (optional, M21) — if a live Plex MCP session is provided,
    its tools are surfaced to Claude alongside the built-in tools.
    `plex_laptop_client` (optional, M24) — if SSH-reachable, the three remote
    diagnostic tools (logs_tail, logs_search, laptop_health) get registered.
    `on_complete` (optional) — called once at the end with a TelemetryRecord
    for UI consumption. The same data is also stderr-logged in the existing
    one-line format. Wrapped in try/except so a UI bug can't poison the turn.
    `engineer_mode` (optional, M26) — when True, append the engineer-mode
    addendum to the system prompt and enable Anthropic's extended thinking
    feature with a 5k-token reasoning budget. Captured per-turn from
    `ui.is_engineer_mode()`; mid-turn toggles apply to the next turn.
    """
    started_at = time.monotonic()
    client = anthropic.Anthropic(api_key=api_key)
    system_text = build_system_prompt(
        summaries,
        plex_available=plex_client is not None,
        plex_laptop_available=plex_laptop_client is not None,
        engineer_mode=engineer_mode,
    )
    system_param = [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    tools = [
        WEB_SEARCH_TOOL, WEB_FETCH_TOOL,
        SPORTS_TOOL, WEATHER_TOOL, GAMES_TOOL,
        PC_DIAGNOSTICS_TOOL, SYSTEM_CONTROL_TOOL,
        READ_LOCAL_FILE_TOOL, RUN_PC_DIAGNOSTICS_COLLECTOR_TOOL,
    ]
    if plex_laptop_client is not None:
        tools.extend([
            PLEX_LOGS_TAIL_TOOL,
            PLEX_LOGS_SEARCH_TOOL,
            PLEX_LAPTOP_HEALTH_TOOL,
            PLEX_ACTION_TOOL,
        ])
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
    diagnostics_called = False
    sysctl_called = False
    read_file_called = False
    collector_called = False
    paused = False
    plex_tools_called: set[str] = set()
    plex_laptop_tools_called: set[str] = set()
    total_input = total_output = total_cache_read = total_cache_create = 0
    iterations = 0

    # Build per-turn stream kwargs once; reuse across agentic-loop iterations.
    # max_tokens MUST exceed thinking budget when thinking is on; 8192 leaves
    # room for both the reasoning + a generous structured reply. Anthropic
    # also requires temperature=1 with extended thinking (we don't set it,
    # so default of 1 holds). Thinking blocks are preserved in the assistant
    # turn we append below — required when thinking + tool_use are combined.
    stream_kwargs: dict = {
        "model": model,
        "max_tokens": _ENGINEER_MAX_TOKENS if engineer_mode else _DEFAULT_MAX_TOKENS,
        "system": system_param,
        "messages": working,
        "tools": tools,
    }
    if engineer_mode:
        stream_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": _ENGINEER_THINKING_BUDGET,
        }

    while iterations < _MAX_LOOP_ITERATIONS:
        iterations += 1

        # Refresh messages each iteration since the agentic loop appends to
        # `working`. Other kwargs are stable across iterations.
        stream_kwargs["messages"] = working
        with client.messages.stream(**stream_kwargs) as stream:
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
            elif name == "pc_diagnostics":
                diagnostics_called = True
            elif name == "system_control":
                sysctl_called = True
            elif name == "read_local_file":
                read_file_called = True
            elif name == "run_pc_diagnostics_collector":
                collector_called = True
            elif name in _PLEX_LAPTOP_TOOL_NAMES:
                plex_laptop_tools_called.add(name)
            elif plex_client is not None and name in plex_client.tool_names:
                plex_tools_called.add(name)
            result_text = _execute_client_tool(
                name,
                block.input or {},
                plex_client=plex_client,
                plex_laptop_client=plex_laptop_client,
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
    if diagnostics_called:
        extra += " diagnostics=yes"
    if sysctl_called:
        extra += " sysctl=yes"
    if read_file_called:
        extra += " read_file=yes"
    if collector_called:
        extra += " diag_collector=yes"
    if plex_tools_called:
        extra += f" mcp_tools={','.join(sorted(plex_tools_called))}"
    if plex_laptop_tools_called:
        extra += f" plex_laptop_tools={','.join(sorted(plex_laptop_tools_called))}"
    if paused:
        extra += " PAUSED_TURN(10-iter cap)"
    if iterations > 1:
        extra += f" iters={iterations}"
    if engineer_mode:
        extra += " thinking=on"

    elapsed = time.monotonic() - started_at
    print(
        f"[llm] tokens: input={total_input} output={total_output} "
        f"cache_read={total_cache_read} cache_create={total_cache_create} "
        f"history_msgs={len(messages)} summaries={len(summaries) if summaries else 0} "
        f"elapsed={elapsed:.1f}s"
        f"{extra}",
        file=sys.stderr,
    )

    if on_complete is not None:
        # Build the structured record. Order matches "what an SRE wants to see
        # first": which tools fired (the verb), then how many turns of the
        # agentic loop, then time, then cost.
        tools_used: list[str] = []
        if web_searched:
            tools_used.append("web_search")
        if web_fetched:
            tools_used.append("web_fetch")
        if sports_called:
            tools_used.append("get_sports_info")
        if weather_called:
            tools_used.append("get_weather")
        if games_called:
            tools_used.append("get_game_info")
        if diagnostics_called:
            tools_used.append("pc_diagnostics")
        if sysctl_called:
            tools_used.append("system_control")
        if read_file_called:
            tools_used.append("read_local_file")
        if collector_called:
            tools_used.append("run_pc_diagnostics_collector")
        tools_used.extend(sorted(plex_laptop_tools_called))
        tools_used.extend(sorted(plex_tools_called))

        record = TelemetryRecord(
            elapsed_sec=elapsed,
            iterations=iterations,
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_tokens=total_cache_read,
            cache_create_tokens=total_cache_create,
            tools_used=tools_used,
            paused=paused,
            thinking_enabled=engineer_mode,
        )
        try:
            on_complete(record)
        except Exception as exc:
            # A UI bug must never break the listen loop.
            print(f"[llm] on_complete callback raised: {exc}", file=sys.stderr)
