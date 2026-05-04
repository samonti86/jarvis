"""Anthropic SDK client: streaming Claude responses with prompt caching on the system prompt.

Caching note: Sonnet 4.6's minimum cacheable prefix is 2048 tokens. The
JARVIS_SYSTEM_PROMPT alone is well below that, but as recent-conversation
summaries get injected (M10), the prompt grows toward the threshold and
caching will start to activate automatically.

Multi-turn: caller passes the full history (alternating user/assistant) plus
the new user message as the last entry. The generator yields text chunks; the
caller is responsible for appending the assistant response to the history.

Long-term memory: caller can pass `summaries` (a list of SummaryRecord from
src.memory). They get formatted with relative timestamps and appended to the
system prompt under a "Recent conversations" section, giving Jarvis context
about what was discussed in earlier (now-sealed) sessions.

Web search (M12): Anthropic's server-side web_search tool is enabled. Claude
decides when to use it (auto tool_choice). During a search the text stream
pauses for 1-2s, then resumes with the answer — text_stream handles this
transparently. The server-side sampling loop has a 10-iteration cap; if hit,
stop_reason is "pause_turn" and we'd need a manual continuation flow. Voice
queries that exhaust 10 search iterations are extremely unlikely; we detect
and log this case but don't auto-resume.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Iterator

import anthropic

from src.memory import SummaryRecord, format_summaries_for_prompt


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
  prices, news, "current" anything — treat it as STALE and use web_search to get the
  fresh answer. Do NOT parrot the old summary back. The user asking the question again
  is itself a signal they want a current answer, not a memory recall.
- Don't volunteer the summaries unsolicited — only reference them when relevant to the question.
- If a memory isn't there, say so plainly. Don't invent or guess at past discussions.

Live information (web_search tool):
- You have a web_search tool. Use it for time-sensitive questions you can't answer
  reliably from training data: current events, sports scores and rosters, weather,
  news, market prices, recent releases, "who is the current X", anything that
  changes over time.
- ALWAYS prefer web_search over memory or training data for these categories. Even if
  a similar answer appears in your "Recent conversations" memory or feels familiar
  from training, search again — the world has likely moved on.
- Do NOT search for things that don't change: math, geography, definitions,
  established historical facts, well-known general knowledge. Answer those directly.
- Don't pre-announce ("Let me check…") — just search when needed and answer.
- This is voice. After searching, summarize in one or two sentences. Don't read
  citations, URLs, or source names aloud. If sources disagree, give the most
  likely answer and note that reports vary.

Knowledge limits:
- For questions outside the scope of web search (your own internal state, future
  events, opinions you don't have), say so plainly rather than fabricating."""


# Anthropic's server-side web search tool. The "_20260209" version includes
# dynamic filtering — Claude writes code to filter results before they hit
# the context window. Supported on Sonnet 4.6, Opus 4.6, Opus 4.7.
WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
}


def _format_today() -> str:
    """Cross-platform 'Sunday, May 4, 2026' (no leading zero on day)."""
    now = datetime.now()
    return now.strftime("%A, %B ") + str(now.day) + now.strftime(", %Y")


def build_system_prompt(summaries: list[SummaryRecord] | None = None) -> str:
    """Compose the system prompt with the current date and optional memory.

    The current date gives Claude a temporal anchor for reasoning about what's
    stale vs. current. We use date precision (not time) so the cache breakpoint
    invalidates at most once per day, not per turn.
    """
    base = f"{JARVIS_SYSTEM_PROMPT}\n\nToday is {_format_today()}."
    if not summaries:
        return base
    memory_block = format_summaries_for_prompt(summaries)
    return (
        base
        + "\n\nRecent conversations (for your memory — only mention these if relevant):\n"
        + memory_block
    )


def stream_response(
    api_key: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
    summaries: list[SummaryRecord] | None = None,
) -> Iterator[str]:
    """Stream Claude's response. `messages` must be the full alternating history
    ending with a user message. `summaries` (optional) prepends recent-session
    context. Yields text chunks; caller assembles + appends to history."""
    client = anthropic.Anthropic(api_key=api_key)
    system_text = build_system_prompt(summaries)

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
        tools=[WEB_SEARCH_TOOL],
    ) as stream:
        for text in stream.text_stream:
            yield text

        final = stream.get_final_message()
        u = final.usage
        # Did Claude actually invoke web_search this turn? (For telemetry.)
        web_searched = any(
            getattr(b, "type", None) == "server_tool_use"
            and getattr(b, "name", None) == "web_search"
            for b in final.content
        )
        # The server tool loop has a 10-iteration cap. If hit, stop_reason is
        # "pause_turn" and the response is incomplete — we'd need to re-send
        # to continue. Voice queries that loop 10x are unlikely; just log it
        # so we notice if it ever happens in real use.
        paused = final.stop_reason == "pause_turn"
        extra = " web_search=yes" if web_searched else ""
        if paused:
            extra += " PAUSED_TURN(10-iter cap)"
        print(
            f"[llm] tokens: input={u.input_tokens} output={u.output_tokens} "
            f"cache_read={u.cache_read_input_tokens} cache_create={u.cache_creation_input_tokens} "
            f"history_msgs={len(messages)} summaries={len(summaries) if summaries else 0}"
            f"{extra}",
            file=sys.stderr,
        )
