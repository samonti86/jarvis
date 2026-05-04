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
"""

from __future__ import annotations

import sys
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
- When the user references something from earlier ("what did we talk about this morning?",
  "remember that thing about Docker?"), check that section before saying you don't recall.
- Don't volunteer the summaries unsolicited — only reference them when relevant to the question.
- If a memory isn't there, say so plainly. Don't invent or guess at past discussions.

Knowledge limits:
- You do not have live data (current weather, time, news) unless explicitly told.
- If asked, briefly say you don't have access, and offer the closest thing you can do."""


def build_system_prompt(summaries: list[SummaryRecord] | None = None) -> str:
    """Compose the system prompt with optional memory of past sessions."""
    if not summaries:
        return JARVIS_SYSTEM_PROMPT
    memory_block = format_summaries_for_prompt(summaries)
    return (
        JARVIS_SYSTEM_PROMPT
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
    ) as stream:
        for text in stream.text_stream:
            yield text

        final = stream.get_final_message()
        u = final.usage
        print(
            f"[llm] tokens: input={u.input_tokens} output={u.output_tokens} "
            f"cache_read={u.cache_read_input_tokens} cache_create={u.cache_creation_input_tokens} "
            f"history_msgs={len(messages)} summaries={len(summaries) if summaries else 0}",
            file=sys.stderr,
        )
