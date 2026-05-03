"""Anthropic SDK client: streaming Claude responses with prompt caching on the system prompt.

Caching note: Sonnet 4.6's minimum cacheable prefix is 2048 tokens. The current
JARVIS_SYSTEM_PROMPT is well below that, so cache_control is currently a no-op
(cache_creation/read tokens will be 0). The wiring is correct — caching will
activate automatically once the system prompt grows past the threshold.

Multi-turn: caller passes the full history (alternating user/assistant) plus
the new user message as the last entry. The generator yields text chunks; the
caller is responsible for appending the assistant response to the history.
"""

from __future__ import annotations

import sys
from typing import Iterator

import anthropic


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

Knowledge limits:
- You do not have live data (current weather, time, news) unless explicitly told.
- If asked, briefly say you don't have access, and offer the closest thing you can do."""


def stream_response(
    api_key: str,
    messages: list[dict],
    model: str = "claude-sonnet-4-6",
) -> Iterator[str]:
    """Stream Claude's response. `messages` must be the full alternating history
    ending with a user message. Yields text chunks; caller assembles + appends to history."""
    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": JARVIS_SYSTEM_PROMPT,
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
            f"history_msgs={len(messages)}",
            file=sys.stderr,
        )
