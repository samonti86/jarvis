"""M72 — one-shot scene description for proactive multimodal alerts.

When Jarvis hears something notable while armed (M58 acoustic awareness), he
now LOOKS through the camera and says what he sees. This is the "look" half:
a single, non-streaming Claude vision call that takes a JPEG frame + what was
heard and returns ONE concise butler-tone sentence ("the office looks
undisturbed, no one's in view, sir"). The caller pushes that + the photo to
Discord (see main.py's acoustic visual-alert wiring + notifications.send_discord_photo).

Deliberately NOT the streaming agentic stream_response path — this is a leaf
call with no tools, no history, no TTS, fired off a background thread on a rare
event. Defensive contract: never raises; any failure (no key, network, API
error, odd response shape) returns None and the caller falls back to text-only.
"""

from __future__ import annotations

import base64
import sys
from typing import Optional


# Short cap — we want one or two sentences for a phone push, not an essay.
_MAX_TOKENS = 160


def build_prompt(heard: str) -> str:
    """The instruction paired with the frame. Pure + tested: keeps the wording
    in one place and lets a test assert the heard-event is woven in."""
    event = (heard or "a sound").replace("_", " ").strip() or "a sound"
    return (
        f"You are Jarvis, a home security assistant. While the home is armed you "
        f"just heard {event}. Look at this webcam frame from the room and report, "
        f"in ONE concise sentence, what you see that's relevant — focus on any "
        f"people, movement, or anything notable. If the room looks normal and "
        f"empty, say so plainly. British-butler tone, address the user as 'sir', "
        f"no preamble, no narration that you're looking at a photo."
    )


def describe_scene(
    jpeg_bytes: bytes,
    heard: str,
    *,
    api_key: str,
    model: str,
) -> Optional[str]:
    """One-shot Claude vision call. Returns a short sentence, or None on any
    failure (fail-soft — the caller then pushes the photo with fallback text)."""
    if not jpeg_bytes or not api_key:
        return None
    try:
        import anthropic  # noqa: PLC0415 — lazy; never block import on a bg feature
        client = anthropic.Anthropic(api_key=api_key)
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        msg = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    }},
                    {"type": "text", "text": build_prompt(heard)},
                ],
            }],
        )
        parts = [
            b.text for b in msg.content
            if getattr(b, "type", None) == "text" and getattr(b, "text", None)
        ]
        text = " ".join(parts).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — proactive feature must never raise
        print(f"[vision] describe_scene failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None
