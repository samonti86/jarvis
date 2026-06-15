"""M88 — conversation mode (full-duplex, Phase 1).

"Jarvis, let's talk" → he drops the wake-word-per-turn rhythm and stays in a
continuous, hands-free conversation: you speak, your pause ends the turn (the
existing VAD endpointing), he answers, and he's listening again instantly — no
"Hey Jarvis" between turns — until you say "that's all" / "exit conversation",
or a stretch of silence auto-exits him back to standby.

This is the M51 follow-up window made PERSISTENT, applied to normal Q&A (full
tools/persona/memory — unlike interpreter mode, which relays). It reuses the
entire turn pipeline; the listen loop just keeps `followup` effectively pinned
while this mode is on and re-arms on silence instead of dropping to the wake
word. Phase 2 (hands-free talk-over — interrupting him by just speaking, no wake
word) is the hard part and gated behind an AEC/echo feasibility spike; until
then, the M52 wake-word barge-in still works inside conversation mode.

This module is the pure, I/O-free core — the activation/exit intent matchers and
the spoken confirmations — so it's trivially unit-testable
(scripts/conversation_mode_test.py), like interpreter.py and
sound_detector._rule_window_passes.
"""

from __future__ import annotations

import re
import unicodedata

# Activation is matched EXACTLY (after stripping a leading wake-word/name and
# trailing politeness), NOT as a substring — because "let's talk ABOUT the
# Knicks" is a normal question, not a request to enter the mode. Exact-match on
# a short canonical phrase keeps a conversational "let's talk about X" from
# hijacking into mode-entry. (Interpreter mode could use substring safely
# because "be my interpreter" never prefixes a real question; "let's talk"
# does.)
_START_PHRASES = frozenset({
    "lets talk",
    "let us talk",
    "lets chat",
    "lets converse",
    "lets have a conversation",
    "lets have a chat",
    "lets keep talking",
    "lets keep chatting",
    "talk with me",
    "talk to me",
    "conversation mode",
    "lets have a conversation mode",
    # Spanish
    "vamos a hablar",
    "conversemos",
    "hablemos",
    "modo conversacion",
})

# Words stripped from the FRONT before the exact match (Whisper often prefixes
# the wake word / a filler).
_LEAD_STRIP = ("hey jarvis", "okay jarvis", "ok jarvis", "jarvis", "hey", "okay", "ok", "so", "well")
# Trailing politeness stripped before the exact match.
_TRAIL_STRIP = ("please", "jarvis", "buddy", "for a bit", "for a while", "a while", "now")

# Exit is matched as a SUBSTRING — these phrases are specific enough that a
# false positive is unlikely, and the user may wrap them ("okay Jarvis, let's
# exit conversation mode"). The natural sign-offs ("that's all", "goodbye", …)
# are handled by main._is_dismissal, which also exits the mode.
_STOP_PATTERNS = (
    "exit conversation",
    "leave conversation",
    "stop conversation",
    "end conversation",
    "conversation mode off",
    "stop the conversation mode",
    "done talking",
    "done chatting",
    "enough talking",
    "salir del modo conversacion",
    "termina la conversacion",
)


def _normalize(text: str) -> str:
    """Lowercase, DELETE apostrophes (so "let's" → "lets"), fold diacritics (so
    "conversación" → "conversacion"), turn other punctuation into spaces,
    collapse whitespace."""
    text = (text or "").replace("'", "").replace("’", "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_affixes(norm: str) -> str:
    """Strip a leading wake-word/filler and trailing politeness so the core
    phrase can be matched exactly. Applied repeatedly so "hey jarvis so ..."
    peels fully."""
    changed = True
    while changed:
        changed = False
        for lead in _LEAD_STRIP:
            if norm == lead:
                return ""
            if norm.startswith(lead + " "):
                norm = norm[len(lead) + 1:]
                changed = True
        for trail in _TRAIL_STRIP:
            if norm.endswith(" " + trail):
                norm = norm[: -(len(trail) + 1)]
                changed = True
    return norm.strip()


def is_start_intent(text: str) -> bool:
    """True only if `text` is (essentially) a bare request to enter conversation
    mode — exact match on a canonical phrase after affix stripping. "let's talk
    about the game" does NOT match (it carries a topic), so it routes to Claude
    as a normal question."""
    return _strip_affixes(_normalize(text)) in _START_PHRASES


def is_stop_intent(text: str) -> bool:
    """True if `text` explicitly asks to leave conversation mode. (Natural
    sign-offs like "that's all" are caught separately by _is_dismissal, which
    also exits the mode.)"""
    norm = _normalize(text)
    return any(p in norm for p in _STOP_PATTERNS)


# --- Spoken confirmations --------------------------------------------------
# Language-keyed so the confirmation matches the language the user activated in.

_START_CONFIRM = {
    "en": "Of course, sir. I'm listening — just speak, and say 'that's all' when you're done.",
    "es": "Por supuesto. Le escucho — hable cuando quiera, y diga 'eso es todo' cuando termine.",
}
_STOP_CONFIRM = {
    "en": "Of course, sir.",
    "es": "Por supuesto.",
}


def start_confirmation(lang: str = "en") -> str:
    return _START_CONFIRM.get(lang, _START_CONFIRM["en"])


def stop_confirmation(lang: str = "en") -> str:
    return _STOP_CONFIRM.get(lang, _STOP_CONFIRM["en"])
