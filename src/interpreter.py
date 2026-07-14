"""M87 — interpreter mode: Jarvis as a live two-way interpreter.

The project was architected multilingual from day one (Whisper detects the
language per turn; text_to_speech.VOICE_BY_LANG maps en→a British male,
es→a formal Mexican male). Until now that only bought "Jarvis answers in the
language you spoke". Interpreter mode flips the turn pipeline: instead of
*answering* an utterance, Jarvis *translates* it into the OTHER language of a
configured pair and speaks it in that language's voice — so two people who
don't share a language can talk THROUGH Jarvis.

Motivating case: a household where not everyone speaks the same language, and
a conversation has to bridge that gap without either party learning the other's.

This module is the pure, model-free, I/O-free core — the language-pair logic,
the activation/exit intent matchers, the translation system prompt, and the
spoken confirmation lines. The actual capture→translate→speak loop lives in
main.py (listen_loop + TurnRunner.interpret), and the streaming translation
call is llm.stream_translation. Keeping the decisions here makes them trivially
unit-testable (scripts/interpreter_test.py), exactly like sound_detector's
_rule_window_passes.

Activation is voice-only by design: you say "Jarvis, be my interpreter" in
normal mode, the mode engages, and from then on EVERY utterance is translated
with no wake word — until you say "stop interpreting". (A tray toggle is a
clean follow-on; v1 is the standing-with-someone voice flow.)
"""

from __future__ import annotations

import os
import re
import unicodedata

# Human-readable names for the spoken confirmations + the translation prompt.
LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
}


def _parse_pair(raw: str) -> tuple[str, str]:
    """Parse JARVIS_INTERPRETER_LANGS ('en,es') into a 2-tuple. Falls back to
    the en/es default on anything malformed — a bad env var must never break
    activation. The FIRST element is the 'primary' (the user's language); the
    SECOND is the other party's (e.g. the Spanish-speaking family member)."""
    parts = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return ("en", "es")


# The configured language pair. (primary, secondary).
LANG_PAIR: tuple[str, str] = _parse_pair(os.getenv("JARVIS_INTERPRETER_LANGS", ""))


def other_language(spoken: str, pair: "tuple[str, str] | None" = None) -> str:
    """The language to translate INTO, given what was just spoken.

    Two-party model: if the secondary language was spoken, translate to the
    primary; otherwise translate to the secondary. So in the default en/es pair,
    Spanish → English and everything-else (English, or a Whisper mis-detect) →
    Spanish. That biases an ambiguous detection toward the other party's
    language, which is the safe direction for a conversation where the user
    drives in the primary language."""
    a, b = pair or LANG_PAIR
    if spoken == b:
        return a
    return b


# --- Activation / exit intent matching ------------------------------------
# Matched against the normalized transcript (lowercased, punctuation stripped,
# whitespace collapsed) as a substring — robust to Whisper's surrounding words
# ("Jarvis, could you be my interpreter please").

def _normalize(text: str) -> str:
    # Fold diacritics so the Spanish patterns match regardless of accents
    # ("intérprete" → "interprete") — \w keeps accented letters under Unicode,
    # so without this the accent-free patterns would never hit.
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)      # drop punctuation
    return re.sub(r"\s+", " ", text).strip()


# Deliberately mode-specific phrases — the noun "interpreter"/"translator" or an
# explicit "start interpreting" — so a one-off "how do you say X in Spanish"
# never trips it. Spanish variants too, in case the user activates in Spanish.
_START_PATTERNS = (
    "be my interpreter",
    "be my translator",
    "be an interpreter",
    "interpreter mode",
    "translator mode",
    "start interpreting",
    "start translating",
    "interpret for me",
    "translate for me",
    "modo interprete",          # normalized (accent stripped)
    "se mi interprete",
    "traduce para mi",
)

_STOP_PATTERNS = (
    "stop interpreting",
    "stop translating",
    "stop the interpreter",
    "stop interpreter mode",
    "exit interpreter",
    "end interpreter",
    "turn off the interpreter",
    "turn off interpreter",
    "deja de traducir",
    "deja de interpretar",
    "para de traducir",
    "para de interpretar",
)


def is_start_intent(text: str) -> bool:
    """True if `text` asks Jarvis to become the interpreter."""
    norm = _normalize(text)
    return any(p in norm for p in _START_PATTERNS)


def is_stop_intent(text: str) -> bool:
    """True if `text` asks Jarvis to leave interpreter mode."""
    norm = _normalize(text)
    return any(p in norm for p in _STOP_PATTERNS)


# --- Translation prompt ----------------------------------------------------

def build_translation_prompt(target_lang: str) -> str:
    """System prompt for the translate-only LLM call. The whole contract: emit
    ONLY the translation, nothing else — no persona, no "sir", no quotes — so
    speak_streaming reads out a clean relay. Formal usted for Spanish (the user
    addresses his family member respectfully). Distinct from JARVIS_SYSTEM_PROMPT on
    purpose: an interpreter is invisible."""
    name = LANG_NAMES.get(target_lang, target_lang)
    register = ""
    if target_lang == "es":
        register = (" Use Mexican Spanish and the formal 'usted' register, not "
                    "Castilian and not informal 'tú'.")
    return (
        "You are a live, two-way interpreter standing between two people who do "
        f"not share a language. Translate the user's message into {name}, "
        f"faithfully and completely.{register} Output ONLY the translation — no "
        "preamble, no quotation marks, no commentary, no explanation, no notes. "
        "Preserve the speaker's tone, meaning, names, and numbers. If the message "
        "is already in the target language, simply repeat it unchanged."
    )


# --- Spoken confirmations (start / stop) -----------------------------------
# Spoken in BOTH languages on entry so the other party (the non-English-speaking family member)
# also hears what's happening. The English line is always spoken; the Spanish
# line is added by the caller only when 'es' is in the configured pair.

START_CONFIRM_EN = (
    "Interpreter mode on, sir. I'll translate between you and the other person. "
    "Say 'stop interpreting' when you're finished."
)
START_CONFIRM_ES = (
    "Modo intérprete activado. Voy a traducir la conversación. "
    "Hábleme con naturalidad."
)
STOP_CONFIRM_EN = "Interpreter mode off, sir."
STOP_CONFIRM_ES = "Modo intérprete desactivado."
