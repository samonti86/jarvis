"""M87 — tests for interpreter mode (the pure core in src/interpreter.py).

Interpreter mode turns Jarvis into a live two-way translator so two people who
don't share a language can talk through him. The model-free, I/O-free cores:

  - other_language(spoken, pair): which language to translate INTO.
  - is_start_intent / is_stop_intent: the voice activation / exit matchers.
  - build_translation_prompt: the translate-only system prompt (no persona).
  - _parse_pair / LANG_PAIR: the JARVIS_INTERPRETER_LANGS config.

Hermetic: no model, no audio, no network.

    python tests/interpreter_test.py   # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import interpreter as it  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, condition: bool) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


EN_ES = ("en", "es")

# --- other_language: the two-party routing ---------------------------------
check("English spoken -> translate to Spanish",
      it.other_language("en", EN_ES) == "es")
check("Spanish spoken -> translate to English",
      it.other_language("es", EN_ES) == "en")
check("unknown/mis-detected language -> defaults to the secondary (es)",
      it.other_language("fr", EN_ES) == "es")
check("empty language -> defaults to the secondary (es)",
      it.other_language("", EN_ES) == "es")
# A custom pair routes symmetrically.
check("custom pair en/fr: English -> French",
      it.other_language("en", ("en", "fr")) == "fr")
check("custom pair en/fr: French -> English",
      it.other_language("fr", ("en", "fr")) == "en")
check("other_language falls back to module LANG_PAIR when pair omitted",
      it.other_language("en") in ("es", it.LANG_PAIR[1]))


# --- _parse_pair / config --------------------------------------------------
check("parse 'en,es' -> ('en','es')", it._parse_pair("en,es") == ("en", "es"))
check("parse with spaces/caps -> normalized", it._parse_pair(" EN , ES ") == ("en", "es"))
check("parse 'en,fr' -> ('en','fr')", it._parse_pair("en,fr") == ("en", "fr"))
check("parse empty -> default ('en','es')", it._parse_pair("") == ("en", "es"))
check("parse single token -> default ('en','es')", it._parse_pair("en") == ("en", "es"))
check("parse junk -> default ('en','es')", it._parse_pair(",,,") == ("en", "es"))
check("LANG_PAIR default is ('en','es')", it.LANG_PAIR == ("en", "es"))


# --- is_start_intent -------------------------------------------------------
for phrase in (
    "Jarvis, be my interpreter",
    "be my translator please",
    "interpreter mode",
    "start interpreting",
    "start translating now",
    "could you translate for me",
    "interpret for me",
    "modo intérprete",          # accented — normalizer strips the accent
    "sé mi intérprete",
):
    check(f"start intent: {phrase!r}", it.is_start_intent(phrase) is True)

for phrase in (
    "what's the weather",
    "how do you say hello in Spanish",   # one-off translation, NOT mode
    "tell me about the interpreter of maladies",  # noun, but no command verb
    "",
):
    # The maladies line DOES contain 'interpreter' but not a start phrase like
    # "be my interpreter"/"interpreter mode" — assert it does NOT trigger.
    check(f"NOT a start intent: {phrase!r}", it.is_start_intent(phrase) is False)


# --- is_stop_intent --------------------------------------------------------
for phrase in (
    "stop interpreting",
    "Jarvis stop translating",
    "stop the interpreter",
    "exit interpreter",
    "turn off the interpreter",
    "deja de traducir",
    "para de interpretar",
):
    check(f"stop intent: {phrase!r}", it.is_stop_intent(phrase) is True)

for phrase in (
    "stop the music",
    "translate this for me",   # a start-ish phrase is not a stop
    "what time is it",
    "",
):
    check(f"NOT a stop intent: {phrase!r}", it.is_stop_intent(phrase) is False)

# start and stop are disjoint on their own canonical phrases.
check("'be my interpreter' is start, not stop",
      it.is_start_intent("be my interpreter") and not it.is_stop_intent("be my interpreter"))
check("'stop interpreting' is stop, not start",
      it.is_stop_intent("stop interpreting") and not it.is_start_intent("stop interpreting"))


# --- build_translation_prompt ---------------------------------------------
es_prompt = it.build_translation_prompt("es")
check("es prompt names Spanish", "Spanish" in es_prompt)
check("es prompt demands usted (formal register)", "usted" in es_prompt)
check("es prompt says output ONLY the translation",
      "ONLY the translation" in es_prompt)
check("es prompt forbids commentary", "no commentary" in es_prompt)

en_prompt = it.build_translation_prompt("en")
check("en prompt names English", "English" in en_prompt)
check("en prompt does NOT mention usted", "usted" not in en_prompt)
check("en prompt says output ONLY the translation",
      "ONLY the translation" in en_prompt)


# --- confirmation strings exist + are distinct -----------------------------
check("start/stop EN confirmations are non-empty and distinct",
      bool(it.START_CONFIRM_EN) and bool(it.STOP_CONFIRM_EN)
      and it.START_CONFIRM_EN != it.STOP_CONFIRM_EN)
check("start/stop ES confirmations are non-empty and distinct",
      bool(it.START_CONFIRM_ES) and bool(it.STOP_CONFIRM_ES)
      and it.START_CONFIRM_ES != it.STOP_CONFIRM_ES)
check("start confirmation tells the user how to exit",
      "stop interpreting" in it.START_CONFIRM_EN.lower())


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
