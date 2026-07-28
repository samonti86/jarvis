"""M88 — tests for conversation mode (the pure core in src/conversation_mode.py).

Conversation mode is full-duplex Phase 1: "Jarvis, let's talk" drops the
wake-word-per-turn rhythm and stays in continuous hands-free Q&A until an exit.
The model-free, I/O-free cores tested here:

  - is_start_intent: EXACT-match activation (so "let's talk about X" — a real
    question — does NOT hijack into mode entry), with affix stripping.
  - is_stop_intent: SUBSTRING-match explicit exits.
  - start_confirmation / stop_confirmation: language-keyed spoken lines.

Hermetic: no model, no audio, no network, no listen loop.

    python tests/conversation_mode_test.py   # exit 0 = all pass
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import conversation_mode as cm  # noqa: E402

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


# --- is_start_intent: positives (bare requests, with affixes) --------------
for phrase in (
    "let's talk",
    "lets talk",
    "Hey Jarvis, let's talk",
    "okay Jarvis let's chat",
    "let's have a conversation",
    "let's chat please",
    "conversation mode",
    "let's keep talking",
    "talk to me",
    "vamos a hablar",        # Spanish
    "conversemos",
    "modo conversación",     # accented — handled? cm._normalize keeps accents
):
    check(f"start intent: {phrase!r}", cm.is_start_intent(phrase) is True)

# --- is_start_intent: negatives (a topic = a real question, NOT mode entry) -
for phrase in (
    "let's talk about the Knicks",
    "let's talk about my schedule tomorrow",
    "what's the weather",
    "talk to me about the playoffs",
    "can we talk about something",
    "",
):
    check(f"NOT a start intent: {phrase!r}", cm.is_start_intent(phrase) is False)


# --- is_stop_intent --------------------------------------------------------
for phrase in (
    "exit conversation",
    "exit conversation mode",
    "okay let's exit conversation mode",
    "stop conversation",
    "end conversation",
    "conversation mode off",
    "we're done talking",
    "that's enough talking",
):
    check(f"stop intent: {phrase!r}", cm.is_stop_intent(phrase) is True)

for phrase in (
    "stop the music",
    "what time is it",
    "let's talk",            # a start phrase is not a stop
    "",
):
    check(f"NOT a stop intent: {phrase!r}", cm.is_stop_intent(phrase) is False)

# start and stop are disjoint on their canonical phrases.
check("'let's talk' is start, not stop",
      cm.is_start_intent("let's talk") and not cm.is_stop_intent("let's talk"))
check("'exit conversation' is stop, not start",
      cm.is_stop_intent("exit conversation") and not cm.is_start_intent("exit conversation"))


# --- affix stripping -------------------------------------------------------
check("leading 'hey jarvis' stripped", cm._strip_affixes("hey jarvis lets talk") == "lets talk")
check("trailing 'please' stripped", cm._strip_affixes("lets chat please") == "lets chat")
check("apostrophes deleted in normalize ('let's' -> 'lets')",
      cm._normalize("let's") == "lets")
check("bare wake word strips to empty", cm._strip_affixes("jarvis") == "")


# --- confirmations ---------------------------------------------------------
check("EN start confirmation is non-empty", bool(cm.start_confirmation("en")))
check("EN stop confirmation is non-empty", bool(cm.stop_confirmation("en")))
check("ES start confirmation differs from EN",
      cm.start_confirmation("es") != cm.start_confirmation("en"))
check("unknown lang falls back to EN start",
      cm.start_confirmation("fr") == cm.start_confirmation("en"))
check("start confirmation tells the user how to exit",
      "that's all" in cm.start_confirmation("en").lower())


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
