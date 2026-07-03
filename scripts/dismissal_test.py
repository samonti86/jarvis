r"""Regression test for _is_dismissal — the sign-off classifier (M51 + the
2026-06-02 follow-up short-circuit).

WHY THIS EXISTS:
The voice loop now short-circuits a FOLLOW-UP turn that is a pure sign-off
BEFORE running the LLM, because a dismissal turn reaching Claude (with a
freshly-set reminder in context) re-issued set_reminder and created a DUPLICATE
("fired twice"). That short-circuit hinges entirely on _is_dismissal, so its
classification is now load-bearing: a false NEGATIVE re-opens the duplicate
bug; a false POSITIVE would swallow a real question. This locks both edges,
including the exact phrase that triggered the bug.

    python scripts/dismissal_test.py     # exit 0 = pass
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402 — _is_dismissal lives in the loop module

_is_dismissal = main._is_dismissal

PASSED = 0
FAILED = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok: {label}")
    else:
        FAILED += 1
        print(f"  FAIL: {label}  {detail}")


print("\n[group] sign-offs ARE dismissals")
POSITIVES = [
    "that is all",
    "that's all",
    "that will be all",
    "that is it",
    "that is everything",
    "nothing else",
    "no thank you",
    "no thanks",
    "I'm done",
    "we're done",
    "all done",
    "goodbye",
    "thank you Jarvis",
    # courtesy-stripped tails (the M51 follow-on):
    "that is all, thank you",
    "that's all thanks so much",
    # the EXACT phrase that caused the 2026-06-02 duplicate reminder:
    "Thank you that is all",
    "thank you Jarvis, that is all",
]
for p in POSITIVES:
    check(f"dismissal: {p!r}", _is_dismissal(p) is True)


print("\n[group] real utterances are NOT dismissals")
NEGATIVES = [
    "remind me to check my email in two minutes",
    "what is all this about",
    "that's all I need — what about the Jets?",  # tail is a question, not a sign-off
    "how do you say thank you in spanish",
    "how do you say thank you",                  # bare courtesy, even as a tail
    "thank you",                                  # bare politeness ≠ sign-off (M51 choice)
    "thanks",
    "set a timer for ten minutes",
    "",                                           # empty transcript
    # 2026-07-02 QA: "good night" must reach the LLM so it can route to the
    # M63 get_good_night wrap — as a dismissal it short-circuited the wrap
    # on every follow-up/conversation-mode turn.
    "good night",
    "goodnight",
]
for n in NEGATIVES:
    check(f"not a dismissal: {n!r}", _is_dismissal(n) is False)


print("\n" + "=" * 50)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 50)
sys.exit(1 if FAILED else 0)
