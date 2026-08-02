r"""Regression test for the voice-lock / security-challenge ordering bug.

WHY THIS EXISTS (jarvis.log, 2026-08-02 13:57-13:58):
The user was standing in front of the armed camera speaking the passphrase.
Armed-mode CPU load had degraded the mic capture (565 PortAudio input overflows
in that armed window vs 1 in the 112 unarmed minutes before it), which dragged
the M69 speaker-ID score from 0.82 down to 0.64/0.67 — under the 0.75 gate. The
voice-lock therefore dropped BOTH passphrase attempts before
`security.handle_transcript` ever ran, the 15s challenge timed out, and Jarvis
played "Identity not confirmed. Authorities have been notified." at his owner.
Voice disarm was unreachable, so the process had to be killed to stand down.

The fix is `_challenge_overrides_voice_lock`: while a challenge (or the LOCKED
state that follows a timeout) is live, an unrecognized voice's transcript is
still routed to the passphrase comparator. This test pins BOTH edges, because
each direction is a real failure:

  - a FALSE NEGATIVE re-opens the bug above (owner cannot authenticate);
  - a FALSE POSITIVE would widen the security boundary, letting an
    unrecognized voice reach security intents when NO challenge is open —
    i.e. a stranger saying "stand down". That must stay shut.

    python tests/challenge_voice_lock_test.py    # exit 0 = pass
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.listen_loop import _challenge_overrides_voice_lock  # noqa: E402

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


class FakeWatcher:
    """Minimal stand-in for SecurityWatcher's two state predicates."""

    def __init__(self, in_challenge=False, locked=False, raises=False):
        self._in_challenge = in_challenge
        self._locked = locked
        self._raises = raises

    def is_in_challenge(self) -> bool:
        if self._raises:
            raise RuntimeError("watcher exploded")
        return self._in_challenge

    def is_locked(self) -> bool:
        if self._raises:
            raise RuntimeError("watcher exploded")
        return self._locked


print("\n[group] challenge open -> override the voice lock")
check(
    "active CHALLENGE overrides the gate",
    _challenge_overrides_voice_lock(FakeWatcher(in_challenge=True)) is True,
)
check(
    "LOCKED state overrides the gate (post-timeout recovery)",
    _challenge_overrides_voice_lock(FakeWatcher(locked=True)) is True,
)
check(
    "both flags set still overrides",
    _challenge_overrides_voice_lock(
        FakeWatcher(in_challenge=True, locked=True)) is True,
)

print("\n[group] no challenge -> voice lock STAYS in force")
check(
    "idle armed watcher does NOT override (stranger can't 'stand down')",
    _challenge_overrides_voice_lock(FakeWatcher()) is False,
)
check(
    "no watcher configured does NOT override",
    _challenge_overrides_voice_lock(None) is False,
)

print("\n[group] fail-closed on a broken watcher")
check(
    "a raising watcher leaves the voice lock ON (fails closed, never raises)",
    _challenge_overrides_voice_lock(FakeWatcher(raises=True)) is False,
)


class PartialWatcher:
    """A watcher missing is_locked entirely — the loop must not die on it."""

    def is_in_challenge(self) -> bool:
        return False


check(
    "a watcher missing is_locked fails closed rather than raising",
    _challenge_overrides_voice_lock(PartialWatcher()) is False,
)

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
