r"""Regression test for the cooldown/presence deadlock (2026-08-02).

WHY THIS EXISTS (jarvis.log 2026-08-02 14:59-15:01):
The user cleared a challenge by passphrase at 14:59:17, which set a 60 s
cooldown. He disarmed, re-armed at 14:59:43, and walked back into frame. At
15:00:16 the watcher saw him and refused to challenge — "in cooldown (1.1s
left)" — because the NEW arm had inherited the OLD arm's cooldown. Then nothing
happened for the rest of the armed window: he reported Jarvis "froze" and never
saw him.

Two independent defects produced that, and each is pinned below:

  1. `activate()` did not reset `_cooldown_until`, so a fresh arming session
     inherited stale state from one that had already ended.

  2. `_enter_challenge` returns early during cooldown, but the watcher has
     ALREADY latched `_person_present = True`. A person standing still keeps
     refreshing `_person_last_seen`, so the 30 s presence-clear never elapses,
     no second EMPTY->PERSON transition ever fires, and the challenge is
     suppressed for the WHOLE remaining armed window.

Defect 2 is a security hole, not a UX wrinkle: an intruder arriving during a
cooldown is never challenged as long as they stay in frame. But the re-try must
NOT fire for someone who already authenticated — otherwise you get re-challenged
for sitting at your own desk 60 s after identifying yourself. Both directions are
asserted.

    python tests/challenge_cooldown_test.py     # exit 0 = pass
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import security as sec  # noqa: E402

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


def make_watcher():
    """A SecurityWatcher with every side effect stubbed out.

    __init__ only stores config, so no camera/model/network is touched here.
    """
    w = sec.SecurityWatcher(
        camera_index=0,
        passphrase="strongest avenger",
        announce=lambda *a, **k: None,
        evidence_dir=Path("."),
    )
    w._announce = lambda *a, **k: None
    return w


# --- Defect 1: a fresh arm must not inherit the previous arm's cooldown ------
print("\n[group] cooldown does not survive a re-arm")

w = make_watcher()
w._safe_call = lambda fn, *a, **k: None      # skip announce/UI callbacks
w._cooldown_until = time.monotonic() + 60.0  # as if we just authenticated
w._presence_deferred_by_cooldown = True
# activate() spawns the watcher thread; stub it so nothing real runs.
w._watch_loop = lambda: None
w.activate()

check(
    "activate() clears the stale cooldown",
    w._cooldown_until == 0.0,
    f"_cooldown_until={w._cooldown_until}",
)
check(
    "activate() clears the deferred-presence flag",
    w._presence_deferred_by_cooldown is False,
)
w._armed.clear()


# --- Defect 2: a cooldown-suppressed presence is re-tried, once -------------
print("\n[group] a person seen DURING cooldown is not lost")

w = make_watcher()
now = time.monotonic()
w._armed.set()
w._cooldown_until = now + 30.0

w._enter_challenge(b"", source="local", now=now)
check(
    "a cooldown-blocked motion sets the deferred flag",
    w._presence_deferred_by_cooldown is True,
)
check(
    "...and does NOT open a challenge",
    w._challenge_active is False,
)

# Once the cooldown lapses, entering again must actually challenge.
entered = {"n": 0}
w._start_challenge_timer_called = False
w._safe_call = lambda fn, *a, **k: entered.__setitem__("n", entered["n"] + 1)
w._enter_challenge(b"", source="local", now=w._cooldown_until + 0.1)
check(
    "after the cooldown lapses, the challenge DOES open",
    w._challenge_active is True,
)
check(
    "opening a challenge clears the deferred flag",
    w._presence_deferred_by_cooldown is False,
)


# --- The flag must not be set when the challenge was declined for other -----
# --- reasons, so the re-try can't fire for an authenticated person ----------
print("\n[group] the re-try is scoped to cooldown-only suppression")

w = make_watcher()
now = time.monotonic()
w._armed.set()
w._cooldown_until = 0.0            # no cooldown
w._safe_call = lambda fn, *a, **k: None
w._enter_challenge(b"", source="local", now=now)
check(
    "a normal challenge entry leaves the deferred flag False",
    w._presence_deferred_by_cooldown is False,
)

# A second motion while the challenge is already open must not set it either.
w._enter_challenge(b"", source="local", now=now + 1.0)
check(
    "motion during an ACTIVE challenge leaves the deferred flag False",
    w._presence_deferred_by_cooldown is False,
)

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
