"""M58 — unit test for the per-class alert state machine (_ClassTracker).

The same discipline as M56's homelab_alert_test.py: isolate the alert-hygiene
logic from all I/O so the must-be-correct piece is asserted deterministically.
_ClassTracker is the sustain-then-fire + cooldown gate; this exercises every
documented edge (single-hit debounced, miss resets the streak, cooldown blocks
re-fire, cooldown expires cleanly, sustain clamps, etc.).

    python scripts/acoustic_alert_test.py     # exit 0 = all pass, 1 = a failure
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sound_detector import _ClassTracker  # noqa: E402


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


# --- Test 1: a stream of misses never fires --------------------------------
t = _ClassTracker(sustain=2, cooldown_seconds=10.0)
results = [t.observe(False, float(i)) for i in range(5)]
check("all-miss stream -> no fires", all(r is False for r in results))


# --- Test 2: one hit with sustain=2 -> debounced (no fire) ----------------
t = _ClassTracker(sustain=2, cooldown_seconds=10.0)
check("1 hit (sustain=2) -> no fire (debounced)",
      t.observe(True, 0.0) is False)


# --- Test 3: two consecutive hits with sustain=2 -> fire on the 2nd -------
t = _ClassTracker(sustain=2, cooldown_seconds=10.0)
f1 = t.observe(True, 0.0)
f2 = t.observe(True, 2.0)
check("2 hits (sustain=2) -> fire on the 2nd only",
      f1 is False and f2 is True)


# --- Test 4: cooldown blocks an immediate re-fire --------------------------
t = _ClassTracker(sustain=1, cooldown_seconds=10.0)
f1 = t.observe(True, 0.0)
f2 = t.observe(True, 1.0)        # well inside cooldown
check("cooldown blocks an immediate re-fire",
      f1 is True and f2 is False)


# --- Test 5: a re-fire is allowed once cooldown elapses --------------------
t = _ClassTracker(sustain=1, cooldown_seconds=10.0)
t.observe(True, 0.0)
check("re-fires after cooldown elapses",
      t.observe(True, 11.0) is True)


# --- Test 6: a single miss resets the consecutive-hit counter -------------
t = _ClassTracker(sustain=3, cooldown_seconds=10.0)
results = [
    t.observe(True, 0.0),    # 1/3
    t.observe(True, 2.0),    # 2/3
    t.observe(False, 4.0),   # streak broken
    t.observe(True, 6.0),    # 1/3 again
    t.observe(True, 8.0),    # 2/3
    t.observe(True, 10.0),   # 3/3 -> fire
]
check("a miss resets the streak; fires only on 3-in-a-row",
      results == [False, False, False, False, False, True])


# --- Test 7: continuous hits while in cooldown stay silent ----------------
t = _ClassTracker(sustain=1, cooldown_seconds=10.0)
t.observe(True, 0.0)             # fires; cooldown_until = 10
results = [t.observe(True, x) for x in (1.0, 2.0, 5.0, 9.0)]
check("continuous hits during cooldown -> all silent",
      all(r is False for r in results))


# --- Test 8: reset() clears both the streak and the cooldown --------------
t = _ClassTracker(sustain=2, cooldown_seconds=100.0)
t.observe(True, 0.0)             # 1/2
t.reset()
check("reset() clears the streak -> first hit after reset doesn't fire",
      t.observe(True, 5.0) is False)
check("reset() clears the cooldown -> second hit fires immediately",
      t.observe(True, 7.0) is True)


# --- Test 9: sustain=1 -> fires on the very first hit ---------------------
t = _ClassTracker(sustain=1, cooldown_seconds=10.0)
check("sustain=1 -> first hit fires immediately",
      t.observe(True, 0.0) is True)


# --- Test 10: sustain=0 clamps to a minimum of 1 --------------------------
t = _ClassTracker(sustain=0, cooldown_seconds=10.0)
check("sustain=0 clamps to >=1",
      t.observe(True, 0.0) is True)


# --- Test 11: the streak survives an in-cooldown silent stretch -----------
# (No miss between fires, so the consecutive counter keeps growing; once the
# cooldown elapses, the next hit fires without needing a fresh sustain.)
t = _ClassTracker(sustain=2, cooldown_seconds=10.0)
t.observe(True, 0.0)             # 1/2
t.observe(True, 2.0)             # 2/2 -> fires, cooldown_until = 12
t.observe(True, 5.0)             # in cooldown, silent (streak grows)
t.observe(True, 8.0)             # still in cooldown, silent
check("streak survives a silent in-cooldown stretch (no misses); fires once cooldown clears",
      t.observe(True, 13.0) is True)


# --- summary ---------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
