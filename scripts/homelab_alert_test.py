"""M56 — unit test for the homelab monitor's alert state machine.

The interesting, must-be-correct core of M56 is `_CheckTracker`: the per-check
edge-triggered alert engine with flap damping. The parts that are hard to
live-test (Plex actually going down) are exactly the parts this exercises
deterministically — feed it synthetic ok/fail sequences, assert that an alert
fires once per real transition and never otherwise.

Same instrument discipline as scripts/leak_repro.py and
scripts/barge_stutter_soak.py: a standalone, runnable, asserting harness, so a
regression in the alerting logic is caught structurally rather than by ear in
production.

    python scripts/homelab_alert_test.py     # exit 0 = all pass, 1 = a failure
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/homelab_alert_test.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.homelab_monitor import _CheckTracker  # noqa: E402


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


def feed(tracker: _CheckTracker, oks: list[bool]) -> list[str | None]:
    """Observe a sequence, return the edge each observation produced."""
    return [tracker.observe(ok) for ok in oks]


# --- Test 1: a clean OK stream never alerts --------------------------------
t = _CheckTracker(fail_threshold=3)
edges = feed(t, [True] * 10)
check("clean OK stream -> no edges at all", all(e is None for e in edges))
check("clean OK stream -> not down", not t.is_down)


# --- Test 2: flap damping — a short failure blip is debounced --------------
t = _CheckTracker(fail_threshold=3)
edges = feed(t, [False, False, True])  # 2 fails (below threshold), then OK
check("2-fail blip (threshold 3) -> no alert (damped)", edges == [None, None, None])
check("2-fail blip -> recovered to not-down", not t.is_down)


# --- Test 3: the OK->DOWN edge fires once, exactly on the threshold ---------
t = _CheckTracker(fail_threshold=3)
edges = feed(t, [False, False, False])
check("3rd consecutive fail (threshold 3) -> 'down' on the 3rd only",
      edges == [None, None, "down"])
check("after 'down' edge -> is_down True", t.is_down)


# --- Test 4: while DOWN, further failures are silent (edge-triggered) ------
t = _CheckTracker(fail_threshold=3)
feed(t, [False, False, False])               # -> down
edges = feed(t, [False] * 5)                 # stay-down polls
check("stay-down polls -> no repeat alerts", all(e is None for e in edges))


# --- Test 5: the DOWN->OK recovery edge fires once -------------------------
t = _CheckTracker(fail_threshold=3)
feed(t, [False, False, False])               # -> down
edges = feed(t, [True])
check("first OK after down -> exactly one 'recovered'", edges == ["recovered"])
check("after recovery -> is_down False", not t.is_down)
edges = feed(t, [True, True])
check("further OK polls after recovery -> silent", all(e is None for e in edges))


# --- Test 6: no phantom 'recovered' without a preceding 'down' ------------
t = _CheckTracker(fail_threshold=3)
edges = feed(t, [True, True, True])
check("OK stream with no prior down -> never emits 'recovered'",
      "recovered" not in edges)


# --- Test 7: a check can go down again after recovering -------------------
t = _CheckTracker(fail_threshold=3)
feed(t, [False, False, False])               # down
feed(t, [True])                              # recovered
edges = feed(t, [False, False, False])       # down again
check("down -> recover -> down again -> second 'down' fires",
      edges == [None, None, "down"])


# --- Test 8: the failure counter resets on any OK (consecutive, not total) -
t = _CheckTracker(fail_threshold=3)
edges = feed(t, [False, False, True, False, False, False])
check("fail,fail,OK,fail,fail,fail -> 'down' only on the final 3rd-in-a-row",
      edges == [None, None, None, None, None, "down"])


# --- Test 9: reset() returns the tracker to a clean OK state --------------
t = _CheckTracker(fail_threshold=3)
feed(t, [False, False, False])               # down
t.reset()
check("reset() after down -> is_down False", not t.is_down)
edges = feed(t, [False, False])              # below threshold from a clean slate
check("reset() clears the failure counter -> 2 fails no longer alert",
      edges == [None, None])


# --- Test 10: threshold 1 = alert on the first failure (no damping) -------
t = _CheckTracker(fail_threshold=1)
edges = feed(t, [False])
check("threshold 1 -> first failure alerts immediately", edges == ["down"])


# --- summary ---------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
