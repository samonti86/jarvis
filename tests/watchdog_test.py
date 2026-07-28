"""M65 — unit tests for the crash-resilience watchdog.

The watchdog has two interesting cores:

  - `_is_crash_looping(history)` — pure decision: are the last N restarts
    all within the cap window?
  - `watchdog_loop()` — spawn → wait → decide. Tested end-to-end via a
    monkeypatched `_spawn_child` that returns a synthetic child whose
    `wait()` immediately yields a controlled exit code. No real
    subprocesses, no real main.py, no real timing.

The integration test exercises the FOUR exit-code dispatches:
  - 0  → loop returns 0 (clean quit)
  - 42 → respawn (the "please restart me" sentinel)
  - 1  (and any non-0/42) → counts against the crash budget; loop
        respawns OR gives up when the budget is exhausted

    python tests/watchdog_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

# jarvis_watchdog.pyw isn't on the import path by default — it's a .pyw
# file in the project root. Load it explicitly via importlib.
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

_spec = importlib.util.spec_from_file_location(
    "jarvis_watchdog", _HERE / "jarvis_watchdog.pyw"
)
assert _spec is not None and _spec.loader is not None
jarvis_watchdog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jarvis_watchdog)


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


# --- _is_crash_looping ----------------------------------------------------

print("\n_is_crash_looping:")

# Empty history → never looping.
check("empty history -> not looping",
      jarvis_watchdog._is_crash_looping([]) is False)

# Single restart, however recent → not looping (need _RESTART_HISTORY).
check("single restart -> not looping",
      jarvis_watchdog._is_crash_looping([time.time()]) is False)

# N-1 restarts → not looping.
n = jarvis_watchdog._RESTART_HISTORY
recent = [time.time() - 5.0] * (n - 1)
check(f"{n - 1} restarts -> not looping",
      jarvis_watchdog._is_crash_looping(recent) is False)

# N restarts all within the window → looping.
recent = [time.time() - 5.0] * n
check(f"{n} restarts within window -> looping",
      jarvis_watchdog._is_crash_looping(recent) is True)

# N restarts spread BEYOND the window → not looping. Oldest is far past
# the window cutoff; only the recent ones matter.
window = jarvis_watchdog._RESTART_WINDOW
spread = ([time.time() - window - 60.0] * (n - 1)) + [time.time()]
# Helper looks at history[-_RESTART_HISTORY], so we need >= N entries
# total. The "oldest of the last N" calculation should see the old ones.
recent_old = [time.time() - window - 10.0] * n
check(f"{n} restarts older than window -> not looping",
      jarvis_watchdog._is_crash_looping(recent_old) is False)

# Mix: 2 very old + N recent → looking only at last N, all recent
# → looping (the very-old ones don't help us).
mixed = [time.time() - 9999.0] * 2 + [time.time() - 5.0] * n
check("old + N recent (looks at last N) -> looping",
      jarvis_watchdog._is_crash_looping(mixed) is True)


# --- watchdog_loop with synthetic children -------------------------------

print("\nwatchdog_loop (synthetic children):")


class _FakeChild:
    def __init__(self, rc: int) -> None:
        self._rc = rc

    def wait(self) -> int:
        return self._rc

    def terminate(self) -> None:
        pass


def _stub_spawner(exit_codes: list[int]):
    """Return a _spawn_child stub that yields fake children with the given
    exit codes in order. Asserts on too-many spawns (the loop is buggy)."""
    iterator = iter(exit_codes)
    spawn_count = [0]

    def _spawn() -> _FakeChild:
        spawn_count[0] += 1
        try:
            rc = next(iterator)
        except StopIteration as exc:
            raise AssertionError(
                f"watchdog spawned more children than expected "
                f"({spawn_count[0]} > {len(exit_codes)})"
            ) from exc
        return _FakeChild(rc)

    return _spawn, spawn_count


_orig_spawn = jarvis_watchdog._spawn_child
_orig_sleep = time.sleep
try:
    # Make backoff instant for tests.
    time.sleep = lambda _: None  # type: ignore[assignment]

    # Scenario A: child exits 0 immediately → watchdog returns 0, spawns 1.
    spawn, count = _stub_spawner([0])
    jarvis_watchdog._spawn_child = spawn
    rc = jarvis_watchdog.watchdog_loop()
    check("rc=0 first try -> loop returns 0", rc == 0)
    check("rc=0 -> spawned exactly once", count[0] == 1)

    # Scenario B: child exits 42 (restart) then 0 → loop returns 0, 2 spawns.
    spawn, count = _stub_spawner([42, 0])
    jarvis_watchdog._spawn_child = spawn
    rc = jarvis_watchdog.watchdog_loop()
    check("rc=42 then rc=0 -> loop returns 0 (clean handoff)", rc == 0)
    check("rc=42 then rc=0 -> spawned exactly twice", count[0] == 2)

    # Scenario C: child crashes once, then quits cleanly → loop returns 0,
    # 2 spawns. The crash is below the cap so we respawn.
    spawn, count = _stub_spawner([1, 0])
    jarvis_watchdog._spawn_child = spawn
    rc = jarvis_watchdog.watchdog_loop()
    check("crash + clean -> loop returns 0", rc == 0)
    check("crash + clean -> spawned twice", count[0] == 2)

    # Scenario D: child crashes _RESTART_HISTORY times in a row → cap
    # triggers, loop returns 1. Spawn count = _RESTART_HISTORY exactly
    # (the cap fires AFTER the Nth wait()).
    n = jarvis_watchdog._RESTART_HISTORY
    crashes = [1] * n
    spawn, count = _stub_spawner(crashes)
    jarvis_watchdog._spawn_child = spawn
    rc = jarvis_watchdog.watchdog_loop()
    check(f"{n} crashes in a row -> loop returns 1 (give up)", rc == 1)
    check(f"{n} crashes -> spawned exactly {n} times before cap",
          count[0] == n)

    # Scenario E: user-driven restarts (42) are NOT counted against the
    # crash budget — many 42's in a row are fine. 100 user-restarts then
    # a clean exit → returns 0, no give-up.
    seq = [42] * 100 + [0]
    spawn, count = _stub_spawner(seq)
    jarvis_watchdog._spawn_child = spawn
    rc = jarvis_watchdog.watchdog_loop()
    check("100 user-restarts + clean -> returns 0",
          rc == 0)
    check("100 user-restarts not counted as crashes -> 101 spawns",
          count[0] == 101)

    # Scenario F: alternating crash + restart — crashes accumulate but
    # the restarts in between don't reset history. After N crashes total
    # we cap, even with restarts mixed in.
    n = jarvis_watchdog._RESTART_HISTORY
    # crash, restart, crash, restart, ... ending on the Nth crash
    seq = []
    for _ in range(n):
        seq.extend([1, 42])
    seq.pop()  # last entry was a 42; remove so we end on a crash that caps
    spawn, count = _stub_spawner(seq)
    jarvis_watchdog._spawn_child = spawn
    rc = jarvis_watchdog.watchdog_loop()
    check("crash + restart alternation still hits cap after N crashes",
          rc == 1)
finally:
    jarvis_watchdog._spawn_child = _orig_spawn
    time.sleep = _orig_sleep


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
