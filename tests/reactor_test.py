"""M98 — tests for the pre-rendered arc-reactor renderer.

The look itself is judged by eye (screenshots, see docs/MILESTONES.md). What is
pinned here is everything the look DEPENDS on and that would fail silently:

  - the async cache contract — frames() must never render on the caller's
    thread, because that thread is Tk's and a 300 ms stall is a visible hitch;
  - the chroma-key clip — the whole reason the glow doesn't show as an opaque
    black disc over a light window. A regression here is invisible on the
    author's black desktop and ugly on everyone else's;
  - fail-soft — a render error must disable the reactor, not raise into a
    UI thread and kill the console;
  - that the frames actually differ, i.e. the rotation is real. A cache that
    returned 12 identical images would look "fine" in a screenshot and be a
    dead animation in production.

    python tests/reactor_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import reactor  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


def await_frames(size: int, color: str, timeout: float = 30.0):
    """Poll frames() until the background warm lands."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = reactor.frames(size, color)
        if got is not None:
            return got
        time.sleep(0.05)
    return None


# --- pure colour helpers ---------------------------------------------------
check("hex_to_rgb parses cyan", reactor._hex_to_rgb("#22d3ee") == (0x22, 0xd3, 0xee))
check("hex_to_rgb tolerates no-hash", reactor._hex_to_rgb("22d3ee") == (0x22, 0xd3, 0xee))
check("lerp at t=0 is A", reactor._lerp((0, 0, 0), (10, 20, 30), 0.0) == (0, 0, 0))
check("lerp at t=1 is B", reactor._lerp((0, 0, 0), (10, 20, 30), 1.0) == (10, 20, 30))
check("lerp midpoint", reactor._lerp((0, 0, 0), (10, 20, 30), 0.5) == (5, 10, 15))


# --- the async cache contract ----------------------------------------------
# The load-bearing property: the FIRST call must return None rather than block.
# If someone "helpfully" makes frames() render synchronously, the console stalls
# for ~300 ms on every state change and this is the test that catches it.
t0 = time.perf_counter()
first = reactor.frames(64, "#22d3ee")
elapsed = time.perf_counter() - t0
check("frames() returns None before the warm completes", first is None)
check(f"frames() does not block the caller ({elapsed * 1000:.1f} ms < 50)", elapsed < 0.05)

cycle = await_frames(64, "#22d3ee")
check("frames() returns a cycle once warmed", cycle is not None)
check("cycle length is _FRAMES", cycle is not None and len(cycle) == reactor._FRAMES)
check("frames are the requested size",
      cycle is not None and cycle[0].size == (64, 64))
check("frames are RGBA", cycle is not None and cycle[0].mode == "RGBA")
check("a warmed cycle is returned from cache, not re-rendered",
      reactor.frames(64, "#22d3ee") is cycle)


# --- the rotation is real --------------------------------------------------
# A cache returning 12 identical frames screenshots perfectly and animates not
# at all, so compare bytes rather than trusting the render.
if cycle:
    b0 = cycle[0].tobytes()
    check("frame 1 differs from frame 0 (rotation happens)",
          cycle[1].tobytes() != b0)
    check("frame 6 differs from frame 0", cycle[6].tobytes() != b0)
    distinct = len({f.tobytes() for f in cycle})
    check(f"all {reactor._FRAMES} frames are distinct ({distinct})",
          distinct == reactor._FRAMES)


# --- the chroma-key clip ---------------------------------------------------
# Windows -transparentcolor is BINARY: near-key pixels stay fully opaque. Any
# pixel that would composite to within _KEY_CUT of the key must therefore be
# alpha 0, or the glow's faint tail becomes a black disc over a light window.
if cycle:
    f = cycle[0]
    px = f.load()
    check("image corners are fully transparent", px[0, 0][3] == 0)
    check("the core is fully opaque", px[32, 32][3] == 255)

    key = reactor._KEY_RGB
    offenders = []
    for y in range(0, 64, 2):
        for x in range(0, 64, 2):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            k = a / 255.0
            delta = max(abs(c * k + kc * (1 - k) - kc)
                        for c, kc in zip((r, g, b), key))
            if delta <= reactor._KEY_CUT:
                offenders.append((x, y, delta))
    check(f"no opaque pixel composites to ~the key ({len(offenders)} offenders)",
          not offenders)


# --- fail-soft, and fail ONCE ----------------------------------------------
# A render that raises must disable the reactor, never propagate: the caller is
# a Tk animation tick and an exception there kills the whole UI.
#
# It must also never be retried. frames() calls warm() on every miss, and the
# animation misses 20 times a second — so a permanently-broken render (no
# Pillow, an OOM) would spawn 20 doomed threads per second, forever. The first
# version of this module did exactly that; this is the regression test.
_real_render = reactor._render
_key = (32, "#ff0000", reactor._FRAMES)
try:
    reactor._render = lambda *a, **k: 1 / 0
    reactor.warm(32, "#ff0000")
    deadline = time.time() + 10
    while _key in reactor._warming and time.time() < deadline:
        time.sleep(0.05)
    check("a failing render clears its in-flight marker",
          _key not in reactor._warming)
    check("a failing render is recorded as failed", _key in reactor._failed)

    threads_before = threading.active_count()
    for _ in range(30):
        check_none = reactor.frames(32, "#ff0000")
    check("a failed key keeps returning None", check_none is None)
    check(f"30 further misses spawn NO retry threads "
          f"({threading.active_count() - threads_before})",
          threading.active_count() <= threads_before)
    check("a failing render leaves nothing cached", _key not in reactor._cache)
finally:
    reactor._render = _real_render
    reactor._failed.discard(_key)


# --- warm() is idempotent --------------------------------------------------
# Every frames() miss calls warm(). At 20 fps that is 20 calls a second, so a
# warm that spawned a thread per call would fork hundreds before the first
# finished.
before = threading.active_count()
for _ in range(25):
    reactor.warm(48, "#34e6d0")
spawned = threading.active_count() - before
check(f"25 warm() calls spawn at most 1 thread ({spawned})", spawned <= 1)
check("warm() on an already-cached size is a no-op",
      (reactor.warm(64, "#22d3ee") is None
       and threading.active_count() - before <= 1))

# warm_all fans out over the state palette without duplicating work.
reactor.warm_all(48, ["#34e6d0", "#34e6d0", "#34e6d0"])
check("warm_all dedupes repeated colours",
      len([k for k in reactor._warming if k[1] == "#34e6d0"]) <= 1)


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
