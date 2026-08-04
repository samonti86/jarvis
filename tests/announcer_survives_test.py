r"""Regression test: the Announcer thread must survive a hostile UI.

WHY THIS EXISTS (jarvis.log 2026-07-30 19:17:59, found by `self_review`):
A catch-up reminder was announced ~1 s after the Announcer thread started,
before Tk's mainloop was running. `ui.add_system_text` -> `console.add_system_text`
-> `root.after` raised RuntimeError("main thread is not in main loop"). That call
sat OUTSIDE `_announcer_loop`'s try, so the exception escaped the `while` loop and
**killed the Announcer thread one second into the session**. Jarvis then ran for
12+ hours, overnight, unable to speak a single proactive announcement — no
reminders, no weather alerts, no homelab alerts, and no SECURITY alerts. Nothing
in the log said so. It simply went quiet.

Two independent guards now exist and both are asserted here:

  1. `JarvisUI._console_call` swallows console/Tk failures (the fan-out contract
     `_remote_call` always had, finally applied to the other sink).
  2. `_announcer_loop` guards the whole per-item body, so no callee can kill the
     one thread by which Jarvis speaks unprompted.

    python tests/announcer_survives_test.py    # exit 0 = pass
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# --- Guard 1: the UI facade absorbs a console that raises ------------------
print("\n[group] JarvisUI._console_call absorbs a hostile console")

from src.ui import JarvisUI  # noqa: E402


class ExplodingConsole:
    """Every display call raises the exact Tk error seen in production."""

    def __init__(self):
        self.calls = 0

    def __getattr__(self, name):
        def boom(*a, **k):
            self.calls += 1
            raise RuntimeError("main thread is not in main loop")
        return boom


ui = JarvisUI.__new__(JarvisUI)          # bypass __init__ (it builds real Tk)
ui.console = ExplodingConsole()
ui._remote = None
ui._tray = None
ui._console_fanout_broken = False

raised = None
try:
    ui.add_system_text("a catch-up reminder")
    ui.add_jarvis_text("hello")
    ui.add_user_text("hi", "en")
    ui.set_amplitude(0.5)
except Exception as exc:  # noqa: BLE001
    raised = exc

check("display fan-out does not propagate a Tk RuntimeError", raised is None,
      f"raised {raised!r}")
check("the console was actually exercised (test is not vacuous)",
      ui.console.calls >= 4, f"calls={ui.console.calls}")
check("repeat failures are latched to one log line",
      ui._console_fanout_broken is True)


# --- Guard 2: the announcer loop survives an item that blows up ------------
print("\n[group] _announcer_loop survives a failing item")

import src.bootstrap as bootstrap  # noqa: E402

spoken: list[str] = []


class HostileUI:
    """Raises on the FIRST add_system_text, then behaves. Mirrors the real
    bug: a transient startup-window failure, not a permanent one."""

    def __init__(self):
        self.n = 0
        self.shutdown = threading.Event()

    def add_system_text(self, text):
        self.n += 1
        if self.n == 1:
            raise RuntimeError("main thread is not in main loop")

    def set_state(self, *a, **k):
        pass

    def set_amplitude(self, *a, **k):
        pass


hostile = HostileUI()
_real_speak = bootstrap.speak_streaming
bootstrap.speak_streaming = (
    lambda chunks, lang, **kw: spoken.append("".join(list(chunks)))
)
try:
    ann = bootstrap.build_announcer(hostile, threading.Event())
    ann.announce("first — this one detonates the UI call")
    ann.announce("second — must still be spoken")
    deadline = time.monotonic() + 5.0
    while len(spoken) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
finally:
    bootstrap.speak_streaming = _real_speak
    try:
        ann.shutdown()
    except Exception:
        pass

check("the item that raised did NOT kill the announcer", len(spoken) >= 1,
      f"spoken={spoken}")
check("a LATER announcement is still delivered — the loop survived",
      len(spoken) == 2, f"spoken={spoken}")
check("the surviving announcement is the right one",
      len(spoken) == 2 and "second" in spoken[1], f"spoken={spoken}")

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
