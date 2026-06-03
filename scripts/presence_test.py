"""M70 — unit tests for geofenced auto-arm (presence policy + the /presence
route glue).

The must-be-correct cores:

  - `normalize_event`                    — raw geofence string -> arm/disarm/None
  - `PresenceController.handle_event`     — the policy: deferred arm, idempotent
    leave, boundary-flap cancel, greet-only-on-real-transition, and the
    generation guard against the cancel-vs-fire race.
  - `RemoteConsoleServer._presence_token_from` / `_presence_authed` /
    `_handle_presence` — token extraction (header/query precedence),
    constant-time auth, the 200/401 route glue (exercised with a duck-typed
    fake Request so no socket is opened).

All isolated from the real SecurityWatcher / Announcer / network via fakes and
an injected scheduler — deterministic, no waiting on a Timer. Same instrument
discipline as scripts/calendar_monitor_test.py and scripts/homelab_alert_test.py.

    python scripts/presence_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python scripts/presence_test.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.presence import PresenceController, normalize_event  # noqa: E402
from src.remote_console import RemoteConsoleServer  # noqa: E402


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


# --- fakes -----------------------------------------------------------------

class FakeSecurity:
    def __init__(self, armed: bool = False) -> None:
        self._armed = armed
        self.arm_calls = 0
        self.disarm_calls = 0

    def arm(self) -> None:
        self._armed = True
        self.arm_calls += 1

    def disarm(self) -> None:
        self._armed = False
        self.disarm_calls += 1

    def is_armed(self) -> bool:
        return self._armed


class FakeTimer:
    """Records (delay, fn). `.cancel()` marks cancelled; `.fire()` runs fn."""

    def __init__(self, delay: float, fn) -> None:
        self.delay = delay
        self.fn = fn
        self.cancelled = False
        self.fired = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.fired = True
        self.fn()


class FakeScheduler:
    def __init__(self) -> None:
        self.timers: list[FakeTimer] = []

    def __call__(self, delay: float, fn) -> FakeTimer:
        t = FakeTimer(delay, fn)
        self.timers.append(t)
        return t

    @property
    def last(self) -> FakeTimer:
        return self.timers[-1]


def make_controller(armed: bool = False, arm_delay: float = 60.0):
    sec = FakeSecurity(armed)
    sched = FakeScheduler()
    greetings: list[str] = []
    ctrl = PresenceController(
        arm=sec.arm,
        disarm=sec.disarm,
        is_armed=sec.is_armed,
        announce=lambda t: greetings.append(t),
        arm_delay=arm_delay,
        greeting="Welcome home, sir.",
        schedule=sched,
    )
    return ctrl, sec, sched, greetings


# --- normalize_event -------------------------------------------------------

check("normalize leave -> arm", normalize_event("leave") == "arm")
check("normalize arrive -> disarm", normalize_event("arrive") == "disarm")
check("normalize away -> arm", normalize_event("away") == "arm")
check("normalize home -> disarm", normalize_event("home") == "disarm")
check("normalize explicit arm/disarm verbs",
      normalize_event("arm") == "arm" and normalize_event("disarm") == "disarm")
check("normalize is case/space-insensitive", normalize_event("  LEAVE ") == "arm")
check("normalize unknown -> None", normalize_event("garbage") is None)
check("normalize empty -> None", normalize_event("") is None)
check("normalize non-str -> None",
      normalize_event(None) is None and normalize_event(123) is None)


# --- leave: deferred arm ---------------------------------------------------

ctrl, sec, sched, _ = make_controller(armed=False, arm_delay=60.0)
r = ctrl.handle_event("leave")
check("leave schedules a deferred arm (not yet armed)",
      r["action"] == "arm-scheduled" and r["armed"] is False and not sec._armed)
check("leave scheduled one timer at the configured delay",
      len(sched.timers) == 1 and sched.last.delay == 60.0)
sched.last.fire()
check("deferred timer firing actually arms", sec._armed and sec.arm_calls == 1)

# idempotent leave while a deferred arm is pending
ctrl, sec, sched, _ = make_controller(armed=False, arm_delay=60.0)
ctrl.handle_event("leave")
r = ctrl.handle_event("leave")
check("second leave while pending is a no-op (arm-pending)",
      r["action"] == "arm-pending")
check("second leave does NOT schedule a second timer", len(sched.timers) == 1)

# leave while already armed
ctrl, sec, sched, _ = make_controller(armed=True, arm_delay=60.0)
r = ctrl.handle_event("leave")
check("leave while already armed -> already-armed no-op",
      r["action"] == "already-armed" and r["armed"] is True)
check("leave while armed schedules nothing", len(sched.timers) == 0)

# delay == 0 -> arm immediately
ctrl, sec, sched, _ = make_controller(armed=False, arm_delay=0.0)
r = ctrl.handle_event("leave")
check("arm_delay=0 arms immediately",
      r["action"] == "armed" and sec._armed and len(sched.timers) == 0)


# --- arrive: disarm + greet ------------------------------------------------

ctrl, sec, sched, greetings = make_controller(armed=True)
r = ctrl.handle_event("arrive")
check("arrive on an armed house disarms",
      r["action"] == "disarmed" and not sec._armed and sec.disarm_calls == 1)
check("arrive on an armed house greets once",
      r["greeted"] is True and greetings == ["Welcome home, sir."])

ctrl, sec, sched, greetings = make_controller(armed=False)
r = ctrl.handle_event("arrive")
check("arrive on a disarmed house -> no-op, no greeting",
      r["action"] == "no-op" and r["greeted"] is False and greetings == [])


# --- boundary-flap: leave then arrive --------------------------------------

ctrl, sec, sched, greetings = make_controller(armed=False, arm_delay=60.0)
ctrl.handle_event("leave")
r = ctrl.handle_event("arrive")
check("arrive cancels a pending arm (flap-cancelled)",
      r["action"] == "flap-cancelled" and sched.last.cancelled)
check("flap: house was never armed, no greeting",
      not sec._armed and sec.arm_calls == 0 and greetings == [])
# The race: a Timer that fired at the same instant arrive cancelled it must
# still abort thanks to the generation guard.
sched.last.fire()
check("a superseded deferred-arm timer firing after arrive does NOT arm",
      sec.arm_calls == 0 and not sec._armed)


# --- re-arm cycle ----------------------------------------------------------

ctrl, sec, sched, greetings = make_controller(armed=False, arm_delay=30.0)
ctrl.handle_event("leave")
sched.last.fire()                       # armed
ctrl.handle_event("arrive")             # disarmed + greeted
ctrl.handle_event("leave")              # schedule again
sched.last.fire()                       # armed again
check("re-arm after a full leave->arrive->leave cycle works",
      sec._armed and sec.arm_calls == 2 and sec.disarm_calls == 1
      and greetings == ["Welcome home, sir."])


# --- unknown event ---------------------------------------------------------

ctrl, sec, sched, _ = make_controller(armed=False)
r = ctrl.handle_event("teleport")
check("unknown event -> ok:false, no security calls",
      r["ok"] is False and sec.arm_calls == 0 and sec.disarm_calls == 0
      and len(sched.timers) == 0)


# --- defensive: a misbehaving hook never crashes the policy ----------------

def _boom() -> None:
    raise RuntimeError("hook exploded")

ctrl = PresenceController(
    arm=_boom, disarm=lambda: None, is_armed=lambda: False,
    announce=None, arm_delay=0.0, schedule=FakeScheduler(),
)
try:
    r = ctrl.handle_event("leave")
    check("arm hook raising is swallowed (handle_event still returns)",
          isinstance(r, dict) and r["ok"] is True)
except Exception:
    check("arm hook raising is swallowed (handle_event still returns)", False)

ctrl = PresenceController(
    arm=lambda: None, disarm=lambda: None,
    is_armed=_boom,  # raises on every read
    announce=None, arm_delay=60.0, schedule=FakeScheduler(),
)
try:
    r = ctrl.handle_event("arrive")
    check("is_armed raising is swallowed (treated as not-armed)",
          isinstance(r, dict) and r["armed"] is False)
except Exception:
    check("is_armed raising is swallowed (treated as not-armed)", False)

# announce raising must not break the disarm/greet path
ctrl = PresenceController(
    arm=lambda: None, disarm=lambda: None, is_armed=lambda: True,
    announce=_boom, arm_delay=0.0, schedule=FakeScheduler(),
)
try:
    r = ctrl.handle_event("arrive")
    check("announce raising is swallowed (greeted still True, ok)",
          r["ok"] is True and r["greeted"] is True)
except Exception:
    check("announce raising is swallowed (greeted still True, ok)", False)


# --- token extraction (precedence + scheme parsing) ------------------------

tk = RemoteConsoleServer._presence_token_from
check("token from 'Bearer <tok>' header", tk("Bearer abc123", None, None) == "abc123")
check("token Bearer scheme is case-insensitive", tk("bearer abc", None, None) == "abc")
check("token from a bare Authorization value", tk("abc", None, None) == "abc")
check("token from X-Jarvis-Token header", tk(None, "xyz", None) == "xyz")
check("token from ?token= query", tk(None, None, "qtok") == "qtok")
check("no token anywhere -> empty string", tk(None, None, None) == "")
check("Authorization header wins over the others", tk("Bearer A", "B", "C") == "A")
check("X-Jarvis-Token wins over query", tk(None, "B", "C") == "B")
check("bearer token is trimmed", tk("Bearer   spaced  ", None, None) == "spaced")


# --- route glue: _handle_presence via a duck-typed fake Request ------------

class FakeHeaders:
    def __init__(self, d: dict) -> None:
        self._d = d

    def get(self, key, default=None):
        return self._d.get(key, default)


class FakeRequest:
    def __init__(self, path: str, headers: dict | None = None) -> None:
        self.path = path
        self.headers = FakeHeaders(headers or {})


def make_server():
    captured: list[str] = []

    def on_presence(event: str) -> dict:
        captured.append(event)
        return {"ok": True, "armed": False, "action": "arm-scheduled", "echo": event}

    srv = RemoteConsoleServer(
        token="s3cret", host="127.0.0.1", port=0, on_presence=on_presence
    )
    return srv, captured


srv, captured = make_server()
resp = srv._handle_presence(
    FakeRequest("/presence?event=leave", {"Authorization": "Bearer s3cret"})
)
check("valid Bearer header -> 200", resp.status_code == 200)
check("on_presence received the parsed event", captured == ["leave"])
check("route returns the handler's JSON body",
      json.loads(resp.body.decode())["echo"] == "leave")

srv, captured = make_server()
resp = srv._handle_presence(
    FakeRequest("/presence?event=arrive", {"X-Jarvis-Token": "s3cret"})
)
check("valid X-Jarvis-Token header -> 200 + event parsed",
      resp.status_code == 200 and captured == ["arrive"])

srv, captured = make_server()
resp = srv._handle_presence(FakeRequest("/presence?event=leave&token=s3cret"))
check("valid ?token= query -> 200 + event parsed",
      resp.status_code == 200 and captured == ["leave"])

srv, captured = make_server()
resp = srv._handle_presence(
    FakeRequest("/presence?event=leave", {"Authorization": "Bearer wrong"})
)
check("wrong token -> 401", resp.status_code == 401)
check("wrong token -> on_presence NOT called", captured == [])

srv, captured = make_server()
resp = srv._handle_presence(FakeRequest("/presence?event=leave"))
check("missing token -> 401", resp.status_code == 401 and captured == [])

srv, captured = make_server()
resp = srv._handle_presence(
    FakeRequest("/presence", {"Authorization": "Bearer s3cret"})
)
check("auth'd but missing event -> 200, empty event passed to handler",
      resp.status_code == 200 and captured == [""])


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
