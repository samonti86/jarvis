"""M70 — geofenced auto-arm: phone presence drives security state.

The security flow (M34) has exactly one friction point: remembering to arm it.
The phone is already a Jarvis client (M48). This closes the loop — an iOS
Shortcuts personal automation fires an HTTP request to Jarvis's remote console
on a geofence boundary:

    "When I leave home"   → POST /presence?event=leave   (arm)
    "When I arrive home"  → POST /presence?event=arrive  (disarm + welcome home)

**Geofencing is OFFLOADED to iOS, deliberately.** The OS does native,
battery-efficient, hysteresis-buffered geofence detection far better than a web
app could: PWA background geolocation is crippled on iOS (won't run while the
page is asleep), and WiFi/Tailscale presence can't tell home-from-away (the
Tailscale tunnel stays up on cellular). So Jarvis exposes a thin token-gated
endpoint and the OS supplies the "am I home?" signal. We only apply *policy*.

Policy lives HERE, isolated from the transport (remote_console) and the I/O
(SecurityWatcher / Announcer), so it's unit-testable in isolation — the same
discipline as `homelab_monitor._CheckTracker` and `calendar_monitor`'s
`_should_announce`. The three rules:

  - **Idempotent.** A duplicate "leave" while already armed is a no-op; iOS
    geofences can re-fire the same boundary crossing more than once.
  - **Boundary-flap damping.** Arming is DEFERRED by a short delay (default
    60s), cancelled if an "arrive" lands first. This absorbs the classic
    geofence failure: skirting the boundary fires leave→arrive within seconds,
    which would otherwise arm-then-disarm and greet you on your own couch. The
    geofence is already ~100m out, so a minute's delay after you've physically
    left costs nothing real.
  - **Greet only on a real transition.** "Welcome home" fires only when an
    arrive actually disarmed an armed system (was_armed → now disarmed), never
    on a spurious arrive to an already-disarmed house.

Defensive contract (same as every optional subsystem): handle_event NEVER
raises — every callback into security/announce is guarded; a misbehaving hook
degrades to a logged no-op, never a crash on the server's event loop.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable, Optional


def _log(msg: str) -> None:
    print(f"[presence] {msg}", file=sys.stderr)


# Event vocabulary — iOS Shortcuts personal automations are phrased
# "When I arrive" / "When I leave". Accept those, a few natural synonyms, and
# the explicit arm/disarm verbs (handy for a curl smoke test).
_LEAVE_WORDS = {"leave", "left", "leaving", "away", "depart", "departed",
                "exit", "exited", "arm"}
_ARRIVE_WORDS = {"arrive", "arrived", "arriving", "home", "return", "returned",
                 "enter", "entered", "disarm"}


def normalize_event(event: object) -> Optional[str]:
    """Map a raw presence-event string to 'arm' | 'disarm' | None.
    Case- and whitespace-insensitive. Unknown ⇒ None (caller rejects)."""
    if not isinstance(event, str):
        return None
    e = event.strip().lower()
    if e in _LEAVE_WORDS:
        return "arm"
    if e in _ARRIVE_WORDS:
        return "disarm"
    return None


def _default_schedule(delay: float, fn: Callable[[], None]) -> threading.Timer:
    """Real deferred-call scheduler: a daemon Timer with a `.cancel()` handle.
    Injected with a fake in tests so timing is deterministic."""
    timer = threading.Timer(delay, fn)
    timer.daemon = True
    timer.start()
    return timer


class PresenceController:
    """Presence policy. `handle_event(raw_event)` returns a result dict and
    NEVER raises. All security/announce side effects go through guarded
    callbacks injected at construction (so tests use fakes, and main.py binds
    the live SecurityWatcher + Announcer)."""

    def __init__(
        self,
        *,
        arm: Callable[[], None],
        disarm: Callable[[], None],
        is_armed: Callable[[], bool],
        announce: Optional[Callable[[str], None]] = None,
        arm_delay: float = 60.0,
        greeting: str = "Welcome home, sir.",
        schedule: Optional[Callable[[float, Callable[[], None]], object]] = None,
    ) -> None:
        self._arm = arm
        self._disarm = disarm
        self._is_armed = is_armed
        self._announce = announce
        self._arm_delay = max(0.0, float(arm_delay))
        self._greeting = greeting
        self._schedule = schedule or _default_schedule
        # `_pending` is the cancel handle for a deferred arm (or None). `_gen`
        # is a monotonic generation token: every schedule/cancel bumps it, and
        # a fired timer validates its captured token before arming — this
        # closes the race where a Timer fires at the same instant an "arrive"
        # cancels it (cancel() can't stop a Timer whose callback already
        # started, so the generation guard is the real defense).
        self._pending: Optional[object] = None
        self._gen = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def handle_event(self, event: object) -> dict:
        action = normalize_event(event)
        if action == "arm":
            return self._on_leave()
        if action == "disarm":
            return self._on_arrive()
        _log(f"ignored unknown event {event!r}")
        return {"ok": False, "error": "unknown event",
                "armed": self._safe_is_armed()}

    # ------------------------------------------------------------------
    # leave → arm (deferred + idempotent)
    # ------------------------------------------------------------------

    def _on_leave(self) -> dict:
        with self._lock:
            if self._safe_is_armed():
                _log("leave: already armed — no-op")
                return {"ok": True, "armed": True, "action": "already-armed"}
            if self._pending is not None:
                _log("leave: arm already pending — no-op")
                return {"ok": True, "armed": False, "action": "arm-pending"}
            if self._arm_delay <= 0:
                self._do_arm()
                return {"ok": True, "armed": self._safe_is_armed(),
                        "action": "armed"}
            self._gen += 1
            my_gen = self._gen
            _log(f"leave: arming in {self._arm_delay:.0f}s (flap-damping window)")
            self._pending = self._schedule(
                self._arm_delay, lambda: self._fire_deferred_arm(my_gen)
            )
            return {"ok": True, "armed": False, "action": "arm-scheduled",
                    "delay_seconds": self._arm_delay}

    def _fire_deferred_arm(self, gen: int) -> None:
        with self._lock:
            if gen != self._gen:
                # Superseded by an arrive (cancel) or a newer leave — abort.
                return
            self._pending = None
            if not self._safe_is_armed():
                self._do_arm()

    def _do_arm(self) -> None:
        self._safe(self._arm, "arm")
        _log("armed (presence: away)")

    # ------------------------------------------------------------------
    # arrive → disarm + greet on a real transition
    # ------------------------------------------------------------------

    def _on_arrive(self) -> dict:
        with self._lock:
            cancelled = self._pending is not None
            if cancelled:
                self._safe_cancel(self._pending)
                self._pending = None
            # Bump the generation unconditionally so a just-fired-but-blocked
            # deferred arm aborts when it finally acquires the lock.
            self._gen += 1
            was_armed = self._safe_is_armed()
            self._safe(self._disarm, "disarm")
            greeted = False
            if was_armed and self._announce is not None:
                self._safe(lambda: self._announce(self._greeting), "announce")
                greeted = True
            if was_armed:
                action = "disarmed"
            elif cancelled:
                action = "flap-cancelled"
            else:
                action = "no-op"
            _log(f"arrive: {action}" + (" + welcomed home" if greeted else ""))
            return {"ok": True, "armed": self._safe_is_armed(),
                    "action": action, "greeted": greeted}

    # ------------------------------------------------------------------
    # Defensive helpers
    # ------------------------------------------------------------------

    def _safe_is_armed(self) -> bool:
        try:
            return bool(self._is_armed())
        except Exception as exc:  # noqa: BLE001 — a bad hook must not crash policy
            _log(f"is_armed raised: {exc}")
            return False

    @staticmethod
    def _safe(fn: Callable[[], None], label: str) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _log(f"{label} raised: {exc}")

    @staticmethod
    def _safe_cancel(handle: object) -> None:
        try:
            cancel = getattr(handle, "cancel", None)
            if callable(cancel):
                cancel()
        except Exception as exc:  # noqa: BLE001
            _log(f"cancel raised: {exc}")
