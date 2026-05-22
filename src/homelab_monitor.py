"""Proactive homelab monitoring (M56) — Jarvis's first unprompted heartbeat.

Everything Jarvis did before this was *reactive*: you speak, he answers. Even
the M55 briefing only happens when you ask. This module gives him a heartbeat —
a background thread that quietly watches the homelab and speaks up ON ITS OWN
when something changes: Plex stops responding, the Plex laptop goes dark, a
disk crosses a threshold. And when it recovers, it says so. That reactive →
proactive jump is the milestone.

It is almost entirely a recombination of infrastructure that already exists:
  - thread model      = reminders.run_scheduler — one defensive daemon poll
                        loop, a stop Event, every poll wrapped so a single bad
                        check can never kill the thread.
  - lifecycle/opt-in  = SecurityWatcher — activate() / deactivate() /
                        is_active(), a tray toggle, an .env flag, default OFF.
  - remote reach      = the existing PlexLaptopClient (its run() is already
                        lock-serialised, so this thread shares the one SSH
                        connection the tool path uses — no new auth surface).
  - voice out         = main.py's _announce (the WASAPI-safe Announcer path).
  - phone push        = notifications.send_discord_message.

The interesting part — the alert state machine
-----------------------------------------------
A monitoring tool lives or dies on *alert hygiene*. The naive version polls
every minute and, while Plex is down, says "Plex is down" every single minute —
that is alert fatigue, the thing that makes people mute their monitoring. The
fix is three classic ideas, each ~a handful of lines:

  1. Flap damping / hysteresis (`_CheckTracker`). A single failed poll is
     noise — one dropped packet, a GC pause. A check must fail
     `_FAIL_THRESHOLD` polls IN A ROW before it transitions OK → DOWN. This is
     Nagios's soft-vs-hard state (`max_check_attempts`); it is a Schmitt
     trigger — you do not want a sensor chattering on the boundary.

  2. Edge-triggered, not level-triggered. The alert fires ONCE, on the
     transition. Stay-down polls are silent. Recovery (DOWN → OK) is also an
     edge and fires its own one "it's back" notice — closing the loop matters.

  3. Dependency suppression. `plex` and `disk` declare `depends_on="host"`. If
     the laptop itself is unreachable, those two checks are skipped rather than
     each firing their own alarm — you get ONE "laptop is down", not three.
     This is Nagios parent/child host dependencies: do not page for fifty
     services when the router they sit behind is the thing that died.

Defensive contract, same as every tool module: nothing here raises into the
poll thread or the listen loop. A dead host, a broken probe, a timeout — all
become log lines and a fail-soft result, never an exception.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
from dataclasses import dataclass
from typing import Callable, NamedTuple

import httpx


# --- Tunables --------------------------------------------------------------
# These are read once at import (same pattern as security.py's _YOLO_THREADS /
# memory-watchdog knobs — a subsystem owns its own tunables; the core Config
# only carries the on/off flag main.py needs at startup).

def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, value)


# Poll cadence. 60 s is plenty — a homelab outage does not need 10 s
# granularity, and a TCP-connect + an HTTP GET + one SSH call per minute is
# negligible load. With the default 3-poll threshold an outage is announced
# ~3 minutes in, which matches the "stopped responding a few minutes ago"
# phrasing a human would use anyway. Floor 30 s so a typo can't busy-loop it.
_POLL_SECONDS = _env_int("JARVIS_HOMELAB_POLL_SECONDS", 60, 30)

# Flap damping: consecutive failed polls required before OK → DOWN. 3 rides
# out a transient blip (one dropped packet, the laptop briefly busy) without
# meaningfully delaying a real alert. Floor 1 = "alert on the first failure"
# (no damping) for anyone who wants it.
_FAIL_THRESHOLD = _env_int("JARVIS_HOMELAB_FAIL_THRESHOLD", 3, 1)

# Disk alert threshold: a drive with less than this percent FREE is "low".
# 10% is the usual "start worrying" line for a media/server box.
_DISK_MIN_PCT = _env_int("JARVIS_HOMELAB_DISK_MIN_PCT", 10, 1)

# Network timeouts. A down host makes the TCP connect take the full timeout
# to fail, so keep it short — three checks at ≤4 s each is still trivial
# inside a 60 s poll.
_CONNECT_TIMEOUT = 4.0
_HTTP_TIMEOUT = 5.0

# Ports probed. 22 = sshd (the laptop runs it for the M24 tools), used as the
# host-reachability signal. 32400 = Plex Media Server's HTTP API.
_SSH_PORT = 22
_PLEX_PORT = 32400


def _host_label() -> str:
    """Spoken name for the monitored box. Default is generic ('the Plex
    laptop') so nothing hardcodes a hostname; override with JARVIS_HOMELAB_LABEL
    (e.g. 'MEDIA-HOST') for a friendlier alert."""
    return os.getenv("JARVIS_HOMELAB_LABEL", "the Plex laptop").strip() or "the Plex laptop"


# --- Check primitives ------------------------------------------------------

class CheckResult(NamedTuple):
    """The outcome of one probe. `detail` is a short human phrase used in the
    alert text and the status report (empty when there is nothing to add)."""
    ok: bool
    detail: str


@dataclass(frozen=True)
class Check:
    """One monitored thing. `run` is the probe; `down`/`up` build the spoken
    alert for each edge (given the triggering result's detail); `status`
    renders the current-state phrase for the on-demand homelab_status tool.

    `depends_on` names another check — while THAT check is DOWN, this one is
    skipped (dependency suppression: a dead laptop should not also alarm about
    Plex and disk separately)."""
    name: str                                  # stable key: "host" / "plex" / "disk"
    title: str                                 # status-report heading
    run: Callable[[], CheckResult]
    down: Callable[[str], str]                 # OK → DOWN alert, given detail
    up: Callable[[str], str]                   # DOWN → OK alert, given detail
    status: Callable[[CheckResult], str]       # current-state phrase
    depends_on: str | None = None


class _CheckTracker:
    """Per-check edge-triggered alert state machine with flap damping.

    The entire interesting core of M56, deliberately isolated from all I/O so
    it is trivially unit-testable (see scripts/homelab_alert_test.py). Feed it
    a stream of ok/not-ok observations; it returns an edge to announce — or
    None — exactly once per real transition.
    """

    def __init__(self, fail_threshold: int) -> None:
        self._fail_threshold = max(1, fail_threshold)
        self._fails = 0          # consecutive failures observed (while still OK)
        self._down = False       # current hard state

    def observe(self, ok: bool) -> str | None:
        """Record one poll. Returns 'down' on the OK→DOWN edge, 'recovered'
        on the DOWN→OK edge, None on every non-transition (including the
        debounced failures BEFORE the threshold and the silent stay-down
        polls AFTER it). Edge-triggered: an alert fires once, never repeats."""
        if ok:
            self._fails = 0
            if self._down:
                self._down = False
                return "recovered"
            return None
        # A failure. If we are already DOWN, stay silent (edge-triggered).
        if self._down:
            return None
        self._fails += 1
        if self._fails >= self._fail_threshold:
            self._down = True
            return "down"
        return None        # damped — not enough consecutive failures yet

    def reset(self) -> None:
        """Back to a clean OK state. Called when monitoring is (re)activated,
        so a fresh arm never inherits a stale verdict. NOT called for
        dependency-suppressed checks — those FREEZE (keep their state), so a
        child recovery during a parent outage still fires its edge."""
        self._fails = 0
        self._down = False

    @property
    def is_down(self) -> bool:
        return self._down


# --- The monitor -----------------------------------------------------------

class HomelabMonitor:
    """Background homelab watcher. Construct once (the homelab_status tool and
    the tray toggle need the instance); the poll loop only runs between
    activate() and deactivate().

    Thread-safe surface: activate / deactivate / shutdown / is_active /
    status_report may be called from any thread.
    """

    def __init__(
        self,
        announce: Callable[[str], None],
        *,
        plex_host: str = "",
        plex_laptop_client=None,
        discord_webhook_url: str = "",
    ) -> None:
        self._announce = announce
        self._plex_host = (plex_host or "").strip()
        self._client = plex_laptop_client
        self._discord = (discord_webhook_url or "").strip()
        self._label = _host_label()

        self._checks = self._build_checks()
        self._trackers = {c.name: _CheckTracker(_FAIL_THRESHOLD) for c in self._checks}
        # Most recent result per check — feeds the telemetry/status path.
        self._last: dict[str, CheckResult] = {}

        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Register as the module's active monitor so the homelab_status tool
        # (a plain param-less function, decoupled from main.py's wiring) can
        # find us — the same file-is-the-IPC decoupling reminders.py uses,
        # via an object here since this state is in-memory, not persisted.
        _set_active_monitor(self)

    # ------------------------------------------------------------------
    # Check registry — only register what this install can actually reach.
    # ------------------------------------------------------------------

    def _build_checks(self) -> list[Check]:
        checks: list[Check] = []
        label = self._label
        if self._plex_host:
            checks.append(Check(
                name="host", title=label, run=self._check_host,
                down=lambda d: f"Sir — {label} has gone unreachable.",
                up=lambda d: f"{label} is back online, sir.",
                status=lambda r: "reachable" if r.ok else "unreachable",
            ))
            checks.append(Check(
                name="plex", title="Plex", run=self._check_plex, depends_on="host",
                down=lambda d: f"Sir — Plex on {label} has stopped responding.",
                up=lambda d: "Plex is responding again, sir.",
                status=lambda r: "responding" if r.ok else "not responding",
            ))
        if self._client is not None:
            checks.append(Check(
                name="disk", title="Disk space", run=self._check_disk,
                depends_on="host",
                down=lambda d: (
                    f"Sir — disk space on {label} is running low"
                    + (f": {d}." if d else ".")
                ),
                up=lambda d: f"Disk space on {label} has recovered, sir.",
                status=lambda r: (
                    ("healthy — " if r.ok else "LOW — ") + (r.detail or "usage unknown")
                ),
            ))
        return checks

    # ------------------------------------------------------------------
    # The probes. Each returns a CheckResult and never raises (the expected
    # failure modes are caught here; an unexpected exception is caught by
    # _safe_run and fails soft to OK).
    # ------------------------------------------------------------------

    def _check_host(self) -> CheckResult:
        """Host reachability — a plain TCP connect to sshd. No ICMP (would
        need admin), no subprocess; a refused/timed-out connect is the
        signal."""
        ok = _tcp_alive(self._plex_host, _SSH_PORT, _CONNECT_TIMEOUT)
        return CheckResult(ok, "" if ok else "no TCP response on port 22")

    def _check_plex(self) -> CheckResult:
        """Plex liveness — GET /identity, which returns 200 + XML with NO auth
        token. The cleanest "is the server process actually serving?" probe."""
        url = f"http://{self._plex_host}:{_PLEX_PORT}/identity"
        try:
            resp = httpx.get(url, timeout=_HTTP_TIMEOUT)
        except httpx.HTTPError:
            return CheckResult(False, "no response")
        if resp.status_code < 400:
            return CheckResult(True, "")
        return CheckResult(False, f"HTTP {resp.status_code}")

    def _check_disk(self) -> CheckResult:
        """Free space on the laptop's fixed drives. Alerts on the WORST drive.
        A read failure fails soft to OK — the host check already covers a real
        outage; a transient SSH hiccup must not manufacture a disk alarm."""
        from src.plex_laptop import fetch_disk_usage  # noqa: PLC0415 — lazy (paramiko)
        drives = fetch_disk_usage(self._client)
        if not drives:
            return CheckResult(True, "")
        worst = min(drives, key=lambda d: d["pct_free"])
        used = 100 - worst["pct_free"]
        detail = (f"drive {worst['id']} at {used}% used "
                  f"({worst['pct_free']}% free)")
        return CheckResult(worst["pct_free"] >= _DISK_MIN_PCT, detail)

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Start the background poll loop. Idempotent. Resets every tracker so
        a re-activation begins from a clean slate, never a stale verdict."""
        if self._active.is_set():
            return
        if not self._checks:
            # PLEX_LAPTOP_HOST unset and no SSH client → nothing to watch.
            print("[homelab] no checks configured (set PLEX_LAPTOP_HOST) — "
                  "not starting", file=sys.stderr)
            self._safe_announce(
                "There's nothing for me to monitor yet, sir — the homelab "
                "isn't configured."
            )
            return
        for tracker in self._trackers.values():
            tracker.reset()
        self._last.clear()
        self._active.set()
        self._stop.clear()
        names = ", ".join(c.name for c in self._checks)
        print(f"[homelab] monitoring activated — checks: {names}, "
              f"poll {_POLL_SECONDS}s, threshold {_FAIL_THRESHOLD}",
              file=sys.stderr)
        self._safe_announce("I'll keep an eye on the homelab, sir.")
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._watch_loop, name="HomelabMonitor", daemon=True
            )
            self._thread.start()

    def deactivate(self) -> None:
        """Stop the poll loop. Idempotent."""
        if not self._active.is_set():
            return
        self._active.clear()
        self._stop.set()
        print("[homelab] monitoring deactivated", file=sys.stderr)
        self._safe_announce("I'll stop watching the homelab, sir.")

    def shutdown(self) -> None:
        """App-quit: wind the thread down silently (it's a daemon, so it dies
        with the process either way; this just lets it exit its poll-wait
        cleanly without log noise)."""
        self._active.clear()
        self._stop.set()

    def is_active(self) -> bool:
        return self._active.is_set()

    # ------------------------------------------------------------------
    # Poll loop.
    # ------------------------------------------------------------------

    def _watch_loop(self) -> None:
        """Daemon: every _POLL_SECONDS, run every check and announce any edge.
        The whole body is wrapped — a dead scheduler would silently drop every
        future alert, so a bad poll must never kill the thread."""
        print("[homelab] monitor loop started", file=sys.stderr)
        while not self._stop.is_set() and self._active.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001 — keep the thread alive
                print(f"[homelab] poll failed: {exc}", file=sys.stderr)
            if self._stop.wait(_POLL_SECONDS):
                break
        print("[homelab] monitor loop exited", file=sys.stderr)

    def _poll_once(self) -> None:
        """One sweep over every check. Dependency-suppressed checks are
        skipped (their tracker FROZEN, not reset) so a parent outage cannot
        cascade into a pile of child alerts."""
        for check in self._checks:
            tracker = self._trackers[check.name]
            dep = check.depends_on
            if dep and self._trackers[dep].is_down:
                # Dependency suppression: the parent is DOWN, so we have no
                # real signal for this check — skip it. FREEZE the tracker,
                # do NOT reset it: if this check was already DOWN before the
                # parent failed, that verdict is real, not stale. Preserving
                # it means a genuine child recovery still fires its DOWN->OK
                # edge once the parent is back, and a still-down child does
                # not re-fire a duplicate DOWN. (M56 v1 reset() here — caught
                # in the first live test: a "Plex stopped responding" that
                # healed during a host outage never got its "Plex is
                # responding again", an alert that opened and never closed.)
                self._last[check.name] = CheckResult(True, "(suppressed)")
                continue
            result = self._safe_run(check)
            self._last[check.name] = result
            edge = tracker.observe(result.ok)
            if edge == "down":
                self._emit(check.down(result.detail))
            elif edge == "recovered":
                self._emit(check.up(result.detail))

    def _safe_run(self, check: Check) -> CheckResult:
        """Run a probe, catching anything it didn't. An exception out of run()
        is a bug, not a real outage — fail soft to OK + log, so a buggy probe
        degrades to 'no monitoring' rather than to a false alarm."""
        try:
            return check.run()
        except Exception as exc:  # noqa: BLE001
            print(f"[homelab] check '{check.name}' raised: {exc}", file=sys.stderr)
            return CheckResult(True, "")

    def _emit(self, text: str) -> None:
        """Announce an alert (spoken via the WASAPI-safe Announcer path) and,
        if configured, push it to Discord on a throwaway thread so a slow POST
        never delays the next poll."""
        print(f"[homelab] ALERT: {text}", file=sys.stderr)
        self._safe_announce(text)
        if self._discord:
            threading.Thread(
                target=self._push_discord, args=(text,),
                name="HomelabNotify", daemon=True,
            ).start()

    def _push_discord(self, text: str) -> None:
        from src.notifications import send_discord_message  # noqa: PLC0415 — lazy
        try:
            send_discord_message(self._discord, f"🖥 {text}")
        except Exception as exc:  # noqa: BLE001 — never break on a push
            print(f"[homelab] discord push failed: {exc}", file=sys.stderr)

    def _safe_announce(self, text: str) -> None:
        try:
            self._announce(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[homelab] announce failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # On-demand status — backs the homelab_status tool.
    # ------------------------------------------------------------------

    def status_report(self) -> str:
        """Probe everything NOW and return a short per-check status. Always
        fresh (a 'how's the homelab?' is a fresh-state request, like asking
        the time) — the background loop owns the alert state, this just looks.
        Dependency-aware: if the host is down, dependent checks read
        'unknown' rather than running a doomed probe."""
        if not self._checks:
            return ("The homelab isn't configured for monitoring, sir — "
                    "set PLEX_LAPTOP_HOST in the environment.")
        results: dict[str, CheckResult] = {}
        lines: list[str] = []
        for check in self._checks:
            dep = check.depends_on
            if dep and dep in results and not results[dep].ok:
                lines.append(f"- {check.title}: unknown ({self._label} unreachable)")
                continue
            result = self._safe_run(check)
            results[check.name] = result
            lines.append(f"- {check.title}: {check.status(result)}")
        state = ("I'm actively watching it." if self.is_active()
                 else "Background monitoring is currently off.")
        return f"Homelab status — {state}\n" + "\n".join(lines)


# --- helpers ---------------------------------------------------------------

def _tcp_alive(host: str, port: int, timeout: float) -> bool:
    """True if a TCP connection to host:port completes within `timeout`."""
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --- module-level active monitor (for the tool) ---------------------------
# The homelab_status tool executor is a plain param-less function (the
# _CLIENT_TOOLS contract). It reaches the live monitor through this module
# global, set by HomelabMonitor.__init__ — the same tool/runtime decoupling
# reminders.py gets from its JSON file.

_ACTIVE_MONITOR: HomelabMonitor | None = None


def _set_active_monitor(monitor: HomelabMonitor) -> None:
    global _ACTIVE_MONITOR
    _ACTIVE_MONITOR = monitor


# --- Anthropic tool: homelab_status ----------------------------------------

HOMELAB_STATUS_TOOL = {
    "name": "homelab_status",
    "description": (
        "Check the current health of the user's homelab right now — whether "
        "the Plex laptop is reachable, whether Plex Media Server is "
        "responding, and its disk space. Use this for 'how's the homelab?', "
        "'is Plex up?', 'is the Plex box okay?', 'check the homelab'. It "
        "probes live and returns a short per-check status; read it back "
        "concisely. This is the on-demand view — Jarvis also watches the "
        "homelab in the background (when monitoring is enabled) and reports "
        "problems on his own. For DETAILED CPU/RAM/disk figures on the Plex "
        "laptop use plex_laptop_health instead; this tool is the quick "
        "up/down roll-up."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


def execute_homelab_status(params: dict) -> str:  # noqa: ARG001 — param-less tool
    """homelab_status executor. Never raises — every failure is a readable
    string for Claude to voice."""
    monitor = _ACTIVE_MONITOR
    if monitor is None:
        return ("Homelab monitoring isn't available in this session, sir.")
    try:
        return monitor.status_report()
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"[homelab] status_report failed: {exc}", file=sys.stderr)
        return "I couldn't get the homelab status just now, sir."
