"""Severe-weather proactive alerts (M77) — Jarvis watches the sky.

The reactive→proactive jump for weather, and personalised to a documented
pain point: the user's desktop has NO UPS, the BIOS won't auto-power-on, and he
manually shuts down for known storms to avoid a hard power-loss (see the
power-loss memory). Today that depends on *him* noticing the forecast. This
module closes that gap — a background poll of the National Weather Service
active-alerts feed for the home location that speaks up on its own:

    "Sir — the National Weather Service has issued a Severe Thunderstorm
     Warning, until 8 PM. With no UPS on the desktop, you may want to shut it
     down before the power goes."

Almost entirely a recombination of the proven monitor pattern:
  - thread model  = calendar_monitor / homelab_monitor — a defensive daemon
                    poll loop, a stop Event, every poll wrapped so one bad
                    fetch can never kill the thread.
  - dedupe        = calendar_monitor.DedupeStore (reused — see note below):
                    an alert is a one-shot edge, announce it exactly once,
                    and survive a restart mid-storm without re-announcing.
  - voice out     = main.py's _announce (the WASAPI-safe Announcer path),
                    tagged ⛈ so weather reads distinctly from 📅/🖥/🔔/🚨/⏰.
  - phone push    = notifications.send_discord_message (same channel as M56+).
  - geocode       = weather._geocode (reused — handles the "City, ST" comma
                    case Open-Meteo's geocoder otherwise misses).

Data source: the NWS API (api.weather.gov, free, no key, US only). Direct
httpx call (not the shared http_util) — like wolfram.py, because we set a
custom User-Agent (NWS asks every client to identify itself) and the error
handling is simple. Non-US home locations just return no alerts (harmless).

Configuration / defaults
------------------------
DEFAULT ON when JARVIS_HOME_LOCATION is set (the calendar-monitor convention,
not homelab's opt-in) — the whole value is the unprompted warning, and the
user explicitly asked for it. Kill switch: JARVIS_WEATHER_ALERTS=0. Only
Severe + Extreme alerts are announced proactively (advisories like a Rip
Current Statement would be noise); the on-demand get_weather_alerts tool
reports everything, since there the user asked.

Tunables (env, read once at import):
  - JARVIS_WEATHER_POLL_SECONDS    — seconds between polls (600, min 60)
  - JARVIS_WEATHER_ALERT_SEVERITY  — lowest severity to announce: "extreme",
                                     "severe" (default), or "moderate"

Defensive contract — same as every monitor: nothing here raises into the poll
thread or the listen loop. A bad fetch, a geocode miss, a malformed feature —
all become log lines and a fail-soft pass.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

# DedupeStore is reused from calendar_monitor — it's a generic
# path+retain+thread-safe atomic-save dedupe set, already unit-tested
# (scripts/calendar_monitor_test.py). This is its SECOND consumer; per the
# rule-of-three it stays put. If a THIRD monitor needs it, extract it to a
# shared src/dedupe_store.py then (and update both imports + the test).
from src.calendar_monitor import DedupeStore


# --- Tunables (module-local, env-driven; the calendar_monitor pattern) ------

def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, value)


# Poll cadence. Weather alerts are issued well ahead (a severe-thunderstorm
# warning is typically 30-60 min of lead) and don't change minute to minute,
# so 10 min keeps us a good NWS citizen while still timely. Floor 60 s so a
# typo can't hammer the API.
_POLL_SECONDS = _env_int("JARVIS_WEATHER_POLL_SECONDS", 600, 60)

# NWS severity ladder: Extreme > Severe > Moderate > Minor > Unknown. We
# announce Severe + Extreme by default (the power-threatening tier); a user
# who wants advisories too can drop the floor to "moderate".
_SEVERITY_RANK = {"unknown": 0, "minor": 1, "moderate": 2, "severe": 3, "extreme": 4}


def _announce_severities() -> set[str]:
    """The set of NWS severity strings to announce proactively, from
    JARVIS_WEATHER_ALERT_SEVERITY (the FLOOR; default 'severe')."""
    floor = os.getenv("JARVIS_WEATHER_ALERT_SEVERITY", "severe").strip().lower()
    floor_rank = _SEVERITY_RANK.get(floor, 3)  # default to 'severe'
    return {sev.title() for sev, rank in _SEVERITY_RANK.items()
            if rank >= floor_rank and rank > 0}


def _home_location() -> str:
    """Read JARVIS_HOME_LOCATION live (the briefing/calendar pattern), so the
    monitor reflects the current .env without a re-import."""
    return os.getenv("JARVIS_HOME_LOCATION", "").strip()


def _kill_switch_set() -> bool:
    """JARVIS_WEATHER_ALERTS=0/false/off/no disables the monitor even when a
    home location is configured. Default: enabled when configured."""
    raw = os.getenv("JARVIS_WEATHER_ALERTS", "").strip().lower()
    return raw in ("0", "false", "no", "off")


# Event-name keywords for the "no UPS — shut down" nudge. These are the
# power-threatening events; a flood / heat / rip-current alert is real but
# doesn't argue for shutting the PC down, so it gets the plain announce.
_POWER_RISK_KEYWORDS = (
    "thunderstorm", "tornado", "hurricane", "tropical storm", "tropical depression",
    "high wind", "wind", "winter storm", "ice storm", "blizzard", "snow squall",
)


def _power_risk(event: str) -> bool:
    e = (event or "").lower()
    return any(k in e for k in _POWER_RISK_KEYWORDS)


# --- NWS data model + fetch ------------------------------------------------

# NWS asks every client to send a User-Agent identifying the app + a contact.
_NWS_BASE = "https://api.weather.gov"
_USER_AGENT = "Jarvis personal assistant (github.com/samonti86/jarvis)"


@dataclass(frozen=True)
class WeatherAlert:
    id: str
    event: str
    severity: str
    headline: str
    area: str
    ends_iso: str | None


def _parse_features(data: dict) -> list[WeatherAlert]:
    """Extract WeatherAlerts from an NWS GeoJSON payload. Skips features
    without a stable id (the dedupe key). Never raises."""
    out: list[WeatherAlert] = []
    for feat in (data.get("features") or []):
        if not isinstance(feat, dict):
            continue
        p = feat.get("properties") or {}
        aid = p.get("id") or feat.get("id") or p.get("@id")
        if not aid:
            continue
        out.append(WeatherAlert(
            id=str(aid),
            event=str(p.get("event") or "Weather alert"),
            severity=str(p.get("severity") or "Unknown"),
            headline=str(p.get("headline") or ""),
            area=str(p.get("areaDesc") or ""),
            ends_iso=(p.get("ends") or p.get("expires")),
        ))
    return out


def fetch_active_alerts(lat: float, lon: float) -> tuple[list[WeatherAlert] | None, str | None]:
    """Fetch active NWS alerts for a point. Returns (alerts, None) on success
    or (None, error_string) on failure — never raises. Direct httpx (not
    http_util) so we can set the NWS-requested User-Agent."""
    import httpx  # noqa: PLC0415 — lazy (already a project dep)
    try:
        r = httpx.get(
            f"{_NWS_BASE}/alerts/active",
            params={"point": f"{lat},{lon}"},
            headers={"User-Agent": _USER_AGENT, "Accept": "application/geo+json"},
            timeout=20.0,
            follow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 — network can fail many ways
        return None, f"request failed: {type(exc).__name__}: {exc}"
    if r.status_code >= 400:
        return None, f"HTTP {r.status_code}"
    try:
        data = r.json()
    except ValueError as exc:
        return None, f"bad JSON: {exc}"
    return _parse_features(data), None


# --- Pure decision + formatting (testable in isolation) --------------------

def _should_announce(alert: WeatherAlert, announced: set[str],
                     severities: set[str]) -> bool:
    """Pure: announce this alert iff its severity is in the announce set AND
    it hasn't already been announced. Isolated from fetch / announce / disk so
    the firing rule is structurally testable."""
    if alert.severity not in severities:
        return False
    if alert.id in announced:
        return False
    return True


def _fmt_local_time(iso: str | None) -> str:
    """NWS end time (ISO-8601 with offset) → a terse local clock string like
    '8 PM' / '8:30 PM'. Empty string on missing/unparseable (caller omits)."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso).astimezone()
    except (ValueError, TypeError):
        return ""
    hour = dt.strftime("%I").lstrip("0") or "12"
    if dt.minute:
        return f"{hour}:{dt.strftime('%M')} {dt.strftime('%p')}"
    return f"{hour} {dt.strftime('%p')}"


def _alert_speech(alert: WeatherAlert) -> str:
    """The spoken announce text for a fired alert."""
    text = (f"Sir — the National Weather Service has issued a {alert.event}.")
    ends = _fmt_local_time(alert.ends_iso)
    if ends:
        text += f" In effect until {ends}."
    if _power_risk(alert.event):
        text += (" With no UPS on the desktop, you may want to shut it down "
                 "before the power goes.")
    return text


# --- The monitor -----------------------------------------------------------

def _store_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    return base / "weather_alerts_announced.json"


class WeatherAlertMonitor:
    """Background NWS severe-weather watcher. Announces (spoken + Discord) on
    each new Severe/Extreme alert for the home location, exactly once, and
    survives a restart mid-storm via an on-disk dedupe set.

    Constructed unconditionally so main.py's status registration always has a
    target; activate() is a no-op (logged) when no home location is configured
    or the kill switch is set — same shape as the calendar/homelab monitors.
    """

    def __init__(
        self,
        announce: Callable[[str], None],
        *,
        discord_webhook_url: str = "",
    ) -> None:
        self._announce = announce
        self._discord = (discord_webhook_url or "").strip()
        self._store = DedupeStore(_store_path())
        self._severities = _announce_severities()

        # Geocoded home point, resolved lazily on the first poll and cached.
        self._latlon: tuple[float, float] | None = None
        self._geocode_failed = False

        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def activate(self) -> None:
        if self._active.is_set():
            return
        if _kill_switch_set():
            print("[weather] JARVIS_WEATHER_ALERTS disabled — not starting",
                  file=sys.stderr)
            return
        if not _home_location():
            print("[weather] no JARVIS_HOME_LOCATION — alerts monitor not "
                  "starting", file=sys.stderr)
            return
        self._active.set()
        self._stop.clear()
        print(f"[weather] alerts monitor activated — poll {_POLL_SECONDS}s, "
              f"announce {sorted(self._severities)}, dedupe-size "
              f"{len(self._store)}", file=sys.stderr)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._watch_loop, name="WeatherAlertMonitor", daemon=True,
            )
            self._thread.start()

    def deactivate(self) -> None:
        if not self._active.is_set():
            return
        self._active.clear()
        self._stop.set()
        print("[weather] alerts monitor deactivated", file=sys.stderr)

    def shutdown(self) -> None:
        self._active.clear()
        self._stop.set()

    def is_active(self) -> bool:
        return self._active.is_set()

    def dedupe_size(self) -> int:
        return len(self._store)

    # -- poll loop ---------------------------------------------------------

    def _resolve_home(self) -> tuple[float, float] | None:
        """Geocode the home location once, caching the result. Returns None on
        a miss (retried on the next poll unless it permanently fails)."""
        if self._latlon is not None:
            return self._latlon
        from src.weather import _geocode  # noqa: PLC0415 — reuse the comma-aware geocoder
        hit = _geocode(_home_location())
        if hit is None:
            print("[weather] could not geocode home location — will retry",
                  file=sys.stderr)
            return None
        lat, lon, display = hit
        self._latlon = (lat, lon)
        print(f"[weather] home resolved to {display} ({lat:.3f},{lon:.3f})",
              file=sys.stderr)
        return self._latlon

    def _watch_loop(self) -> None:
        print("[weather] alerts loop started", file=sys.stderr)
        while not self._stop.is_set() and self._active.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001 — keep the thread alive
                print(f"[weather] poll failed: {exc}", file=sys.stderr)
            if self._stop.wait(_POLL_SECONDS):
                break
        print("[weather] alerts loop exited", file=sys.stderr)

    def _poll_once(self) -> None:
        latlon = self._resolve_home()
        if latlon is None:
            return
        alerts, err = fetch_active_alerts(*latlon)
        if err or alerts is None:
            if err:
                print(f"[weather] fetch err (will retry): {err[:200]}",
                      file=sys.stderr)
            return

        announced = self._store.snapshot()
        fired_any = False
        for alert in alerts:
            if not _should_announce(alert, announced, self._severities):
                continue
            # mark BEFORE emit — a raised announce path still must not
            # repeat-fire next poll.
            self._store.mark(alert.id)
            announced.add(alert.id)
            self._emit(alert)
            fired_any = True
        if fired_any:
            self._store.save()

    def _emit(self, alert: WeatherAlert) -> None:
        text = _alert_speech(alert)
        print(f"[weather] ANNOUNCE: {alert.event} ({alert.severity})",
              file=sys.stderr)
        self._safe_announce(text)
        if self._discord:
            push = f"⛈ {alert.headline or alert.event}"
            if alert.area:
                push += f" — {alert.area}"
            threading.Thread(
                target=self._push_discord, args=(push,),
                name="WeatherNotify", daemon=True,
            ).start()

    def _push_discord(self, text: str) -> None:
        from src.notifications import send_discord_message  # noqa: PLC0415
        try:
            send_discord_message(self._discord, text)
        except Exception as exc:  # noqa: BLE001 — never break on a push
            print(f"[weather] discord push failed: {exc}", file=sys.stderr)

    def _safe_announce(self, text: str) -> None:
        try:
            self._announce(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[weather] announce failed: {exc}", file=sys.stderr)


# --- On-demand tool (get_weather_alerts) — the reactive companion ----------
# Mirrors homelab_status' relationship to the homelab monitor: same data, but
# asked on demand and reporting EVERYTHING active (not just Severe+), since the
# user explicitly asked. Reads JARVIS_HOME_LOCATION itself (the briefing
# pattern) so it works whether or not the monitor is running.

GET_WEATHER_ALERTS_TOOL = {
    "name": "get_weather_alerts",
    "description": (
        "Check for ACTIVE official weather alerts, watches, and warnings for "
        "the user's home area (US National Weather Service). Use for 'any "
        "weather alerts?', 'is there a storm warning?', 'are we under a "
        "watch?'. Reports all active alerts with their severity. This is "
        "OFFICIAL alerts, distinct from get_weather (the forecast). US only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "Optional place to check (e.g. 'Springfield, IL'). Omit to use "
                    "the user's configured home location."
                ),
            },
        },
    },
}


def execute_weather_alerts_tool(params: dict) -> str:
    """Run the on-demand tool. Always returns a string — never raises."""
    location = (params.get("location") or "").strip() or _home_location()
    if not location:
        return ("No location to check, sir — set JARVIS_HOME_LOCATION in .env "
                "or tell me a place.")
    try:
        from src.weather import _geocode  # noqa: PLC0415
        hit = _geocode(location)
        if hit is None:
            return f"I couldn't find '{location}' to check alerts, sir."
        lat, lon, display = hit
        alerts, err = fetch_active_alerts(lat, lon)
        if err or alerts is None:
            return f"The weather-alert service is unavailable right now, sir ({err})."
        if not alerts:
            return f"No active weather alerts for {display}, sir."
        # Sort most-severe first so Claude reads the important one first.
        alerts = sorted(
            alerts,
            key=lambda a: _SEVERITY_RANK.get(a.severity.lower(), 0),
            reverse=True,
        )
        lines = [f"Active weather alerts for {display}:"]
        for a in alerts[:8]:
            ends = _fmt_local_time(a.ends_iso)
            tail = f" (until {ends})" if ends else ""
            lines.append(f"  - {a.event} [{a.severity}]{tail}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"[weather] get_weather_alerts failed: {exc}", file=sys.stderr)
        return "I couldn't check the weather alerts just now, sir."
