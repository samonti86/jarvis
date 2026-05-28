"""Good-night wrap (M63) — the symmetric evening counterpart to M55's
morning briefing.

"Jarvis, good night" → one composed spoken wrap-up: current security state,
tomorrow's first event, the reminders queued for tomorrow, and tomorrow's
weather. Bookends the day — M55 sets it up, M63 closes it out.

Design decisions (settled with the user before the build):
  - **One composition tool, mirroring M55.** `get_good_night` gathers every
    section itself and returns one block; Claude calls it once and voices
    it. Same shape as get_briefing — "the wrap" is a defined, reliable
    thing, not an emergent behaviour that depends on Claude remembering to
    call calendar + reminders + weather separately.
  - **Forward-looking, not backward.** A morning briefing sets up the day;
    an evening wrap sets up TOMORROW. Sections are tomorrow's first event,
    tomorrow's reminders, tomorrow's weather. (A "what happened today"
    recap would need a per-fire log the reminders store doesn't keep —
    deferred until the user actually wants it.)
  - **Fail-soft per section.** Each section is gathered independently and
    wrapped — a dead weather API, an Outlook outage, an unconfigured home
    location: one section degrades, never the whole wrap.
  - **Security state via an injected getter, not a global import.** main.py
    calls `register_security_getter(security_watcher.is_armed)` at startup;
    the module-level singleton decouples this tool from the security
    subsystem the same way `homelab_monitor`'s _ACTIVE_MONITOR decouples
    `homelab_status`. Unset getter ⇒ the security line silently omits.

v1 is on-demand. The *scheduled* version ("wrap me up every night at 10")
is wired through the M59 reminders dispatch — `action="good_night"`
triggers this composition the same way `action="briefing"` triggers M55.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Callable

from src.reminders import list_pending


# --- Anthropic tool definition ---------------------------------------------
GOOD_NIGHT_TOOL = {
    "name": "get_good_night",
    "description": (
        "Produce the user's 'good night' wrap — one composed evening "
        "update covering current security state, tomorrow's first event, "
        "the reminders queued for tomorrow, and tomorrow's weather. The "
        "symmetric counterpart to get_briefing's morning briefing. Use "
        "this whenever the user says 'good night', 'wrap up the day', "
        "'my evening wrap', 'end of day', or 'shut down for the night'. "
        "Read the result back as a natural, calm spoken wrap-up — greet "
        "the user warmly ('Good evening, sir'), then a sentence or two "
        "per section; do NOT recite verbatim. A wrap is a fresh-state "
        "request: ALWAYS call this tool even if you wrapped up earlier "
        "in this same conversation. The user may want it twice."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


# --- Security-state injection (the homelab_monitor decoupling pattern) -----

# Returns the live armed bool. Registered by main.py at startup; unset means
# the security subsystem isn't available, the section silently omits.
_security_getter: Callable[[], bool] | None = None


def register_security_getter(getter: Callable[[], bool]) -> None:
    """Wire main.py's live `security_watcher.is_armed` into this module so
    the wrap can report the current armed state without a stream_response
    parameter. Mirrors homelab_monitor's _set_active_monitor."""
    global _security_getter
    _security_getter = getter


# --- Section helpers (each fails soft; same contract as briefing.py) -------

def _home_location() -> str:
    return os.getenv("JARVIS_HOME_LOCATION", "").strip()


def _home_units() -> str:
    u = os.getenv("JARVIS_HOME_UNITS", "imperial").strip().lower()
    return u if u in ("imperial", "metric") else "imperial"


def _fmt_time(dt: datetime) -> str:
    """'8:05 AM' — leading zero stripped, for a spoken wrap."""
    return dt.strftime("%I:%M %p").lstrip("0")


def _security_section() -> str:
    """Current armed/standing-down state. Silently omitted when no getter is
    registered (the security subsystem either isn't initialised yet or this
    module was imported before main.py wired it). A reasonable evening line
    matters more than M60's terse 'Security: ARMED' format — Claude voices
    this directly to the user, so phrase it like a butler."""
    if _security_getter is None:
        return ""
    try:
        armed = bool(_security_getter())
    except Exception as exc:  # noqa: BLE001 — a section must never break the wrap
        print(f"[good_night] security read failed: {exc}", file=sys.stderr)
        return "Security: status unavailable."
    if armed:
        return "Security is armed for the night."
    return "Security is currently standing down."


def _calendar_section() -> str:
    """Tomorrow's first TIMED event (all-day items are skipped — they're
    visible in the morning briefing, and 'tomorrow's first thing' is about
    when you need to actually be somewhere). Silently omitted when no
    calendar backend is configured (the M62/M62.1 'don't nag unconfigured
    users' policy)."""
    try:
        from src.outlook_calendar import (  # noqa: PLC0415 — lazy
            CLIENT_ID, ICAL_URL, fetch_events_in_window,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[good_night] calendar import failed: {exc}", file=sys.stderr)
        return ""
    if not (CLIENT_ID or ICAL_URL):
        return ""
    try:
        now_local = datetime.now().astimezone()
        tomorrow_start = (now_local.replace(hour=0, minute=0, second=0,
                                            microsecond=0)
                          + timedelta(days=1))
        tomorrow_end = tomorrow_start + timedelta(days=1)
        events, err = fetch_events_in_window(tomorrow_start, tomorrow_end)
    except Exception as exc:  # noqa: BLE001
        print(f"[good_night] calendar fetch failed: {exc}", file=sys.stderr)
        return "Tomorrow's calendar: status unavailable."
    if err:
        # Configured but auth expired / Graph erroring. SURFACE it — a
        # silent skip here would hide a missed meeting (same reasoning as
        # M55's _calendar_section).
        return f"Tomorrow's calendar: {err}"
    timed = [e for e in (events or []) if not e.is_all_day]
    if not timed:
        return "Tomorrow's calendar: nothing scheduled."
    first = timed[0]
    line = (f"Tomorrow's first event: {_fmt_time(first.start_local)} — "
            f"{first.subject}")
    if first.location:
        line += f" ({first.location})"
    return line + "."


def _reminders_section() -> str:
    """Pending reminders whose next fire is TOMORROW (local date). Mirrors
    briefing.py:_reminders_section's filtering on the M53/M54 store, swung
    forward one day."""
    try:
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        items: list[tuple[datetime, str]] = []
        for r in list_pending():
            try:
                dt = datetime.fromisoformat(r.get("fire_at", ""))
            except (ValueError, TypeError):
                continue
            if dt.date() == tomorrow:
                items.append((dt, r.get("message", "?")))
        if not items:
            return "Reminders tomorrow: none scheduled."
        items.sort()
        lines = [f"Reminders tomorrow ({len(items)}):"]
        for dt, msg in items:
            lines.append(f'- "{msg}" at {_fmt_time(dt)}')
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        print(f"[good_night] reminders section failed: {exc}", file=sys.stderr)
        return "Reminders tomorrow: unavailable."


def _weather_section() -> str:
    """Tomorrow's forecast — high/low + outlook + precip chance. Uses the
    new `weather.tomorrow_forecast` helper, same pattern as M55 calls
    `execute_weather_tool` with mode='today'."""
    home = _home_location()
    if not home:
        return ("Weather: not configured — set JARVIS_HOME_LOCATION in .env "
                "to include tomorrow's forecast.")
    try:
        from src.weather import tomorrow_forecast  # noqa: PLC0415 — lazy
        result = tomorrow_forecast(home, _home_units())
        return f"Weather — {result}"
    except Exception as exc:  # noqa: BLE001
        print(f"[good_night] weather section failed: {exc}", file=sys.stderr)
        return "Weather: tomorrow unavailable just now."


def execute_good_night_tool(params: dict) -> str:  # noqa: ARG001 — param-less
    """Compose the evening wrap. Never raises — every section fails soft on
    its own. Section order: security (the 'is the house safe' line a user
    actually wants to hear last thing before bed), then forward-looking
    tomorrow (calendar, reminders, weather)."""
    sections = [
        _security_section(),
        _calendar_section(),
        _reminders_section(),
        _weather_section(),
    ]
    # Drop empties so silent-omit sections (security unconfigured, calendar
    # unconfigured) don't produce double-blank-line gaps in the output.
    return "\n\n".join(s for s in sections if s)
