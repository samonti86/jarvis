"""Good-morning briefing tool (M55) — one composed start-to-the-day reply.

"Jarvis, good morning" → a single spoken briefing: today's weather, sports
and gaming headlines, anything the security watcher caught overnight, and
the reminders due today. The films' JARVIS greeting Tony with the state of
his world before he's had coffee.

Design decisions (settled with the user before the build):
  - **One composition tool, not Claude-orchestrates.** `get_briefing`
    gathers every section itself and returns one block; Claude calls it once
    and voices it. That makes "the briefing" a *defined, reliable* thing —
    not an emergent behaviour that depends on Claude remembering to call
    weather, then news, then... separately.
  - **Sports + gaming, not general news.** The user's explicit call: general
    and world news are draining, and a *morning* briefing should set the day
    up, not weigh it down. The news section is `get_news`'s sports (NHL /
    NFL / WWE) and gaming (PlayStation / Nintendo) categories — added to
    `news.py` for exactly this.
  - **Fail-soft per section.** Each section is gathered independently and
    wrapped — a dead news feed or a weather outage degrades that one line,
    never blanks the briefing. Same never-raises contract as every tool.
  - **Self-contained config.** Reads its own `JARVIS_HOME_LOCATION` /
    `JARVIS_HOME_UNITS` env at call time (the wolfram/tmdb pattern — a
    tool-specific setting doesn't belong in the core Config). Unset home
    location ⇒ the weather line is politely omitted, briefing still works.

v1 is on-demand only. The *scheduled* version ("brief me every morning at
7") is a clean follow-on — the M54 recurring-reminder hook exists, but a
recurring reminder currently speaks a static string; wiring one to trigger
this composition is a bit more plumbing, deliberately deferred.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

from src.news import execute_news_tool
from src.reminders import list_pending
from src.weather import execute_weather_tool


# --- Anthropic tool definition ---------------------------------------------
BRIEFING_TOOL = {
    "name": "get_briefing",
    "description": (
        "Produce the user's 'good morning' briefing — one composed update "
        "covering today's weather, sports and gaming headlines, any overnight "
        "security events, and the reminders due today. Use this whenever the "
        "user says 'good morning', asks for 'my briefing' / 'the morning "
        "briefing', or asks 'what's my day looking like'. Read the result "
        "back as a natural, concise spoken briefing — greet the user, then a "
        "sentence or two per section; do NOT read it out verbatim or recite "
        "every headline."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

# How far back the "overnight" security section looks. 12h comfortably spans
# a night for a morning briefing; it's a plain window, not a clock-anchored
# "since you went to bed", which keeps it correct whenever the briefing runs.
_OVERNIGHT_HOURS = 12


def _home_location() -> str:
    return os.getenv("JARVIS_HOME_LOCATION", "").strip()


def _home_units() -> str:
    u = os.getenv("JARVIS_HOME_UNITS", "imperial").strip().lower()
    return u if u in ("imperial", "metric") else "imperial"


def _fmt_time(dt: datetime) -> str:
    """'8:05 AM' — leading zero stripped, for a spoken briefing."""
    return dt.strftime("%I:%M %p").lstrip("0")


def _weather_section() -> str:
    """Today's forecast for the configured home location. Omitted (with a
    one-time hint) when JARVIS_HOME_LOCATION isn't set."""
    home = _home_location()
    if not home:
        return ("Weather: not configured — set JARVIS_HOME_LOCATION in .env "
                "to include the local forecast.")
    try:
        result = execute_weather_tool(
            {"location": home, "mode": "today", "units": _home_units()}
        )
        return f"Weather — {result}"
    except Exception as exc:  # noqa: BLE001 — a section must never break the briefing
        print(f"[briefing] weather section failed: {exc}", file=sys.stderr)
        return "Weather: unavailable just now."


def _news_section(category: str, label: str) -> str:
    """A 3-headline pull from a get_news category (sports / gaming)."""
    try:
        return execute_news_tool({"category": category, "limit": 3})
    except Exception as exc:  # noqa: BLE001
        print(f"[briefing] {category} section failed: {exc}", file=sys.stderr)
        return f"{label}: unavailable just now."


def _security_section() -> str:
    """Count the M35 deterrent-evidence snapshots ({timestamp}.jpg) written in
    the last _OVERNIGHT_HOURS. File mtime is the signal — robust regardless of
    the filename format. No dir / no recent files ⇒ 'all quiet'."""
    try:
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
        events_dir = base / "security" / "events"
        if not events_dir.is_dir():
            return "Overnight security: all quiet — nothing to report."
        cutoff = time.time() - _OVERNIGHT_HOURS * 3600
        recent: list[float] = []
        for f in events_dir.glob("*.jpg"):
            try:
                mt = f.stat().st_mtime
            except OSError:
                continue
            if mt >= cutoff:
                recent.append(mt)
        if not recent:
            return ("Overnight security: all quiet — no events in the last "
                    f"{_OVERNIGHT_HOURS} hours.")
        recent.sort()
        latest = _fmt_time(datetime.fromtimestamp(recent[-1]))
        n = len(recent)
        return (f"Overnight security: {n} security event"
                f"{'s' if n != 1 else ''} in the last {_OVERNIGHT_HOURS} "
                f"hours — latest at {latest}. Evidence was saved.")
    except Exception as exc:  # noqa: BLE001
        print(f"[briefing] security section failed: {exc}", file=sys.stderr)
        return "Overnight security: status unavailable."


def _calendar_section() -> str:
    """Today's Outlook calendar events (M62). Silently omitted when
    OUTLOOK_CLIENT_ID isn't set — we don't want to nag a user who never
    configured the integration. Configured-but-unauthorised / configured-
    but-Graph-erroring states ARE surfaced so the user knows to re-auth
    (silent failure here would mean a missed meeting)."""
    try:
        from src.outlook_calendar import (  # noqa: PLC0415 — lazy
            CLIENT_ID, ICAL_URL, today_events,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[briefing] calendar import failed: {exc}", file=sys.stderr)
        return ""
    if not (CLIENT_ID or ICAL_URL):
        # Silently omit when neither backend is configured. M62.1 added the
        # iCal path — the original CLIENT_ID-only check would have silently
        # skipped the section for users who configured iCal but not Graph.
        return ""
    events, err = today_events()
    if err:
        # Configured but auth expired / Graph erroring. SURFACE this — a
        # silent skip here would hide a missed meeting.
        return f"Calendar: {err}"
    if events is None or len(events) == 0:
        return "Calendar: nothing scheduled today."
    n = len(events)
    lines = [f"Calendar — {n} event{'s' if n != 1 else ''} today:"]
    for ev in events:
        if ev.is_all_day:
            lines.append(f"- (all day) {ev.subject}")
        else:
            line = f"- {ev.start_local.strftime('%H:%M')} {ev.subject}"
            if ev.location:
                line += f"  ({ev.location})"
            lines.append(line)
    return "\n".join(lines)


def _reminders_section() -> str:
    """The pending reminders whose next fire is today (M53/M54 store)."""
    try:
        today = datetime.now().date()
        todays: list[tuple[datetime, str]] = []
        for r in list_pending():
            try:
                dt = datetime.fromisoformat(r.get("fire_at", ""))
            except (ValueError, TypeError):
                continue
            if dt.date() == today:
                todays.append((dt, r.get("message", "?")))
        if not todays:
            return "Reminders today: none scheduled."
        todays.sort()
        lines = [f"Reminders today ({len(todays)}):"]
        for dt, msg in todays:
            lines.append(f'- "{msg}" at {_fmt_time(dt)}')
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        print(f"[briefing] reminders section failed: {exc}", file=sys.stderr)
        return "Reminders today: unavailable."


def execute_briefing_tool(params: dict) -> str:  # noqa: ARG001 — param-less tool
    """Compose the morning briefing. Never raises — every section fails soft
    on its own, so one dead feed or outage costs a line, not the briefing.
    Section order: practical first (weather, overnight security, the day's
    reminders), the lighter sports + gaming to finish on."""
    sections = [
        _weather_section(),
        _security_section(),
        _calendar_section(),    # M62 — silent skip when Outlook isn't configured
        _reminders_section(),
        _news_section("sports", "Sports"),
        _news_section("gaming", "Gaming"),
    ]
    # Drop empty sections so an unconfigured optional channel (currently
    # only calendar) doesn't produce a double-blank-line gap in the output.
    return "\n\n".join(s for s in sections if s)
