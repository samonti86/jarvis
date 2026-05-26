"""Outlook calendar awareness (M62) — read-only Microsoft Graph integration.

Jarvis reads the user's personal Microsoft account calendar (outlook.com /
hotmail.com / live.com) via the Graph API, scoped to `Calendars.Read` only.
The morning briefing gains today's events, and Claude has a tool for
on-demand queries ("what's on my calendar this week?", "do I have anything
later?").

Two paths, separated by design:

1. **One-time interactive OAuth** via `scripts/outlook_auth.py`. The MSAL
   device-code flow prints a code + URL; the user enters them once at
   microsoft.com/devicelogin. MSAL serializes a refresh token to
   `%LOCALAPPDATA%\\Jarvis\\msal_cache.bin`. THIS is the only blocking
   user-interactive path; it runs ONCE and is done.

2. **Silent silent runtime** via the tool / briefing. Every subsequent Graph
   call uses `acquire_token_silent()` — auto-refresh, no user interaction,
   fast. If the cache is missing or the refresh token expired, the tool /
   briefing return a clear "please re-authorise" message rather than block
   on an interactive prompt. (A voice assistant can't usefully run
   device-code in the middle of a tool call — it would block Claude's reply
   for up to 15 minutes.)

Architecture
------------
- `msal.PublicClientApplication` with authority
  `https://login.microsoftonline.com/consumers` — the personal-account
  endpoint (user chose this over work/school in the M62 design Q&A).
- `Calendars.Read` scope only; read-only — Jarvis cannot create or modify
  events. Future write capability would need a NEW scope + a different tool.
- Graph endpoint `GET /me/calendarView?startDateTime=…&endDateTime=…` —
  this is the *expanded-recurrence* view (vs. `/events` which returns the
  master + exceptions), so "today's events" correctly includes the Nth
  instance of a weekly meeting.
- Times come back in UTC by default; we convert to local on display.
- Token cache is plain JSON in `%LOCALAPPDATA%\\Jarvis\\msal_cache.bin`,
  protected by NTFS ACLs (only the Windows user can read it). For a future
  hardening pass we'd switch to `msal-extensions` for DPAPI encryption.

Defensive contract — every tool / briefing entry point returns a readable
string and never raises. A Graph 5xx, an expired refresh token, a missing
client ID, network blip — all become voice-friendly messages.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import httpx


# --- Configuration ---------------------------------------------------------

CLIENT_ID = os.getenv("OUTLOOK_CLIENT_ID", "").strip()

# Personal Microsoft accounts (outlook.com / hotmail.com / live.com). For
# work/school accounts swap this for the org's tenant ID — different
# milestone if/when needed.
AUTHORITY = "https://login.microsoftonline.com/consumers"

SCOPES = ["Calendars.Read"]
GRAPH_API = "https://graph.microsoft.com/v1.0"


def _cache_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    return base / "msal_cache.bin"


# --- OAuth client (msal wrapper) ------------------------------------------

class _OAuthClient:
    """Wraps `msal.PublicClientApplication` with a file-backed token cache.

    `silent_token()` is the hot path used by the tool / briefing — fast,
    never blocks for user interaction. `acquire_via_device_code()` is the
    one-shot setup path used by `scripts/outlook_auth.py` only.
    """

    def __init__(self, client_id: str) -> None:
        # Imported here so the module is importable even before msal is
        # installed (we surface a clear error from get_access_token instead).
        import msal  # noqa: PLC0415 — lazy
        self._cache = msal.SerializableTokenCache()
        self._cache_path = _cache_path()
        self._load_cache()
        self._app = msal.PublicClientApplication(
            client_id, authority=AUTHORITY, token_cache=self._cache,
        )

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            try:
                self._cache.deserialize(
                    self._cache_path.read_text(encoding="utf-8"),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[outlook] cache read failed: {exc}", file=sys.stderr)

    def _save_cache(self) -> None:
        if not self._cache.has_state_changed:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                self._cache.serialize(), encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[outlook] cache write failed: {exc}", file=sys.stderr)

    def silent_token(self) -> str | None:
        """Return a current access token from the cache, refreshing if
        needed. None if no token is cached — caller must surface 'please
        authorise' rather than block on a device-code flow."""
        accounts = self._app.get_accounts()
        if not accounts:
            return None
        result = self._app.acquire_token_silent(SCOPES, account=accounts[0])
        if not result or "access_token" not in result:
            return None
        self._save_cache()
        return result["access_token"]

    def acquire_via_device_code(
        self, on_prompt: Callable[[str], None] | None = None,
    ) -> bool:
        """The interactive one-shot setup path (`scripts/outlook_auth.py`).
        Blocks for the user to enter the code at microsoft.com/devicelogin
        (up to 15 minutes — MSAL polls Microsoft). `on_prompt` receives the
        full user-facing message (URL + code + countdown) so a caller can
        ALSO speak it; the message is always printed."""
        flow = self._app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            print("[outlook] initiate_device_flow failed: "
                  f"{flow.get('error_description', flow)}", file=sys.stderr)
            return False
        print(flow["message"])
        if on_prompt is not None:
            try:
                on_prompt(flow["message"])
            except Exception:
                pass  # never break auth on a logging callback
        result = self._app.acquire_token_by_device_flow(flow)
        if "access_token" not in result:
            print(f"[outlook] device flow failed: "
                  f"{result.get('error_description', result)}", file=sys.stderr)
            return False
        self._save_cache()
        return True


_client: _OAuthClient | None = None


def _get_client() -> _OAuthClient | None:
    """Process-wide singleton. Returns None if OUTLOOK_CLIENT_ID is unset,
    or if msal isn't installed."""
    global _client
    if not CLIENT_ID:
        return None
    if _client is None:
        try:
            _client = _OAuthClient(CLIENT_ID)
        except ImportError as exc:
            print(f"[outlook] msal not installed: {exc}", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[outlook] client init failed: {exc}", file=sys.stderr)
            return None
    return _client


# --- Graph queries --------------------------------------------------------

@dataclass(frozen=True)
class CalendarEvent:
    """Normalised calendar event — just the fields we need for voice +
    briefing, with times pre-converted to local."""
    subject: str
    start_local: datetime
    end_local: datetime
    location: str
    is_all_day: bool


def _fetch_events(start_utc: datetime, end_utc: datetime) -> tuple[list[CalendarEvent] | None, str]:
    """Returns (events, status). On success: (list, ""). On any failure:
    (None, voice-friendly error message). Status messages are intended for
    direct relay to the user."""
    client = _get_client()
    if client is None:
        return (None, "Outlook calendar isn't configured, sir — set "
                "OUTLOOK_CLIENT_ID in .env. See .env.example for the "
                "one-time setup.")
    token = client.silent_token()
    if not token:
        return (None, "Outlook calendar authorisation has expired or "
                "isn't set up yet, sir — run "
                "`python scripts/outlook_auth.py` from the project folder "
                "to authorise me, then try again.")

    # Graph expects ISO 8601; the /calendarView endpoint expands recurring
    # events into instances. Without a `Prefer: outlook.timezone="…"`
    # header, both query params and response times default to UTC — we
    # convert to local on the way out so the user sees their wall clock.
    url = f"{GRAPH_API}/me/calendarView"
    params = {
        "startDateTime": start_utc.replace(tzinfo=None).isoformat(),
        "endDateTime": end_utc.replace(tzinfo=None).isoformat(),
        "$select": "subject,start,end,location,isAllDay",
        "$orderby": "start/dateTime",
        "$top": "50",
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=15.0)
    except httpx.HTTPError as exc:
        print(f"[outlook] graph fetch failed: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return (None, "I couldn't reach Microsoft Graph just now, sir.")
    if resp.status_code == 401:
        # Token rejected — cache is stale somehow. Caller can prompt re-auth.
        print(f"[outlook] graph 401: {resp.text[:200]}", file=sys.stderr)
        return (None, "Outlook says my calendar authorisation has been "
                "revoked, sir — run `python scripts/outlook_auth.py` to "
                "re-authorise.")
    if resp.status_code != 200:
        print(f"[outlook] graph HTTP {resp.status_code}: "
              f"{resp.text[:200]}", file=sys.stderr)
        return (None, f"Microsoft Graph returned an error "
                f"(HTTP {resp.status_code}), sir.")
    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"[outlook] graph response parse failed: {exc}", file=sys.stderr)
        return (None, "I couldn't parse the calendar response, sir.")
    raw = payload.get("value") or []
    events = [_normalise(ev) for ev in raw]
    return ([e for e in events if e is not None], "")


def _normalise(ev: dict) -> CalendarEvent | None:
    """Graph event dict → CalendarEvent with local times. None on a
    malformed entry (logged + skipped, not raised)."""
    try:
        subject = (ev.get("subject") or "(no subject)").strip() or "(no subject)"
        location = (ev.get("location") or {}).get("displayName") or ""
        location = location.strip()
        is_all_day = bool(ev.get("isAllDay", False))
        start_str = (ev.get("start") or {}).get("dateTime") or ""
        end_str = (ev.get("end") or {}).get("dateTime") or ""
        start_utc = _parse_graph_dt(start_str)
        end_utc = _parse_graph_dt(end_str)
        if start_utc is None or end_utc is None:
            return None
        return CalendarEvent(
            subject=subject,
            start_local=start_utc.astimezone(),
            end_local=end_utc.astimezone(),
            location=location,
            is_all_day=is_all_day,
        )
    except Exception as exc:  # noqa: BLE001 — defensive per-event
        print(f"[outlook] event normalise failed: {exc}", file=sys.stderr)
        return None


def _parse_graph_dt(s: str) -> datetime | None:
    """Graph returns ISO 8601 like '2026-05-26T13:30:00.0000000' — usually
    naive but representing UTC (no Prefer header). Handles a trailing Z
    too if Microsoft ever changes their mind."""
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Strip sub-microsecond fractional seconds Graph sometimes sends
        # ('.0000000') which fromisoformat rejects on older Pythons.
        try:
            base, _, _ = s.partition(".")
            dt = datetime.fromisoformat(base)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# --- Public helpers (used by briefing + tool) -----------------------------

def today_events() -> tuple[list[CalendarEvent] | None, str]:
    """Events from local-midnight today through local-midnight tomorrow."""
    now_local = datetime.now().astimezone()
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return _fetch_events(start_local.astimezone(timezone.utc),
                         end_local.astimezone(timezone.utc))


# --- Anthropic tool -------------------------------------------------------

GET_CALENDAR_TOOL = {
    "name": "get_calendar_events",
    "description": (
        "Read events from the user's Outlook calendar (personal Microsoft "
        "account, read-only). Use this for 'what's on my calendar', 'do I "
        "have anything later', 'what's my next meeting', 'am I free at 3pm', "
        "'what's tomorrow looking like'. Returns a list of events with "
        "time + subject (+ location when present). Read it back "
        "conversationally — don't enumerate the JSON; mention the times "
        "and subjects naturally. If the calendar isn't configured or "
        "authorisation has expired, the tool returns a clear setup "
        "instruction — relay that plainly and don't fabricate events. The "
        "user cannot create or modify events through Jarvis (the scope is "
        "Calendars.Read only); if they ask to schedule something, say so."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "timeframe": {
                "type": "string",
                "enum": ["today", "tomorrow", "this_week", "next_24h"],
                "description": (
                    "Which window to fetch. "
                    "'today' = local midnight to midnight (the default); "
                    "'tomorrow' = the next calendar day; "
                    "'this_week' = today through the end of the upcoming "
                    "Sunday; "
                    "'next_24h' = the rolling next 24 hours from now "
                    "(use this for 'what's coming up' / 'anything later')."
                ),
            },
        },
    },
}


def _window_for(timeframe: str) -> tuple[datetime, datetime, str] | None:
    """Resolve a timeframe enum to (start_local, end_local, label).
    None if the timeframe is unknown."""
    now_local = datetime.now().astimezone()
    today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if timeframe == "today":
        return (today_start, today_start + timedelta(days=1), "today")
    if timeframe == "tomorrow":
        start = today_start + timedelta(days=1)
        return (start, start + timedelta(days=1), "tomorrow")
    if timeframe == "this_week":
        # today through the end of this Sunday (treating Monday as the start
        # of the week for compatibility with Python's weekday() = 0..6,
        # Mon..Sun). On Sunday, "this week" is just today.
        weekday = now_local.weekday()        # Mon=0 … Sun=6
        days_until_sun_end = 6 - weekday     # 0 if Sunday
        end = today_start + timedelta(days=days_until_sun_end + 1)
        return (today_start, end, "this week")
    if timeframe == "next_24h":
        return (now_local, now_local + timedelta(hours=24),
                "in the next 24 hours")
    return None


def execute_calendar_tool(params: dict) -> str:
    """Tool executor. Never raises — every failure returns a voice-friendly
    string. Time format: HH:MM 24-hour local; Claude voices these as the
    user prefers."""
    raw = (params.get("timeframe") or "today").strip().lower()
    win = _window_for(raw)
    if win is None:
        return (f"I don't recognise the timeframe '{raw}', sir — try "
                f"today, tomorrow, this_week, or next_24h.")
    start_local, end_local, label = win
    events, err = _fetch_events(
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
    )
    if err:
        return err
    return _format_events(events or [], label)


def _format_events(events: list[CalendarEvent], label: str) -> str:
    """Render a CalendarEvent list as a structured-but-spoken-friendly text
    block. Claude voices this conversationally; the format is just for the
    tool result."""
    if not events:
        return f"Nothing on your calendar {label}, sir."
    lines = [f"{len(events)} event(s) {label}:"]
    for ev in events:
        when = _format_event_time(ev)
        line = f"- {when}  {ev.subject}"
        if ev.location:
            line += f"  ({ev.location})"
        lines.append(line)
    return "\n".join(lines)


def _format_event_time(ev: CalendarEvent) -> str:
    if ev.is_all_day:
        return "all day"
    # Same-day window → just the times. Multi-day → include the date.
    same_day = ev.start_local.date() == ev.end_local.date()
    if same_day:
        return f"{ev.start_local.strftime('%H:%M')}–{ev.end_local.strftime('%H:%M')}"
    return (f"{ev.start_local.strftime('%a %H:%M')}–"
            f"{ev.end_local.strftime('%a %H:%M')}")
