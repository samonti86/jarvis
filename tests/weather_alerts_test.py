"""Unit tests for severe-weather proactive alerts (M77).

Network-free. Covers the pure cores — NWS GeoJSON parsing, the env-driven
severity set, the announce decision, the power-risk keyword match, local-time
formatting, and the speech builder (incl. the no-UPS shutdown nudge) — plus
`fetch_active_alerts` with httpx monkeypatched and `execute_weather_alerts_tool`
with the geocode + fetch monkeypatched.

    python tests/weather_alerts_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from src import weather_alerts as wa  # noqa: E402
from src.weather_alerts import (  # noqa: E402
    GET_WEATHER_ALERTS_TOOL,
    WeatherAlert,
    _alert_speech,
    _announce_severities,
    _fmt_local_time,
    _parse_features,
    _power_risk,
    _should_announce,
    execute_weather_alerts_tool,
    fetch_active_alerts,
)


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


def _alert(aid="urn:1", event="Severe Thunderstorm Warning", severity="Severe",
           headline="", area="Orange, FL", ends_iso=None):
    return WeatherAlert(id=aid, event=event, severity=severity,
                        headline=headline, area=area, ends_iso=ends_iso)


def _set_env(name, value):
    import os
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


# --- Schema --------------------------------------------------------------

print("\nschema:")
check("tool name", GET_WEATHER_ALERTS_TOOL.get("name") == "get_weather_alerts")
check("description mentions alert/warning",
      "alert" in GET_WEATHER_ALERTS_TOOL["description"].lower()
      or "warning" in GET_WEATHER_ALERTS_TOOL["description"].lower())
check("location is optional (no required list)",
      not GET_WEATHER_ALERTS_TOOL["input_schema"].get("required"))


# --- _parse_features -----------------------------------------------------

print("\n_parse_features:")
payload = {"features": [
    {"properties": {"id": "urn:a", "event": "Severe Thunderstorm Warning",
                    "severity": "Severe", "headline": "SVR until 8",
                    "areaDesc": "Orange, FL",
                    "ends": "2026-06-05T20:00:00-04:00"}},
    {"properties": {"id": "urn:b", "event": "Rip Current Statement",
                    "severity": "Moderate", "areaDesc": "Coast",
                    "expires": "2026-06-05T18:00:00-04:00"}},  # uses expires
    {"properties": {"event": "No-ID alert", "severity": "Severe"}},  # skipped
]}
alerts = _parse_features(payload)
check("parses the two id'd features (drops the id-less one)", len(alerts) == 2)
check("first event + severity", alerts[0].event == "Severe Thunderstorm Warning"
      and alerts[0].severity == "Severe")
check("ends taken from 'ends'", alerts[0].ends_iso == "2026-06-05T20:00:00-04:00")
check("ends falls back to 'expires'",
      alerts[1].ends_iso == "2026-06-05T18:00:00-04:00")
check("empty payload -> []", _parse_features({}) == [])


# --- _announce_severities (env-driven) ----------------------------------

print("\n_announce_severities:")
_set_env("JARVIS_WEATHER_ALERT_SEVERITY", None)
check("default -> {Severe, Extreme}",
      _announce_severities() == {"Severe", "Extreme"})
_set_env("JARVIS_WEATHER_ALERT_SEVERITY", "extreme")
check("'extreme' -> {Extreme} only", _announce_severities() == {"Extreme"})
_set_env("JARVIS_WEATHER_ALERT_SEVERITY", "moderate")
check("'moderate' -> {Moderate, Severe, Extreme}",
      _announce_severities() == {"Moderate", "Severe", "Extreme"})
_set_env("JARVIS_WEATHER_ALERT_SEVERITY", "garbage")
check("garbage -> defaults to severe floor",
      _announce_severities() == {"Severe", "Extreme"})
_set_env("JARVIS_WEATHER_ALERT_SEVERITY", None)


# --- _should_announce ----------------------------------------------------

print("\n_should_announce:")
sev = {"Severe", "Extreme"}
check("severe + unseen -> True",
      _should_announce(_alert(severity="Severe"), set(), sev) is True)
check("extreme + unseen -> True",
      _should_announce(_alert(severity="Extreme"), set(), sev) is True)
check("moderate (below floor) -> False",
      _should_announce(_alert(severity="Moderate"), set(), sev) is False)
check("already announced -> False",
      _should_announce(_alert(aid="urn:x"), {"urn:x"}, sev) is False)
# Casing robustness: NWS emits Title-case today, but the match must not go dark
# if a feed ever returns a different case (the on-demand path already lowercases).
check("UPPERCASE severity still fires",
      _should_announce(_alert(severity="SEVERE"), set(), sev) is True)
check("lowercase severity still fires",
      _should_announce(_alert(severity="extreme"), set(), sev) is True)


# --- _power_risk ---------------------------------------------------------

print("\n_power_risk:")
check("Severe Thunderstorm Warning -> power risk",
      _power_risk("Severe Thunderstorm Warning") is True)
check("Tornado Warning -> power risk", _power_risk("Tornado Warning") is True)
check("Hurricane Warning -> power risk", _power_risk("Hurricane Warning") is True)
check("Winter Storm Warning -> power risk",
      _power_risk("Winter Storm Warning") is True)
check("High Wind Warning -> power risk (the 'wind' keyword)",
      _power_risk("High Wind Warning") is True)
check("Flood Warning -> NOT power risk", _power_risk("Flood Warning") is False)
check("Excessive Heat Warning -> NOT power risk",
      _power_risk("Excessive Heat Warning") is False)
check("Rip Current Statement -> NOT power risk",
      _power_risk("Rip Current Statement") is False)


# --- _fmt_local_time (build input in LOCAL tz so it's tz-independent) ----

print("\n_fmt_local_time:")
local_tz = datetime.now().astimezone().tzinfo
top = datetime(2026, 6, 5, 20, 0, tzinfo=local_tz).isoformat()    # 8:00 PM local
half = datetime(2026, 6, 5, 20, 30, tzinfo=local_tz).isoformat()  # 8:30 PM local
check("on-the-hour -> '8 PM'", _fmt_local_time(top) == "8 PM")
check("with minutes -> '8:30 PM'", _fmt_local_time(half) == "8:30 PM")
check("None -> ''", _fmt_local_time(None) == "")
check("garbage -> ''", _fmt_local_time("not-a-date") == "")


# --- _alert_speech -------------------------------------------------------

print("\n_alert_speech:")
spk = _alert_speech(_alert(event="Severe Thunderstorm Warning",
                           ends_iso=datetime(2026, 6, 5, 20, 0,
                                             tzinfo=local_tz).isoformat()))
check("names the event", "Severe Thunderstorm Warning" in spk)
check("includes the end time", "until 8 PM" in spk)
check("power-risk event carries the no-UPS shutdown nudge",
      "shut it down" in spk.lower() and "ups" in spk.lower())
flood = _alert_speech(_alert(event="Flood Warning", ends_iso=None))
check("non-power event omits the shutdown nudge",
      "shut it down" not in flood.lower())
check("no end time -> no 'until' clause", "until" not in flood.lower())


# --- fetch_active_alerts (httpx monkeypatched) --------------------------

print("\nfetch_active_alerts:")
_orig_get = httpx.get


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


try:
    httpx.get = lambda *a, **k: _Resp(200, payload)
    alerts, err = fetch_active_alerts(28.5, -81.4)
    check("200 -> (alerts, None)", err is None and alerts is not None
          and len(alerts) == 2)

    httpx.get = lambda *a, **k: _Resp(500, {})
    alerts, err = fetch_active_alerts(28.5, -81.4)
    check("500 -> (None, 'HTTP 500')", alerts is None and "500" in (err or ""))

    httpx.get = lambda *a, **k: _Resp(200, ValueError("bad"))
    alerts, err = fetch_active_alerts(28.5, -81.4)
    check("bad JSON -> (None, err)", alerts is None and err is not None)

    def _boom(*a, **k):
        raise httpx.ConnectError("no network")
    httpx.get = _boom
    alerts, err = fetch_active_alerts(28.5, -81.4)
    check("transport error -> (None, err) (no raise)",
          alerts is None and err is not None)
finally:
    httpx.get = _orig_get


# --- execute_weather_alerts_tool (geocode + fetch monkeypatched) --------

print("\nexecutor:")
from src import weather as weather_mod  # noqa: E402
_orig_geocode = weather_mod._geocode
_orig_fetch = wa.fetch_active_alerts


def _install(geocode_ret, fetch_ret):
    weather_mod._geocode = lambda loc: geocode_ret
    wa.fetch_active_alerts = lambda lat, lon: fetch_ret


def _restore():
    weather_mod._geocode = _orig_geocode
    wa.fetch_active_alerts = _orig_fetch


# No location at all (param empty + env empty).
_set_env("JARVIS_HOME_LOCATION", None)
out = execute_weather_alerts_tool({})
check("no location -> setup hint",
      "home_location" in out.lower() or "tell me a place" in out.lower())

# Geocode miss.
_install(None, ([], None))
try:
    out = execute_weather_alerts_tool({"location": "Nowheresville"})
    check("geocode miss -> couldn't find", "couldn't find" in out.lower())
finally:
    _restore()

# Fetch error.
_install((28.5, -81.4, "Springfield, IL"), (None, "HTTP 500"))
try:
    out = execute_weather_alerts_tool({"location": "Springfield, IL"})
    check("fetch error -> 'unavailable'", "unavailable" in out.lower())
finally:
    _restore()

# No active alerts.
_install((28.5, -81.4, "Springfield, IL"), ([], None))
try:
    out = execute_weather_alerts_tool({"location": "Springfield, IL"})
    check("no alerts -> 'No active weather alerts'",
          "no active weather alerts" in out.lower())
finally:
    _restore()

# Active alerts -> listed, most-severe first.
_install((28.5, -81.4, "Springfield, IL"), ([
    _alert(aid="u1", event="Flood Watch", severity="Moderate"),
    _alert(aid="u2", event="Tornado Warning", severity="Extreme"),
], None))
try:
    out = execute_weather_alerts_tool({"location": "Springfield, IL"})
    check("lists both events", "Tornado Warning" in out and "Flood Watch" in out)
    check("Extreme sorted before Moderate",
          out.index("Tornado Warning") < out.index("Flood Watch"))
    check("includes severity labels", "[Extreme]" in out and "[Moderate]" in out)
finally:
    _restore()
    _set_env("JARVIS_HOME_LOCATION", None)


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
