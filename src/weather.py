"""Weather info tool — current conditions, today's forecast, multi-day forecast.

Backs onto Open-Meteo's free public API (open-meteo.com). No API key, no
account, global coverage. Two-hop call:
  1. geocoding-api.open-meteo.com → resolve "Denver" to (25.7617, -80.1918)
  2. api.open-meteo.com → forecast at those coords

Same defensive contract as src/sports.py: never raises, always returns a
readable string. WMO weather codes are mapped to short voice phrases
("partly cloudy", "light rain") — raw integers don't TTS well.

Coverage: any place Open-Meteo's geocoder recognizes — most of the world's
cities, towns, and well-known regions.
"""

from __future__ import annotations

import sys
from datetime import datetime
from functools import partial
from typing import Any

from src.http_util import http_get_with_retry


# --- Anthropic tool definition ---------------------------------------------
# Description tells Claude when to invoke and how to fill `units`. The
# units hint nudges metric for non-US / Spanish-speaking turns without
# needing logic in our code.
WEATHER_TOOL = {
    "name": "get_weather",
    "description": (
        "Get current weather, today's forecast, or a multi-day forecast for "
        "any city or location worldwide. Use this for any weather question. "
        "Geocoding is built in — pass a natural place name like 'Denver', "
        "'Mexico City', or 'London, UK'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "City or place name. 'Denver', 'Mexico City', 'London', "
                    "'Tokyo', 'San Francisco' — anything Open-Meteo's "
                    "geocoder recognizes."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["current", "today", "forecast"],
                "description": (
                    "current = right-now conditions; today = today's high/low "
                    "and outlook; forecast = next 5 days."
                ),
            },
            "units": {
                "type": "string",
                "enum": ["imperial", "metric"],
                "description": (
                    "imperial = Fahrenheit + mph + inches (default for US "
                    "users); metric = Celsius + km/h + mm (use this when the "
                    "user is speaking Spanish or asking about a non-US "
                    "location)."
                ),
            },
        },
        "required": ["location", "mode"],
    },
}


_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# M43: retry helper extracted to src/http_util.py (was triplicated across
# sports/weather/games — the rule-of-three extraction the M22 note flagged).
# partial() pre-binds the log tag so the _http_get_with_retry(...) call sites
# below stay unchanged. The "transport errors only, never 4xx/5xx" policy and
# the 10s timeout / 0.5s backoff defaults moved with it, unchanged.
_http_get_with_retry = partial(http_get_with_retry, tag="weather")


# WMO weather interpretation codes collapsed to voice-friendly phrases.
# Full table: https://open-meteo.com/en/docs (search "WMO Weather interpretation").
# We bucket the slight/moderate/heavy variants where TTS doesn't benefit
# from the extra precision.
_WMO_PHRASES: dict[int, str] = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


def _wmo_phrase(code: int | None) -> str:
    if code is None:
        return "unknown conditions"
    return _WMO_PHRASES.get(int(code), f"weather code {code}")


def _temp_unit(units: str) -> str:
    return "°C" if units == "metric" else "°F"


def _geocode_query(name: str) -> tuple[float, float, str] | None:
    """One geocoding lookup for a bare place name. Returns None on miss or
    any error — caller treats None as 'place not found'."""
    r = _http_get_with_retry(
        _GEOCODE_URL,
        params={
            "name": name,
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )
    if r is None:
        return None
    try:
        data = r.json()
    except ValueError as exc:
        print(f"[weather] geocode JSON parse failed ({name}): {exc}", file=sys.stderr)
        return None

    results = data.get("results") or []
    if not results:
        return None
    top = results[0]
    lat = top.get("latitude")
    lon = top.get("longitude")
    if lat is None or lon is None:
        return None

    # Build a friendly display string ("Denver, Florida, United States").
    # admin1 is the state/province; country is the country.
    place = top.get("name") or name
    admin1 = top.get("admin1")
    country = top.get("country")
    display = ", ".join(p for p in (place, admin1, country) if p)
    return float(lat), float(lon), display


def _geocode(location: str) -> tuple[float, float, str] | None:
    """Resolve a place name to (lat, lon, display_name). Returns None on miss.

    Open-Meteo's geocoder matches a bare place NAME — it does NOT understand a
    "City, State" / "City, Country" suffix and returns nothing for one
    (verified: "Fort Lauderdale, FL" misses every time, "Fort Lauderdale"
    resolves). But "City, ST" is exactly what a user naturally writes — in
    JARVIS_HOME_LOCATION, or what Claude may pass through. So if the full
    string misses and it contains a comma, retry with just the part before
    the first comma. The resolved result's display name still carries the
    state/country, so the user can see which place it picked."""
    hit = _geocode_query(location)
    if hit is None and "," in location:
        hit = _geocode_query(location.split(",", 1)[0].strip())
    return hit


def _fetch_forecast(
    lat: float, lon: float, units: str, mode: str
) -> dict[str, Any]:
    """One Open-Meteo forecast call. Returns {} on any error.

    We ask only for the fields each mode needs — keeps the payload small
    and the tool result short (Claude has to read it back into context).
    """
    is_metric = units == "metric"
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "temperature_unit": "celsius" if is_metric else "fahrenheit",
        "wind_speed_unit": "kmh" if is_metric else "mph",
        "precipitation_unit": "mm" if is_metric else "inch",
        "timezone": "auto",  # so daily buckets align to local midnight
    }
    if mode == "current":
        params["current"] = (
            "temperature_2m,apparent_temperature,relative_humidity_2m,"
            "weather_code,wind_speed_10m"
        )
    if mode in ("today", "forecast"):
        params["daily"] = (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max"
        )
        params["forecast_days"] = 1 if mode == "today" else 5

    r = _http_get_with_retry(_FORECAST_URL, params=params)
    if r is None:
        return {}
    try:
        return r.json()
    except ValueError as exc:
        print(f"[weather] forecast JSON parse failed: {exc}", file=sys.stderr)
        return {}


def _format_current(data: dict, place: str, units: str) -> str:
    """Voice line for 'right now'. Includes feels-like only when it
    differs from the actual temp by 2°+ — otherwise it's noise."""
    cur = data.get("current") or {}
    temp = cur.get("temperature_2m")
    feels = cur.get("apparent_temperature")
    code = cur.get("weather_code")
    if temp is None:
        return f"No current weather data available for {place}."
    u = _temp_unit(units)
    phrase = _wmo_phrase(code)
    out = f"{round(temp)}{u} and {phrase} in {place}"
    if feels is not None and abs(feels - temp) >= 2:
        out += f", feels like {round(feels)}{u}"
    return out + "."


def _format_today(data: dict, place: str, units: str) -> str:
    """Voice line for today's high/low + outlook + precip chance."""
    daily = data.get("daily") or {}
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    codes = daily.get("weather_code") or []
    pops = daily.get("precipitation_probability_max") or []
    if not highs or not lows:
        return f"No forecast data available for {place}."
    u = _temp_unit(units)
    phrase = _wmo_phrase(codes[0] if codes else None)
    out = f"{place}: high {round(highs[0])}{u}, low {round(lows[0])}{u}, {phrase}"
    pop = pops[0] if pops else None
    if pop is not None and pop > 0:
        out += f", {int(pop)}% chance of precipitation"
    return out + "."


def _format_forecast(data: dict, place: str, units: str) -> str:
    """5-day forecast as one line per day. Claude usually compresses this
    further when reading back to the user — that's fine, the data is here
    if the user asks about a specific day."""
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    codes = daily.get("weather_code") or []
    if not times or not highs or not lows:
        return f"No forecast data available for {place}."
    u = _temp_unit(units)
    lines = [f"{place} 5-day forecast:"]
    for i, day_str in enumerate(times):
        try:
            day_label = datetime.strptime(day_str, "%Y-%m-%d").strftime("%a")
        except (ValueError, TypeError):
            day_label = day_str
        hi = round(highs[i]) if i < len(highs) else "?"
        lo = round(lows[i]) if i < len(lows) else "?"
        phrase = _wmo_phrase(codes[i] if i < len(codes) else None)
        lines.append(f"{day_label}: {hi}/{lo}{u}, {phrase}")
    return "\n".join(lines)


def execute_weather_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises."""
    location = (params.get("location") or "").strip()
    mode = (params.get("mode") or "current").lower().strip()
    units = (params.get("units") or "imperial").lower().strip()
    if units not in ("imperial", "metric"):
        units = "imperial"

    if not location:
        return "Location is required for weather queries."

    geocoded = _geocode(location)
    if geocoded is None:
        return f"Could not find a location matching '{location}'."
    lat, lon, place = geocoded

    data = _fetch_forecast(lat, lon, units, mode)
    if not data:
        return f"Weather service unreachable for {place}."

    if mode == "today":
        return _format_today(data, place, units)
    if mode == "forecast":
        return _format_forecast(data, place, units)
    # Default and explicit "current" both fall here.
    return _format_current(data, place, units)
