"""'where to stream' — unit tests for the TMDB watch-providers mode.

The hard parts of mode=providers on get_movie_tv_info:
  - region keying: TMDB returns availability per ISO-3166-1 country code, and
    we read the user's region from TMDB_WATCH_REGION (default 'US').
  - category → spoken-label mapping (flatrate=Stream, free/ads, rent, buy) and
    read-out order.
  - the per-category provider cap (_MAX_PROVIDERS).
  - the empty-but-present-region case (region key exists with only a JustWatch
    link, no stream/rent/buy) vs the region-absent case.
  - never-raises contract: missing key, no match, transport failure, malformed
    JSON all become voice-friendly strings.

Same monkeypatch shape as tests/person_info_test.py: replace the HTTP call
site (`_http_get_with_retry` inside `src.tmdb`) with a stub returning canned,
URL-keyed responses. No real network.

    python tests/watch_providers_test.py   # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tmdb  # noqa: E402
from src.tmdb import TMDB_TOOL, execute_tmdb_tool  # noqa: E402


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


# --- HTTP stubbing -------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload


class _HttpStub:
    """Routes (url, params) requests to canned payloads keyed on a URL
    substring. A payload of None simulates a transport failure; a ValueError
    instance simulates malformed JSON."""

    def __init__(self):
        self.routes: dict[str, object] = {}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, params=None, **_kwargs):
        self.calls.append((url, params or {}))
        for needle, payload in self.routes.items():
            if needle in url:
                if payload is None:
                    return None
                return _FakeResponse(payload)
        return None


_orig_http = tmdb._http_get_with_retry
_orig_key_fn = tmdb._api_key


def install(stub):
    tmdb._http_get_with_retry = stub
    tmdb._api_key = lambda: "TESTKEY"


def restore():
    tmdb._http_get_with_retry = _orig_http
    tmdb._api_key = _orig_key_fn


# --- Fixtures ------------------------------------------------------------

_DUNE = {"id": 438631, "title": "Dune", "media_type": "movie"}

_DUNE_PROVIDERS = {
    "id": 438631,
    "results": {
        "US": {
            "link": "https://www.themoviedb.org/movie/438631/watch?locale=US",
            "flatrate": [{"provider_name": "Max"}, {"provider_name": "Hulu"}],
            "rent": [{"provider_name": "Apple TV"},
                     {"provider_name": "Amazon Video"}],
            "buy": [{"provider_name": "Apple TV"}],
        },
        "GB": {
            "flatrate": [{"provider_name": "Netflix"}],
        },
    },
}


# --- Schema --------------------------------------------------------------

print("\nTMDB_TOOL schema:")
mode_enum = TMDB_TOOL["input_schema"]["properties"]["mode"]["enum"]
check("mode enum includes 'providers'", "providers" in mode_enum)
check("tool description mentions stream/rent/buy",
      "stream" in TMDB_TOOL["description"].lower())


# --- Missing API key -----------------------------------------------------

print("\nmissing API key:")
saved_key = os.environ.pop("TMDB_API_KEY", None)
restore()  # real _api_key reads env (now missing)
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("missing key -> voice-friendly setup hint",
          "api key" in out.lower() or "tmdb_api_key" in out.lower())
finally:
    if saved_key is not None:
        os.environ["TMDB_API_KEY"] = saved_key


# --- Missing query -------------------------------------------------------

print("\nmissing query:")
stub = _HttpStub()
install(stub)
try:
    out = execute_tmdb_tool({"mode": "providers"})
    check("providers with no query -> 'title is required'",
          "title is required" in out.lower())
finally:
    restore()


# --- No match ------------------------------------------------------------

print("\nno match:")
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": []}
install(stub)
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Nonexistent Film"})
    check("no search match -> 'no movie or show found'",
          "no movie or show found" in out.lower())
finally:
    restore()


# --- Search transport failure -------------------------------------------

print("\nsearch transport failure:")
stub = _HttpStub()
stub.routes["/search/multi"] = None
install(stub)
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("search None -> 'unavailable' (no crash)",
          "unavailable" in out.lower())
finally:
    restore()


# --- providers transport failure ----------------------------------------

print("\nproviders transport failure:")
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": [_DUNE]}
stub.routes["/watch/providers"] = None
install(stub)
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("providers call None -> 'unavailable' (no crash)",
          "unavailable" in out.lower())
finally:
    restore()


# --- providers malformed JSON -------------------------------------------

print("\nproviders malformed JSON:")
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": [_DUNE]}
stub.routes["/watch/providers"] = ValueError("bad json")
install(stub)
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("malformed providers JSON -> 'malformed' (no crash)",
          "malformed" in out.lower())
finally:
    restore()


# --- Happy path (US default region) -------------------------------------

print("\nhappy path (US):")
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": [_DUNE]}
stub.routes["/watch/providers"] = _DUNE_PROVIDERS
install(stub)
saved_region = os.environ.pop("TMDB_WATCH_REGION", None)  # force default US
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("header names the title + region", "Dune" in out and "US" in out)
    check("flatrate rendered as 'Stream:' with providers",
          "Stream:" in out and "Max" in out and "Hulu" in out)
    check("rent rendered as 'Rent:'",
          "Rent:" in out and "Amazon Video" in out)
    check("buy rendered as 'Buy:'", "Buy:" in out and "Apple TV" in out)
    check("read-out order: Stream before Rent before Buy",
          out.index("Stream:") < out.index("Rent:") < out.index("Buy:"))
    check("resolved-id used in providers URL (438631)",
          any("/438631/watch/providers" in u for u, _ in stub.calls))
finally:
    restore()
    if saved_region is not None:
        os.environ["TMDB_WATCH_REGION"] = saved_region


# --- Region override (TMDB_WATCH_REGION=GB) -----------------------------

print("\nregion override (GB):")
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": [_DUNE]}
stub.routes["/watch/providers"] = _DUNE_PROVIDERS
install(stub)
saved_region = os.environ.get("TMDB_WATCH_REGION")
os.environ["TMDB_WATCH_REGION"] = "gb"  # lowercase -> proves we uppercase
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("GB region picks the GB list (Netflix)",
          "Netflix" in out and "GB" in out)
    check("GB region does NOT bleed the US-only Max",
          "Max" not in out)
finally:
    restore()
    if saved_region is None:
        os.environ.pop("TMDB_WATCH_REGION", None)
    else:
        os.environ["TMDB_WATCH_REGION"] = saved_region


# --- Region absent from results -----------------------------------------

print("\nregion absent:")
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": [_DUNE]}
stub.routes["/watch/providers"] = _DUNE_PROVIDERS  # has US + GB, not DE
install(stub)
saved_region = os.environ.get("TMDB_WATCH_REGION")
os.environ["TMDB_WATCH_REGION"] = "DE"
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("region with no data -> 'don't have any streaming availability'",
          "don't have any streaming availability" in out.lower()
          and "DE" in out)
finally:
    restore()
    if saved_region is None:
        os.environ.pop("TMDB_WATCH_REGION", None)
    else:
        os.environ["TMDB_WATCH_REGION"] = saved_region


# --- Region present but no stream/rent/buy options ----------------------

print("\nregion present but empty:")
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": [_DUNE]}
# US key exists but only carries a JustWatch link — no flatrate/rent/buy.
stub.routes["/watch/providers"] = {
    "id": 438631,
    "results": {"US": {"link": "https://example.com/watch"}},
}
install(stub)
saved_region = os.environ.pop("TMDB_WATCH_REGION", None)
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("present-but-empty region -> 'isn't currently listed'",
          "isn't currently listed" in out.lower())
finally:
    restore()
    if saved_region is not None:
        os.environ["TMDB_WATCH_REGION"] = saved_region


# --- free / ads categories ----------------------------------------------

print("\nfree / ads categories:")
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": [_DUNE]}
stub.routes["/watch/providers"] = {
    "id": 438631,
    "results": {"US": {
        "free": [{"provider_name": "YouTube Free"}],
        "ads": [{"provider_name": "Tubi"}, {"provider_name": "Pluto TV"}],
    }},
}
install(stub)
saved_region = os.environ.pop("TMDB_WATCH_REGION", None)
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    check("free rendered as 'Free:'", "Free:" in out and "YouTube Free" in out)
    check("ads rendered as 'Free with ads:'",
          "Free with ads:" in out and "Tubi" in out and "Pluto TV" in out)
finally:
    restore()
    if saved_region is not None:
        os.environ["TMDB_WATCH_REGION"] = saved_region


# --- per-category provider cap (_MAX_PROVIDERS) -------------------------

print("\nprovider cap:")
many = [{"provider_name": f"Svc{i}"} for i in range(10)]
stub = _HttpStub()
stub.routes["/search/multi"] = {"results": [_DUNE]}
stub.routes["/watch/providers"] = {
    "id": 438631, "results": {"US": {"flatrate": many}},
}
install(stub)
saved_region = os.environ.pop("TMDB_WATCH_REGION", None)
try:
    out = execute_tmdb_tool({"mode": "providers", "query": "Dune"})
    shown = sum(1 for i in range(10) if f"Svc{i}" in out)
    check(f"caps providers at {tmdb._MAX_PROVIDERS} (shown={shown})",
          shown == tmdb._MAX_PROVIDERS)
    check("the (cap+1)th provider is dropped",
          f"Svc{tmdb._MAX_PROVIDERS}" not in out)
finally:
    restore()
    if saved_region is not None:
        os.environ["TMDB_WATCH_REGION"] = saved_region


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
