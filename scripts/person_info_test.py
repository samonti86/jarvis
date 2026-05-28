"""M66 — unit tests for the TMDB person-filmography tool.

The hard parts:
  - department filtering (acting / directing / writing / all)
  - sort-by-release-date descending (with TBA dates sorted last)
  - the de-duplication in 'all' mode (a film someone wrote+directed shouldn't
    appear twice — keyed by title+year+job).
  - never-raises contract: missing key, no match, transport failure, malformed
    JSON all become voice-friendly strings.

Same monkeypatch shape as scripts/self_update_test.py: replace the HTTP
call site (`_http_get_with_retry` inside `src.tmdb`) with a stub that
returns canned (url-keyed) responses. No real network.

    python scripts/person_info_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import tmdb  # noqa: E402
from src.tmdb import (  # noqa: E402
    GET_PERSON_INFO_TOOL,
    execute_person_info_tool,
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


# --- HTTP stubbing -------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _HttpStub:
    """Routes (url, params) requests to canned payloads keyed on a URL
    substring. Records every call for inspection."""

    def __init__(self):
        self.routes: dict[str, object] = {}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, params=None, **_kwargs):
        self.calls.append((url, params or {}))
        for needle, payload in self.routes.items():
            if needle in url:
                if payload is None:
                    return None  # simulate transport failure
                return _FakeResponse(payload)
        return None


_orig_http = tmdb._http_get_with_retry
_orig_key_fn = tmdb._api_key


def install(stub):
    tmdb._http_get_with_retry = stub
    # Force a non-empty key so the executor doesn't short-circuit on
    # missing-key checks unrelated to what we're testing.
    tmdb._api_key = lambda: "TESTKEY"


def restore():
    tmdb._http_get_with_retry = _orig_http
    tmdb._api_key = _orig_key_fn


# --- Fixtures ------------------------------------------------------------

_PEDRO = {"id": 12835, "name": "Pedro Pascal", "popularity": 95.3}
_HOMONYM = {"id": 999, "name": "Pedro Pascal", "popularity": 0.5}

# A real-shaped credits payload with: cast credits (acting), crew credits
# (multiple departments — Writing, Directing, Production), TBA dates, and
# a title that someone wrote AND directed (dedup target in 'all' mode).
_PEDRO_CREDITS = {
    "cast": [
        {"id": 1, "title": "Strange Way of Life", "release_date": "2023-09-25",
         "media_type": "movie", "character": "Jake"},
        {"id": 2, "name": "The Last of Us", "first_air_date": "2023-01-15",
         "media_type": "tv", "character": "Joel"},
        {"id": 3, "name": "The Mandalorian", "first_air_date": "2019-11-12",
         "media_type": "tv", "character": "Din Djarin"},
        {"id": 4, "title": "Future Film", "release_date": "",
         "media_type": "movie", "character": "Lead"},  # TBA — sorts last
    ],
    "crew": [
        {"id": 10, "title": "Some Doc", "release_date": "2022-05-01",
         "media_type": "movie", "job": "Director",
         "department": "Directing"},
        {"id": 11, "title": "Some Doc", "release_date": "2022-05-01",
         "media_type": "movie", "job": "Writer",
         "department": "Writing"},  # same title — dedup test
        {"id": 12, "title": "Other Film", "release_date": "2021-01-01",
         "media_type": "movie", "job": "Producer",
         "department": "Production"},
    ],
}


# --- Schema --------------------------------------------------------------

print("\nGET_PERSON_INFO_TOOL schema:")
check("tool name", GET_PERSON_INFO_TOOL.get("name") == "get_person_info")
check("tool description mentions 'filmography' or 'been in'",
      "filmography" in GET_PERSON_INFO_TOOL["description"].lower()
      or "been in" in GET_PERSON_INFO_TOOL["description"].lower())
check("schema requires `name`",
      "name" in GET_PERSON_INFO_TOOL["input_schema"].get("required", []))
check("department enum includes acting/directing/writing/all",
      set(GET_PERSON_INFO_TOOL["input_schema"]["properties"]
          ["department"]["enum"])
      == {"acting", "directing", "writing", "all"})


# --- Missing API key -----------------------------------------------------

print("\nmissing API key:")
saved_key = os.environ.pop("TMDB_API_KEY", None)
restore()  # use the real _api_key (which reads env, now missing)
try:
    out = execute_person_info_tool({"name": "Pedro Pascal"})
    check("missing TMDB_API_KEY -> voice-friendly setup hint",
          "tmdb_api_key" in out.lower() or "api key" in out.lower())
finally:
    if saved_key is not None:
        os.environ["TMDB_API_KEY"] = saved_key


# --- Missing name --------------------------------------------------------

print("\nmissing name:")
stub = _HttpStub()
install(stub)
try:
    out = execute_person_info_tool({})
    check("missing name -> friendly message", "name is required" in out.lower())
    out = execute_person_info_tool({"name": "   "})
    check("blank name -> friendly message", "name is required" in out.lower())
finally:
    restore()


# --- Person not found ----------------------------------------------------

print("\nperson not found:")
stub = _HttpStub()
stub.routes["/search/person"] = {"results": []}
install(stub)
try:
    out = execute_person_info_tool({"name": "Nobody"})
    check("no search results -> 'couldn't find anyone' message",
          "couldn't find" in out.lower())
finally:
    restore()


# --- Search transport failure -------------------------------------------

print("\nsearch transport failure:")
stub = _HttpStub()
stub.routes["/search/person"] = None  # None = simulate failed request
install(stub)
try:
    out = execute_person_info_tool({"name": "Pedro Pascal"})
    check("search returned None -> friendly fail (no crash)",
          "couldn't find" in out.lower() or "unavailable" in out.lower())
finally:
    restore()


# --- Homonym disambiguation (popularity sort) ---------------------------

print("\nhomonym disambiguation:")
stub = _HttpStub()
# Stick the LESS popular candidate first to verify the executor sorts.
stub.routes["/search/person"] = {"results": [_HOMONYM, _PEDRO]}
stub.routes["/person/12835/combined_credits"] = _PEDRO_CREDITS
install(stub)
try:
    out = execute_person_info_tool({"name": "Pedro Pascal"})
    check("picks the higher-popularity match (id=12835)",
          "Joel" in out)  # canonical Pedro's role appears
    check("does NOT pick the homonym (id=999)",
          # if we'd picked 999, the credits-call URL would be /999/...
          # AND the canned route wouldn't match → response would say
          # "filmography unavailable". So the presence of acting credits
          # is the proof we hit the right id.
          "acting credits" in out.lower())
finally:
    restore()


# --- Acting filmography (default department) ----------------------------

print("\nacting filmography (default):")
stub = _HttpStub()
stub.routes["/search/person"] = {"results": [_PEDRO]}
stub.routes["/person/12835/combined_credits"] = _PEDRO_CREDITS
install(stub)
try:
    out = execute_person_info_tool({"name": "Pedro Pascal"})
    check("header includes canonical name + 'acting credits'",
          "Pedro Pascal" in out and "acting credits" in out.lower())
    check("'The Last of Us' (TV) appears with character",
          "The Last of Us" in out and "Joel" in out)
    check("'The Mandalorian' appears with character",
          "The Mandalorian" in out and "Din Djarin" in out)
    check("sorted by release date desc: TLoU (2023) before Mando (2019)",
          out.index("The Last of Us") < out.index("The Mandalorian"))
    check("TBA-dated 'Future Film' appears (sorted last among acting)",
          "Future Film" in out)
    check("crew credits ('Some Doc' director job) NOT in acting filter",
          "Some Doc" not in out)
finally:
    restore()


# --- Directing filter ----------------------------------------------------

print("\ndirecting filter:")
stub = _HttpStub()
stub.routes["/search/person"] = {"results": [_PEDRO]}
stub.routes["/person/12835/combined_credits"] = _PEDRO_CREDITS
install(stub)
try:
    out = execute_person_info_tool(
        {"name": "Pedro Pascal", "department": "directing"}
    )
    check("header mentions 'directing credits'",
          "directing credits" in out.lower())
    check("Some Doc (Director) appears",
          "Some Doc" in out and "Director" in out)
    check("acting credit 'Joel' NOT in directing filter",
          "Joel" not in out)
    check("Other Film (Producer) NOT in directing filter",
          "Other Film" not in out)
finally:
    restore()


# --- Writing filter ------------------------------------------------------

print("\nwriting filter:")
stub = _HttpStub()
stub.routes["/search/person"] = {"results": [_PEDRO]}
stub.routes["/person/12835/combined_credits"] = _PEDRO_CREDITS
install(stub)
try:
    out = execute_person_info_tool(
        {"name": "Pedro Pascal", "department": "writing"}
    )
    check("header mentions 'writing credits'",
          "writing credits" in out.lower())
    check("Some Doc (Writer) appears", "Some Doc" in out and "Writer" in out)
    check("acting credit 'Joel' NOT in writing filter",
          "Joel" not in out)
finally:
    restore()


# --- All mode + dedup ----------------------------------------------------

print("\nall mode:")
stub = _HttpStub()
stub.routes["/search/person"] = {"results": [_PEDRO]}
stub.routes["/person/12835/combined_credits"] = _PEDRO_CREDITS
install(stub)
try:
    out = execute_person_info_tool(
        {"name": "Pedro Pascal", "department": "all"}
    )
    check("all mode -> header just 'credits'",
          " credits:" in out)
    check("all mode includes both acting + crew (Joel + Some Doc)",
          "Joel" in out and "Some Doc" in out)
    # Some Doc is listed as Director AND Writer in the fixture. In `all`
    # mode we dedupe by (title, year, job) — so Director and Writer of
    # the same title appear as TWO entries (different jobs), which is
    # correct: they ARE distinct contributions. But the SAME title +
    # year + job wouldn't appear twice — that's the dedup contract.
    director_count = out.count("Some Doc")
    check("Some Doc appears twice (Director + Writer, different jobs)",
          director_count == 2)
finally:
    restore()


# --- Department fallback (unknown -> acting) ----------------------------

print("\ndepartment fallback:")
stub = _HttpStub()
stub.routes["/search/person"] = {"results": [_PEDRO]}
stub.routes["/person/12835/combined_credits"] = _PEDRO_CREDITS
install(stub)
try:
    out = execute_person_info_tool(
        {"name": "Pedro Pascal", "department": "nonsense"}
    )
    check("unknown department -> falls back to acting",
          "acting credits" in out.lower() and "Joel" in out)
finally:
    restore()


# --- Empty cast (e.g. directing-only person, asked for acting) ----------

print("\nempty-result-set messaging:")
stub = _HttpStub()
stub.routes["/search/person"] = {"results": [_PEDRO]}
stub.routes["/person/12835/combined_credits"] = {"cast": [], "crew": []}
install(stub)
try:
    out = execute_person_info_tool({"name": "Pedro Pascal"})
    check("no credits -> 'couldn't find any' friendly message",
          "couldn't find any" in out.lower())
finally:
    restore()


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
