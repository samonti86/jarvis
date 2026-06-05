"""Unit tests for the get_game_length tool (How Long To Beat).

Network-free: we stub `game_length._get_client` to return a fake HLTB client
whose `.search()` returns canned entries (or raises), so nothing touches the
real site. The hard parts under test:
  - _fmt_hours rounding (nearest half-hour) + zero/missing suppression
  - playstyle-line suppression (omit a style HLTB has no data for)
  - all-styles fallback when the breakdown is entirely empty
  - best-match selection by similarity across multiple candidates
  - never-raises contract: missing name, package unavailable, no results,
    search exception all become voice-friendly strings.

    python scripts/game_length_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import game_length  # noqa: E402
from src.game_length import (  # noqa: E402
    GAME_LENGTH_TOOL,
    execute_game_length_tool,
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


# --- Fakes ---------------------------------------------------------------


class _FakeEntry:
    def __init__(self, game_name, similarity, main_story=0, main_extra=0,
                 completionist=0, all_styles=0):
        self.game_name = game_name
        self.similarity = similarity
        self.main_story = main_story
        self.main_extra = main_extra
        self.completionist = completionist
        self.all_styles = all_styles


class _FakeClient:
    def __init__(self, results=None, exc=None):
        self._results = results
        self._exc = exc
        self.searched = []

    def search(self, name):
        self.searched.append(name)
        if self._exc is not None:
            raise self._exc
        return self._results


_orig_get_client = game_length._get_client


def install(client_or_none):
    game_length._get_client = lambda: client_or_none


def restore():
    game_length._get_client = _orig_get_client


# --- Schema --------------------------------------------------------------

print("\nGAME_LENGTH_TOOL schema:")
check("tool name", GAME_LENGTH_TOOL.get("name") == "get_game_length")
check("requires `name`",
      "name" in GAME_LENGTH_TOOL["input_schema"].get("required", []))
check("description mentions 'beat' or 'how long'",
      "beat" in GAME_LENGTH_TOOL["description"].lower()
      or "how long" in GAME_LENGTH_TOOL["description"].lower())


# --- _fmt_hours rounding -------------------------------------------------

print("\n_fmt_hours:")
check("60.03 -> '60 hours'", game_length._fmt_hours(60.03) == "60 hours")
check("26.95 -> '27 hours' (rounds up)",
      game_length._fmt_hours(26.95) == "27 hours")
check("1.0 -> '1 hour' (singular)", game_length._fmt_hours(1.0) == "1 hour")
check("0.5 -> '0.5 hours' (half kept)",
      game_length._fmt_hours(0.5) == "0.5 hours")
check("0 -> None (suppressed)", game_length._fmt_hours(0) is None)
check("negative -> None", game_length._fmt_hours(-3) is None)
check("None -> None", game_length._fmt_hours(None) is None)
check("garbage str -> None", game_length._fmt_hours("n/a") is None)


# --- Missing name --------------------------------------------------------

print("\nmissing name:")
out = execute_game_length_tool({})
check("no name -> 'title is required'", "title is required" in out.lower())
out = execute_game_length_tool({"name": "   "})
check("blank name -> 'title is required'", "title is required" in out.lower())


# --- Package unavailable -------------------------------------------------

print("\npackage unavailable:")
install(None)
try:
    out = execute_game_length_tool({"name": "Elden Ring"})
    check("client None -> 'not installed' hint",
          "isn't installed" in out.lower() or "howlongtobeatpy" in out.lower())
finally:
    restore()


# --- No results ----------------------------------------------------------

print("\nno results:")
install(_FakeClient(results=[]))
try:
    out = execute_game_length_tool({"name": "Zzxqwv Nonexistent"})
    check("empty results -> 'couldn't find'", "couldn't find" in out.lower())
finally:
    restore()

# also: search returning None (some HLTB versions do)
install(_FakeClient(results=None))
try:
    out = execute_game_length_tool({"name": "Whatever"})
    check("None results -> 'couldn't find'", "couldn't find" in out.lower())
finally:
    restore()


# --- Search raises -------------------------------------------------------

print("\nsearch exception:")
install(_FakeClient(exc=RuntimeError("HLTB changed their HTML")))
try:
    out = execute_game_length_tool({"name": "Elden Ring"})
    check("search raises -> 'lookup failed' (no crash)",
          "lookup failed" in out.lower())
finally:
    restore()


# --- Happy path (full breakdown) ----------------------------------------

print("\nhappy path:")
install(_FakeClient(results=[
    _FakeEntry("Elden Ring", 1.0, main_story=60.03, main_extra=101.2,
               completionist=135.67, all_styles=105.44),
]))
try:
    out = execute_game_length_tool({"name": "Elden Ring"})
    check("header names the matched game", "Elden Ring" in out)
    check("Main story: 60 hours", "Main story: 60 hours" in out)
    check("Main + extras: 101 hours", "Main + extras: 101 hours" in out)
    check("Completionist: 135.5 hours (135.67 -> nearest half)",
          "Completionist: 135.5 hours" in out)
    check("read order: story before extras before completionist",
          out.index("Main story") < out.index("Main + extras")
          < out.index("Completionist"))
finally:
    restore()


# --- Suppress a zero playstyle ------------------------------------------

print("\nsuppress zero playstyle:")
install(_FakeClient(results=[
    _FakeEntry("Some Indie", 0.9, main_story=8.0, main_extra=0,
               completionist=12.0),
]))
try:
    out = execute_game_length_tool({"name": "Some Indie"})
    check("main_extra=0 line omitted", "Main + extras" not in out)
    check("main_story present", "Main story: 8 hours" in out)
    check("completionist present", "Completionist: 12 hours" in out)
finally:
    restore()


# --- all-styles fallback (breakdown empty) ------------------------------

print("\nall-styles fallback:")
install(_FakeClient(results=[
    _FakeEntry("Just Announced", 0.95, main_story=0, main_extra=0,
               completionist=0, all_styles=20.0),
]))
try:
    out = execute_game_length_tool({"name": "Just Announced"})
    check("falls back to all-playstyles line",
          "All playstyles: 20 hours" in out)
finally:
    restore()


# --- no data at all ------------------------------------------------------

print("\nno data at all:")
install(_FakeClient(results=[
    _FakeEntry("Mystery Game", 0.95),  # all zero
]))
try:
    out = execute_game_length_tool({"name": "Mystery Game"})
    check("all-zero -> 'no completion-time data yet'",
          "no completion-time data" in out.lower())
finally:
    restore()


# --- best-match selection by similarity ---------------------------------

print("\nbest-match selection:")
install(_FakeClient(results=[
    _FakeEntry("Wrong Game", 0.30, main_story=5.0),
    _FakeEntry("Right Game", 0.98, main_story=42.0),
    _FakeEntry("Also Wrong", 0.50, main_story=7.0),
]))
try:
    out = execute_game_length_tool({"name": "Right Game"})
    check("picks highest-similarity entry (Right Game)",
          "Right Game" in out and "42 hours" in out)
    check("does not pick a lower-similarity entry",
          "Wrong Game" not in out and "Also Wrong" not in out)
finally:
    restore()


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
