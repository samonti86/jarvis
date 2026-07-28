"""M80 — tests for the per-turn speaker-context block (identity-aware replies).

The load-bearing core is `src.llm._speaker_context_block` — the tiny note that
tells Claude WHO it's currently speaking with (from M69 voice identification),
appended as a SECOND, un-cache_controlled system block so a speaker handoff
between turns never invalidates the cached prompt prefix. These tests pin its
formatting contract (no network, no model):
  - identity line present + name interpolated correctly (incl. apostrophes)
  - empty / whitespace name -> '' (caller skips the block)
  - English speaker -> NO language line (en is the default; keep it tiny)
  - non-English speaker -> a SOFT, subordinate language lean
  - handoff wording present so an alternating-speaker turn is unambiguous

Plus an assembly check: the block is added to system_param WITHOUT a
cache_control key (the cached prefix stays the big block above it).

    python tests/speaker_context_test.py   # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import _speaker_context_block  # noqa: E402

_passed = 0
_failed = 0


def check(desc: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok: {desc}")
    else:
        _failed += 1
        print(f"FAIL: {desc}")


# --- Identity line ---------------------------------------------------------
b = _speaker_context_block("Alice", "en")
check("names the speaker", "speaking with Alice" in b)
check("identified-by-voice phrasing", "identified by voice" in b)
check("instructs name use, sparingly", "by name" in b and "sparingly" in b)
check("handoff wording present", "handoff" in b.lower())
check("name interpolated into handoff (no literal {name})", "{name}" not in b)
check("handoff addresses the right person", "respond to Alice now" in b)

# --- English speaker: no language line -------------------------------------
check("en speaker -> no Spanish lean", "Spanish" not in b)
check("en speaker -> no 'usually speaks' line", "usually speaks" not in b)

# --- Spanish speaker: soft, subordinate lean -------------------------------
es = _speaker_context_block("Bob", "es")
check("es speaker -> names Spanish", "Spanish" in es)
check("es speaker -> 'usually speaks Spanish'", "Bob usually speaks Spanish" in es)
check("es lean is conditional/soft (ambiguous -> lean)",
      "ambiguous" in es and "lean toward Spanish" in es)
check("es block still carries the identity half", "speaking with Bob" in es)

# --- Empty / whitespace name -> skip ----------------------------------------
check("empty name -> ''", _speaker_context_block("", "en") == "")
check("whitespace name -> ''", _speaker_context_block("   ", "es") == "")
check("None name -> '' (defensive)", _speaker_context_block(None, "en") == "")

# --- Name hygiene ----------------------------------------------------------
check("name is stripped", _speaker_context_block("  Saul  ", "en").startswith(
    "You are currently speaking with Saul,"))
ap = _speaker_context_block("D'Angelo", "en")
check("apostrophe name survives", "speaking with D'Angelo" in ap)

# --- Unknown language code: falls back to the raw code, still soft ----------
fr = _speaker_context_block("Pierre", "fr")
check("unknown lang code -> still adds a soft lean", "usually speaks fr" in fr
      or "usually speaks French" in fr)
check("unknown lang is NOT treated as English (line present)",
      "usually speaks" in fr)

# --- Assembly: appended WITHOUT cache_control ------------------------------
# Mirror what stream_response does, and assert the cache breakpoint stays on
# the big block (the whole point of the second-block design).
system_param = [{"type": "text", "text": "BIG PROMPT",
                 "cache_control": {"type": "ephemeral"}}]
blk = _speaker_context_block("Alice", "en")
if blk:
    system_param.append({"type": "text", "text": blk})
check("two system blocks when a speaker is known", len(system_param) == 2)
check("cached prefix block keeps cache_control",
      "cache_control" in system_param[0])
check("speaker block has NO cache_control (re-read each turn)",
      "cache_control" not in system_param[1])

# No speaker -> no second block (remote/unrecognized turns).
system_param2 = [{"type": "text", "text": "BIG PROMPT",
                  "cache_control": {"type": "ephemeral"}}]
blk2 = _speaker_context_block("", "en")
if blk2:
    system_param2.append({"type": "text", "text": blk2})
check("no speaker -> single (cached) system block", len(system_param2) == 1)


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
