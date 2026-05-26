"""Regression test for the knowledge_remember LLM tool path.

The voice-intent path for "remember this permanently: X" is regex-driven
(src/knowledge.py::_REMEMBER_INTENT_RE) and requires the permanence adverb
to disambiguate from episodic memory. The LLM tool path here closes that
gap: Claude can write the corpus on any natural "remember this for me"
phrasing without the adverb. This test verifies:

  - the tool schema is well-formed and exposes a `fact` field
  - an empty fact returns a voice-friendly error (no file written)
  - a real fact writes a dated .md to the configured corpus dir
  - the tool is registered + correctly NOT in _RESTRICTED_DENY (phone-
    allowed, consistent with set_reminder)

Uses JARVIS_KNOWLEDGE_DIR + a temp dir so the user's real corpus
(~/repos/jarvis-knowledge) isn't touched.

    python scripts/knowledge_remember_test.py     # exit 0 = pass
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# --- Test 1: schema is well-formed ----------------------------------------
from src.knowledge import (  # noqa: E402
    KNOWLEDGE_REMEMBER_TOOL, execute_knowledge_remember,
)
schema = KNOWLEDGE_REMEMBER_TOOL
check("KNOWLEDGE_REMEMBER_TOOL: name=knowledge_remember + has description",
      schema.get("name") == "knowledge_remember" and "description" in schema)
props = schema.get("input_schema", {}).get("properties", {})
check("schema exposes a `fact` field (string), required",
      "fact" in props
      and props["fact"].get("type") == "string"
      and "fact" in schema["input_schema"].get("required", []))


# --- Test 2: empty fact returns a voice-friendly error, writes nothing ----
with tempfile.TemporaryDirectory() as tmp:
    os.environ["JARVIS_KNOWLEDGE_DIR"] = tmp
    out = execute_knowledge_remember({"fact": ""})
    check("empty fact -> voice-friendly error",
          "nothing to remember" in out.lower())
    # No .md files should exist.
    md_files = list(Path(tmp).glob("*.md"))
    check("empty fact -> no .md file written", len(md_files) == 0)
    del os.environ["JARVIS_KNOWLEDGE_DIR"]


# --- Test 3: a real fact writes a dated .md file --------------------------
with tempfile.TemporaryDirectory() as tmp:
    os.environ["JARVIS_KNOWLEDGE_DIR"] = tmp
    out = execute_knowledge_remember(
        {"fact": "My four pets are Aria, Basil, Cosmo, and Delta."}
    )
    check("real fact -> voice-friendly confirmation",
          ("noted" in out.lower() or "saved" in out.lower()))
    md_files = list(Path(tmp).glob("*.md"))
    check("real fact -> exactly one .md file written",
          len(md_files) == 1)
    if md_files:
        body = md_files[0].read_text(encoding="utf-8")
        check("file body contains the pet names",
              "Aria" in body and "Delta" in body)
    del os.environ["JARVIS_KNOWLEDGE_DIR"]


# --- Test 4: registered in _CLIENT_TOOLS + NOT in _RESTRICTED_DENY --------
from src.llm import _CLIENT_TOOLS, _RESTRICTED_DENY  # noqa: E402
check("knowledge_remember registered in _CLIENT_TOOLS",
      "knowledge_remember" in _CLIENT_TOOLS)
check("knowledge_remember NOT in _RESTRICTED_DENY (phone may write)",
      "knowledge_remember" not in _RESTRICTED_DENY)


# --- summary --------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
