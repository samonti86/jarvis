"""M60 — regression test for src/self_status.py.

Exercises the registry surface (register / reset / execute_status_report),
defensive behaviour (a getter that raises produces an 'error reading' line,
the report keeps going), and `count_session_errors` (session-marker scan +
fallback to the recent-tail window when no marker is found).

Same instrument discipline as tests/homelab_alert_test.py and
tests/acoustic_alert_test.py.

    python tests/status_report_test.py     # exit 0 = pass
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure the log path resolves to a temp dir BEFORE importing self_status —
# count_session_errors reads %LOCALAPPDATA%\Jarvis\jarvis.log at call time so
# the env override doesn't even need to be in place at import; we use a
# context-manager pattern below to point it where we want for each test.

from src import self_status  # noqa: E402

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


# --- Test 1: empty registry returns a clear message -----------------------
self_status.reset_registry()
out = self_status.execute_status_report({})
check("empty registry -> 'no subsystems registered' message",
      "No subsystems" in out)


# --- Test 2: registered getters render in insertion order -----------------
self_status.reset_registry()
self_status.register("Alpha", lambda: "Alpha: ok")
self_status.register("Beta", lambda: "Beta: ok")
self_status.register("Gamma", lambda: "Gamma: ok")
out = self_status.execute_status_report({})
check("3 registrations -> 3 lines in insertion order",
      out == "Alpha: ok\nBeta: ok\nGamma: ok")


# --- Test 3: a raising getter renders as 'error reading' and the rest of
# the report continues -----------------------------------------------------
self_status.reset_registry()
self_status.register("First", lambda: "First: ok")
def _explodes() -> str:
    raise RuntimeError("boom")
self_status.register("Boom", _explodes)
self_status.register("Last", lambda: "Last: ok")
out = self_status.execute_status_report({})
check("raising getter -> 'error reading' line, report continues",
      "First: ok" in out and "Boom: error reading" in out and "Last: ok" in out)


# --- Test 4: count_session_errors — session marker found ------------------
with tempfile.TemporaryDirectory() as tmp:
    log_dir = Path(tmp) / "Jarvis"
    log_dir.mkdir()
    log_file = log_dir / "jarvis.log"
    log_file.write_text(
        "--- Jarvis started 2026-05-26T09:00:00 ---\n"
        "[plex-mcp] connected\n"
        "[homelab] poll failed: timeout\n"
        "[acoustic] inference call failed\n"
        "--- Jarvis started 2026-05-26T10:00:00 ---\n"     # newer session
        "[plex-mcp] connected\n"
        "[security] MEMORY WATCHDOG TRIPPED\n"
        "[news] some traceback below\n"
        "  Exception details here\n"
        "normal line, not an error\n",
        encoding="utf-8",
    )
    # Point self_status at our temp log
    old = os.environ.get("LOCALAPPDATA")
    os.environ["LOCALAPPDATA"] = tmp
    try:
        n, ctx = self_status.count_session_errors()
    finally:
        if old is None:
            del os.environ["LOCALAPPDATA"]
        else:
            os.environ["LOCALAPPDATA"] = old
    # Should count lines AFTER the NEWER session marker: tripped, traceback,
    # exception -> 3. The older session's "failed" lines are excluded.
    check("session-marker scan counts only since most-recent marker",
          n == 3 and ctx == "since session start")


# --- Test 5: count_session_errors — no marker, fallback to recent tail ---
with tempfile.TemporaryDirectory() as tmp:
    log_dir = Path(tmp) / "Jarvis"
    log_dir.mkdir()
    (log_dir / "jarvis.log").write_text(
        "ordinary log line 1\n"
        "[homelab] poll failed: timeout\n"
        "another ordinary line\n"
        "some thing raised an issue\n",
        encoding="utf-8",
    )
    old = os.environ.get("LOCALAPPDATA")
    os.environ["LOCALAPPDATA"] = tmp
    try:
        n, ctx = self_status.count_session_errors()
    finally:
        if old is None:
            del os.environ["LOCALAPPDATA"]
        else:
            os.environ["LOCALAPPDATA"] = old
    check("no session marker -> fallback to recent-tail count",
          n == 2 and ctx == "in recent log tail")


# --- Test 6: count_session_errors — missing log file is fail-soft ---------
with tempfile.TemporaryDirectory() as tmp:
    # Don't create the log file; just point at the empty dir.
    old = os.environ.get("LOCALAPPDATA")
    os.environ["LOCALAPPDATA"] = tmp
    try:
        n, ctx = self_status.count_session_errors()
    finally:
        if old is None:
            del os.environ["LOCALAPPDATA"]
        else:
            os.environ["LOCALAPPDATA"] = old
    check("missing log file -> (0, 'no log file yet')",
          n == 0 and ctx == "no log file yet")


# --- Test 7: tool schema is well-formed ----------------------------------
schema = self_status.STATUS_REPORT_TOOL
check("STATUS_REPORT_TOOL has name + description + input_schema",
      schema.get("name") == "status_report"
      and "description" in schema
      and schema.get("input_schema", {}).get("type") == "object")


# --- summary --------------------------------------------------------------
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
