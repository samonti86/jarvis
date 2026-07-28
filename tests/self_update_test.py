"""M64 — unit tests for the self-update tool.

The hard parts of M64 — git pull, fast-forward enforcement, working-tree
dirty-detection — aren't tested with a real `git pull` (that would touch
the origin remote, mutate the actual project repo, and depend on network).
Instead we monkeypatch `_run_git` to feed synthetic results and assert the
tool's branching matches the contract:

  - no `confirm` → describe + ask (NO subprocess at all)
  - confirm + dirty tree → refuse with the porcelain status
  - confirm + clean + "Already up to date" → no restart, no callback fire
  - confirm + clean + real pull + restart callback unset → honest "can't
    restart" message
  - confirm + clean + real pull + restart callback set → fires the
    callback (after the configured delay)
  - subprocess error → voice-friendly error, no restart

Same instrument discipline as M62.2 / M63 — exercise the pure decision
points in isolation; live `git pull` is the user's manual test.

    python tests/self_update_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import self_update  # noqa: E402
from src.self_update import (  # noqa: E402
    UPDATE_JARVIS_TOOL,
    execute_update_jarvis,
    register_restart_callback,
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


# --- monkeypatch helpers -------------------------------------------------

class _GitStub:
    """Stub for src.self_update._run_git that returns canned results
    indexed by the command's first arg ('status', 'pull', ...). Stores the
    call log so tests can assert what got invoked."""

    def __init__(self) -> None:
        self.responses: dict[str, tuple[int, str, str]] = {}
        self.calls: list[list[str]] = []

    def __call__(self, args, timeout=60.0):
        self.calls.append(list(args))
        return self.responses.get(args[0], (0, "", ""))


_orig_run_git = self_update._run_git
_orig_delay = self_update._RESTART_DELAY_SECONDS


def install_stub() -> _GitStub:
    stub = _GitStub()
    self_update._run_git = stub
    # Slash the restart delay so the test doesn't take 5 seconds per fire.
    self_update._RESTART_DELAY_SECONDS = 0.05
    return stub


def restore() -> None:
    self_update._run_git = _orig_run_git
    self_update._RESTART_DELAY_SECONDS = _orig_delay


# --- UPDATE_JARVIS_TOOL schema -------------------------------------------

print("\nUPDATE_JARVIS_TOOL schema:")
check("tool name", UPDATE_JARVIS_TOOL.get("name") == "update_jarvis")
check("tool description mentions 'update'",
      "update" in UPDATE_JARVIS_TOOL["description"].lower())
check("schema has `confirm` property",
      "confirm" in UPDATE_JARVIS_TOOL["input_schema"]["properties"])
check("`confirm` is boolean type",
      UPDATE_JARVIS_TOOL["input_schema"]["properties"]["confirm"]["type"] == "boolean")


# --- no confirm → describe path (M64.1 — now PREVIEWS pending commits) ---

print("\nno confirm + pending commits visible (M64.1 preview):")
stub = install_stub()
try:
    stub.responses["fetch"] = (0, "", "")
    stub.responses["log"] = (0, "abc1234 M59 — scheduled briefing\n"
                                "def5678 fix calendar bug\n"
                                "9012345 tighten weather geocode", "")
    out = execute_update_jarvis({})
    check("describe path called fetch + log (no pull, no status)",
          [c[0] for c in stub.calls] == ["fetch", "log"])
    check("describe lists the commit count",
          "3 new commits" in out)
    check("describe lists the first commit subject",
          "scheduled briefing" in out)
    check("describe lists the third commit subject",
          "weather geocode" in out)
    check("describe asks for confirmation",
          "shall i" in out.lower() or "proceed" in out.lower())
finally:
    restore()


print("\nno confirm + nothing pending (M64.1 short-circuit):")
stub = install_stub()
try:
    stub.responses["fetch"] = (0, "", "")
    stub.responses["log"] = (0, "", "")  # empty = nothing pending
    out = execute_update_jarvis({})
    check("nothing pending -> 'already up to date' short-circuit",
          "up to date" in out.lower())
    check("nothing pending -> no `shall I proceed` (no need to confirm)",
          "shall i" not in out.lower() and "proceed" not in out.lower())
finally:
    restore()


print("\nno confirm + fetch fails (M64.1 fallback):")
stub = install_stub()
try:
    stub.responses["fetch"] = (1, "", "fatal: unable to access github.com")
    out = execute_update_jarvis({})
    check("fetch fails -> falls back to generic describe",
          "shall i" in out.lower() or "proceed" in out.lower())
    check("fetch fails -> mentions the network problem",
          "couldn't reach" in out.lower() or "couldn't" in out.lower())
    check("fetch fails -> log NOT called (no point)",
          [c[0] for c in stub.calls] == ["fetch"])
finally:
    restore()


print("\nno confirm + log fails (M64.1 fallback):")
stub = install_stub()
try:
    stub.responses["fetch"] = (0, "", "")
    stub.responses["log"] = (128, "", "fatal: HEAD does not point to a branch")
    out = execute_update_jarvis({})
    check("log fails -> falls back to generic describe",
          "shall i" in out.lower() or "proceed" in out.lower())
    check("log fails -> surfaces git error in fallback",
          "couldn't preview" in out.lower() or "preview" in out.lower())
finally:
    restore()


print("\nconfirm=False explicit (same path as no confirm):")
stub = install_stub()
try:
    stub.responses["fetch"] = (0, "", "")
    stub.responses["log"] = (0, "abc1234 a commit", "")
    out_a = execute_update_jarvis({})
    stub.calls.clear()
    out_b = execute_update_jarvis({"confirm": False})
    check("confirm=False -> same describe path as missing confirm",
          out_a == out_b)
finally:
    restore()


# --- confirm + dirty tree --------------------------------------------------

print("\nconfirm + dirty tree:")
stub = install_stub()
try:
    stub.responses["status"] = (0, " M src/foo.py\n?? extra.txt", "")
    out = execute_update_jarvis({"confirm": True})
    check("dirty tree -> refuses with mention of uncommitted",
          "uncommitted" in out.lower())
    check("dirty tree -> porcelain detail surfaced",
          "src/foo.py" in out)
    check("dirty tree -> only `status` was called, no pull",
          [c[0] for c in stub.calls] == ["status"])
finally:
    restore()


# --- confirm + clean + already up to date --------------------------------

print("\nconfirm + clean + already-up-to-date:")
stub = install_stub()
try:
    stub.responses["status"] = (0, "", "")  # clean
    stub.responses["pull"] = (0, "Already up to date.", "")

    fire_count = [0]
    def cb():
        fire_count[0] += 1
    register_restart_callback(cb)

    out = execute_update_jarvis({"confirm": True})
    check("already up to date -> says so", "up to date" in out.lower())
    check("already up to date -> no restart callback fired",
          fire_count[0] == 0)
    # let any scheduled (none) thread settle
    register_restart_callback(None)  # type: ignore[arg-type]
finally:
    restore()


# --- confirm + clean + real update + no callback registered --------------

print("\nconfirm + clean + real update, no restart wired:")
stub = install_stub()
try:
    stub.responses["status"] = (0, "", "")
    stub.responses["pull"] = (0,
        "Updating abc1234..def5678\nFast-forward\n 2 files changed",
        "")
    self_update._restart_callback = None  # ensure not wired

    out = execute_update_jarvis({"confirm": True})
    check("real update, no callback -> 'not wired to restart' message",
          "not wired" in out.lower() or "restart myself" in out.lower())
finally:
    restore()


# --- confirm + clean + real update + callback fires ---------------------

print("\nconfirm + clean + real update, callback fires:")
stub = install_stub()
try:
    stub.responses["status"] = (0, "", "")
    stub.responses["pull"] = (0,
        "Updating abc1234..def5678\nFast-forward\n 2 files changed",
        "")

    fire_event = threading.Event()
    fire_count = [0]

    def cb():
        fire_count[0] += 1
        fire_event.set()

    register_restart_callback(cb)

    out = execute_update_jarvis({"confirm": True})
    check("real update -> success message", "successful" in out.lower())
    check("real update -> 'restart in a moment' phrasing",
          "restart" in out.lower())
    # Callback fires on a delayed daemon thread (50ms in test) — wait for it.
    fired = fire_event.wait(timeout=2.0)
    check("real update -> restart callback eventually fires", fired)
    check("real update -> callback fires exactly once", fire_count[0] == 1)
finally:
    self_update._restart_callback = None
    restore()


# --- pull fails (non-zero rc) -------------------------------------------

print("\nconfirm + pull fails:")
stub = install_stub()
try:
    stub.responses["status"] = (0, "", "")
    stub.responses["pull"] = (1, "", "fatal: Could not resolve host github.com")

    fire_count = [0]
    def cb():
        fire_count[0] += 1
    register_restart_callback(cb)

    out = execute_update_jarvis({"confirm": True})
    check("pull error -> voice-friendly failure message",
          "failed" in out.lower())
    check("pull error -> stderr leaked into message",
          "github.com" in out)
    check("pull error -> no restart callback fired",
          fire_count[0] == 0)
finally:
    self_update._restart_callback = None
    restore()


# --- git binary missing --------------------------------------------------

print("\nconfirm + git binary missing:")
stub = install_stub()
try:
    stub.responses["status"] = (-1, "", "git executable not found in PATH")
    fire_count = [0]
    def cb():
        fire_count[0] += 1
    register_restart_callback(cb)

    out = execute_update_jarvis({"confirm": True})
    check("git missing -> graceful 'couldn't read git status' message",
          "couldn't" in out.lower() or "git status" in out.lower()
          or "not found" in out.lower())
    check("git missing -> no restart fired", fire_count[0] == 0)
finally:
    self_update._restart_callback = None
    restore()


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
