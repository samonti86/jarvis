"""Unit tests for system_control's Windows service verbs (M75 + the M40
restart_service regression).

Covers the shared `_do_service_action` helper that backs restart_service /
stop_service / start_service — the gates (regex validation, confirmation,
admin) and that each verb produces the right PowerShell cmdlet + wording.
This is the first dedicated test for system_control (the M40/M42 verbs shipped
without one — the "gate with no enforcing test" gap, closed here for the
service family).

Network-free and subprocess-free: we monkeypatch `system_control.subprocess.run`
with a fake that records args and returns canned results, and set
`system_control._IS_ADMIN` per test. No PowerShell ever runs.

    python scripts/system_control_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import system_control as sc  # noqa: E402
from src.system_control import (  # noqa: E402
    SYSTEM_CONTROL_TOOL,
    execute_system_control_tool,
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


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeRun:
    """Stand-in for subprocess.run: records calls, returns a canned result
    or raises a canned exception."""

    def __init__(self, result=None, exc=None):
        self.result = result if result is not None else _FakeCompleted()
        self.exc = exc
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result

    @property
    def last_script(self) -> str:
        """The PowerShell -Command string from the most recent call."""
        if not self.calls:
            return ""
        args = self.calls[-1][0]
        # ["powershell.exe","-NoProfile","-NonInteractive","-Command", <script>]
        return args[4] if len(args) >= 5 else ""


_orig_run = sc.subprocess.run
_orig_admin = sc._IS_ADMIN


def install(run_stub, admin=True):
    sc.subprocess.run = run_stub
    sc._IS_ADMIN = admin


def restore():
    sc.subprocess.run = _orig_run
    sc._IS_ADMIN = _orig_admin


def call(action, target="Spooler", confirmed=True):
    return execute_system_control_tool(
        {"action": action, "target": target, "confirmed": confirmed}
    )


# --- Schema --------------------------------------------------------------

print("\nschema:")
enum = SYSTEM_CONTROL_TOOL["input_schema"]["properties"]["action"]["enum"]
check("enum includes restart_service", "restart_service" in enum)
check("enum includes stop_service", "stop_service" in enum)
check("enum includes start_service", "start_service" in enum)
check("description mentions stop / start",
      "stop_service" in SYSTEM_CONTROL_TOOL["description"]
      and "start_service" in SYSTEM_CONTROL_TOOL["description"])
check("all three service verbs in _SERVICE_CMDLETS map",
      {"restart_service", "stop_service", "start_service"}
      == set(sc._SERVICE_CMDLETS.keys()))


# --- Confirmation gate (no subprocess should fire) ----------------------

print("\nconfirmation gate:")
for action, gerund in [("restart_service", "restarting"),
                       ("stop_service", "stopping"),
                       ("start_service", "starting")]:
    run = _FakeRun()
    install(run, admin=True)
    try:
        out = call(action, confirmed=False)
        check(f"{action}: unconfirmed -> asks for confirmation",
              "confirmation" in out.lower() and gerund in out.lower())
        check(f"{action}: unconfirmed -> NO subprocess fired",
              len(run.calls) == 0)
    finally:
        restore()


# --- Admin gate ----------------------------------------------------------

print("\nadmin gate:")
for action in ("restart_service", "stop_service", "start_service"):
    run = _FakeRun()
    install(run, admin=False)
    try:
        out = call(action, confirmed=True)
        check(f"{action}: not admin -> 'Administrator' message",
              "administrator" in out.lower())
        check(f"{action}: not admin -> NO subprocess fired",
              len(run.calls) == 0)
    finally:
        restore()


# --- Invalid / missing service name -------------------------------------

print("\nname validation:")
run = _FakeRun()
install(run, admin=True)
try:
    out = call("stop_service", target="", confirmed=True)
    check("empty name -> 'Service name required'",
          "service name required" in out.lower())
    out = call("stop_service", target="bad; rm -rf", confirmed=True)
    check("name with shell metachars -> 'invalid characters'",
          "invalid characters" in out.lower())
    check("invalid name -> NO subprocess fired", len(run.calls) == 0)
finally:
    restore()


# --- Success path + correct cmdlet/wording per verb ---------------------

print("\nsuccess path (cmdlet + wording per verb):")
cases = [
    ("restart_service", "Restart-Service", "Restarted"),
    ("stop_service",    "Stop-Service",    "Stopped"),
    ("start_service",   "Start-Service",   "Started"),
]
for action, cmdlet, past in cases:
    run = _FakeRun(_FakeCompleted(returncode=0, stdout=""))
    install(run, admin=True)
    try:
        out = call(action, target="Spooler", confirmed=True)
        check(f"{action}: success -> '{past} Spooler.'",
              out == f"{past} Spooler.")
        check(f"{action}: uses {cmdlet} in the PS script",
              cmdlet in run.last_script)
        check(f"{action}: no -Force in the PS script",
              "-Force" not in run.last_script)
        check(f"{action}: service name reaches the script",
              "Spooler" in run.last_script)
    finally:
        restore()


# --- Success path uses subprocess stdout when present -------------------

print("\nsuccess passthrough of stdout:")
run = _FakeRun(_FakeCompleted(returncode=0, stdout="Restarted Spooler."))
install(run, admin=True)
try:
    out = call("restart_service", target="Spooler", confirmed=True)
    check("returns PS stdout verbatim", out == "Restarted Spooler.")
finally:
    restore()


# --- Failure path (non-zero exit surfaces the error) --------------------

print("\nfailure path:")
run = _FakeRun(_FakeCompleted(
    returncode=1, stdout="",
    stderr="Cannot find any service with service name 'Nope'."))
install(run, admin=True)
try:
    out = call("stop_service", target="Nope", confirmed=True)
    check("non-zero exit -> 'Could not stop'", "could not stop" in out.lower())
    check("surfaces the SCM error message",
          "cannot find any service" in out.lower())
finally:
    restore()

# dependent-services failure (the no -Force consequence we document)
run = _FakeRun(_FakeCompleted(
    returncode=1, stdout="",
    stderr=("Cannot stop service 'X' because it has dependent services. "
            "It can only be stopped if the Force flag is set.")))
install(run, admin=True)
try:
    out = call("stop_service", target="X", confirmed=True)
    check("dependent-services error surfaced (no silent -Force)",
          "dependent services" in out.lower())
finally:
    restore()


# --- Timeout + spawn failure --------------------------------------------

print("\ntimeout + spawn failure:")
run = _FakeRun(exc=subprocess.TimeoutExpired(cmd="powershell", timeout=30))
install(run, admin=True)
try:
    out = call("start_service", target="Spooler", confirmed=True)
    check("timeout -> 'timed out' (no crash)", "timed out" in out.lower())
finally:
    restore()

run = _FakeRun(exc=OSError("powershell.exe not found"))
install(run, admin=True)
try:
    out = call("restart_service", target="Spooler", confirmed=True)
    check("spawn failure -> 'Could not run PowerShell' (no crash)",
          "could not run powershell" in out.lower())
finally:
    restore()


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
