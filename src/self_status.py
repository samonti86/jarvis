"""Self-status (M60) — Jarvis's own subsystem roll-call.

`status_report` is the broader companion to M56's `homelab_status`: where that
one reports the homelab (MEDIA-HOST / Plex / disk), this one reports JARVIS
HIMSELF — which subsystems are alive vs. degraded-silently, plus a count of
concerning lines in this session's `jarvis.log`. *"Are you healthy?"*,
*"status report"*, *"what's the state of your subsystems?"* all route here.

Subsystems degrade silently by design (the project-wide fail-soft contract):
the GPU STT server might be down, Plex MCP might never have connected, the
remote console might be off — and Jarvis just keeps running. That's correct
behaviour, but it means a quiet failure can persist for days without notice.
`status_report` makes the quiet state legible on demand.

Architecture: each subsystem registers a getter via `register(name, getter)`
during `main.py`'s startup. The tool iterates the registry in insertion
order, calls each getter, joins the lines. Same module-singleton pattern
M56 uses for its live monitor — no subsystem refs threaded through
`stream_response`, the tool just reads what main.py has registered.

Defensive contract: a getter that raises is logged and renders as a single
"name: error reading" line — never breaks the whole report. Same posture as
every other tool executor.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Callable

# Registry — name → getter() → ONE status line. Insertion order is preserved
# (Python 3.7+), which is the report's display order. Last-write-wins on a
# repeat register() — fine for idempotent construction (a re-init in tests
# or a re-arm cycle just refreshes the binding).
_REGISTRY: dict[str, Callable[[], str]] = {}


def register(name: str, getter: Callable[[], str]) -> None:
    """Register a subsystem with a getter that returns ONE line for the
    status report. Called from `main.py` at startup."""
    _REGISTRY[name] = getter


def reset_registry() -> None:
    """Clear the registry. For tests only — production never wants this."""
    _REGISTRY.clear()


# --- jarvis.log "concerning lines since session start" --------------------

def _log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    return base / "jarvis.log"


# Patterns observed in real Jarvis subsystem logs: "[homelab] poll failed:",
# "[security] MEMORY WATCHDOG TRIPPED:", "[acoustic] inference call failed",
# "exception" / "traceback" from python's logging. Deliberately NOT matching
# the bare word "error" — too noisy (matches "no error", "permission_error"
# in arg names, etc.). Word boundaries to skip substring matches.
_ERROR_PAT = re.compile(
    r"\b(failed|raised|tripped|traceback|panic|exception)\b",
    re.IGNORECASE,
)

_SESSION_MARKER = "--- Jarvis started"


def count_session_errors() -> tuple[int, str]:
    """Returns (count, context). Counts concerning-pattern lines in
    jarvis.log SINCE the most recent "--- Jarvis started …" session marker.
    If no marker is found (rare — fresh install or just-rotated log), falls
    back to the last 500 lines. The context string explains which window was
    counted, for the user-facing report ("since session start" /
    "in recent log tail")."""
    path = _log_path()
    if not path.exists():
        return (0, "no log file yet")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[status] log read failed: {exc}", file=sys.stderr)
        return (0, "log unreadable")
    lines = text.splitlines()
    start_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if _SESSION_MARKER in lines[i]:
            start_idx = i + 1
            break
    if start_idx == -1:
        start_idx = max(0, len(lines) - 500)
        context = "in recent log tail"
    else:
        context = "since session start"
    n = sum(1 for line in lines[start_idx:] if _ERROR_PAT.search(line))
    return (n, context)


def process_private_mb() -> float:
    """Process private (commit) bytes in MB. The M44.2 metric — the
    leak-growing number that the security-mode watchdog watches. Best-effort
    — psutil is a project dep, but if it ever fails we return 0.0 and let
    the caller's status line read 0."""
    try:
        import psutil  # noqa: PLC0415 — light dep, already a project dep
        mem = psutil.Process().memory_info()
        # On Windows, .private == private bytes (commit). On Linux, fall
        # back to .rss (close enough for this status line).
        raw = getattr(mem, "private", None) or mem.rss
        return raw / (1024 * 1024)
    except Exception as exc:  # noqa: BLE001 — best-effort
        print(f"[status] memory read failed: {exc}", file=sys.stderr)
        return 0.0


# --- The Anthropic tool ---------------------------------------------------

STATUS_REPORT_TOOL = {
    "name": "status_report",
    "description": (
        "Jarvis's own subsystem health roll-call. Use it for 'Jarvis, "
        "status report', 'are you healthy?', 'what's the state of your "
        "subsystems?', 'is everything alive?'. Returns one line per "
        "subsystem (security, acoustic awareness, homelab monitor, Plex "
        "MCP, Plex laptop SSH, remote console, STT backend, reminders, "
        "process memory) plus a count of concerning lines in this "
        "session's log. Read the result back CONCISELY — if everything "
        "is healthy say so in a single sentence ('all systems nominal, "
        "sir'); only enumerate problems unless the user explicitly asked "
        "for a full report. M56's homelab_status is the homelab-only "
        "view; this is the broader in-process roll-call."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


def execute_status_report(params: dict) -> str:  # noqa: ARG001 — param-less
    """Iterate the registry; gather one line per subsystem. Never raises —
    a getter that raises renders as 'name: error reading' and the report
    continues."""
    if not _REGISTRY:
        return ("No subsystems registered for status reporting, sir — "
                "this is unexpected; logs may help.")
    lines: list[str] = []
    for name, getter in _REGISTRY.items():
        try:
            lines.append(getter())
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[status] '{name}' getter raised: {exc}", file=sys.stderr)
            lines.append(f"{name}: error reading ({type(exc).__name__})")
    return "\n".join(lines)
