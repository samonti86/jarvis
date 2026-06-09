"""Destructive actions on the Plex laptop (M27).

Companion to M24's read-only diagnostic tools. Same SSH transport
(`PlexLaptopClient`), same `-EncodedCommand` PowerShell shipping pattern,
but here the operations modify state — restart the Plex server, refresh
a library, empty Plex's trash.

Architecture in three observations:

1. **Single tool, action enum.** Resisted the temptation to make three
   separate tools (`plex_restart`, `plex_library_refresh`, `plex_empty_trash`)
   because they share the same risk profile (destructive, confirmation-gated)
   and the same dispatch path. One tool with an `action` enum means one
   confirmation flow + one telemetry marker, and adding a fourth action
   later is one enum entry instead of three new files.

2. **Server-side `confirmed=true` gate.** Mirrors M23's `kill_process`
   pattern. The tool itself rejects calls where `confirmed` is missing or
   false, returning a "needs confirmation" string instead of firing. Means
   a misbehaving prompt, a hallucinated tool call, or a future model
   regression cannot bypass user consent — the system prompt instructs
   Claude to confirm; the tool enforces it.

3. **Plex token shipped inside the encoded command, not via env.** The
   refresh/empty_trash actions hit Plex's HTTP API at localhost:32400
   which requires `X-Plex-Token`. We embed the token in the PowerShell
   payload, base64-encode it as part of `-EncodedCommand`, and ship over
   SSH (encrypted in transit). Token exists transiently in the laptop's
   PowerShell process memory during execution — the same trust boundary
   as having Plex itself store the token in its data folder. No setup
   step required on the laptop; the only place the token lives at rest
   is on the PC's `.env`.

Defensive contract: same as the rest of the toolkit. Connection failures,
HTTP errors, missing parameters all become readable strings. Never raises.
"""

from __future__ import annotations

import os
from typing import Callable

from src.plex_laptop import PlexLaptopClient, _ps_encoded


# Default Plex install path on the laptop. Could be made configurable via
# env var if a future deployment uses a non-default location, but every
# Windows Plex install since 2017 puts it here.
_PLEX_EXE_PATH = r"C:\Program Files\Plex\Plex Media Server\Plex Media Server.exe"

# Plex's local HTTP API. Port 32400 is fixed by Plex; localhost-bound from
# the laptop side means we don't need any external network rules.
_PLEX_API_BASE = "http://localhost:32400"


# --- Anthropic tool definition --------------------------------------------

PLEX_ACTION_TOOL = {
    "name": "plex_action",
    "description": (
        "Perform a DESTRUCTIVE action on the Plex laptop: restart the Plex "
        "Media Server, refresh a library (re-scan files), or empty Plex's "
        "trash for a library. ALWAYS requires explicit user confirmation — "
        "ask in plain language first ('Confirm: restart Plex?'), wait for "
        "an explicit yes, THEN call this tool with confirmed=true. The tool "
        "rejects calls with confirmed=false or omitted as a safety gate. "
        "For diagnostic-only queries use plex_logs_tail / plex_logs_search "
        "/ plex_laptop_health instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "restart_plex_server",
                    "refresh_library",
                    "empty_trash",
                ],
                "description": (
                    "restart_plex_server = restart the Plex Media Server "
                    "process on the laptop (kill + relaunch); "
                    "refresh_library = ask Plex to re-scan files in a "
                    "specific library so newly-added/changed media is "
                    "picked up (requires library_id); "
                    "empty_trash = permanently delete Plex's trashed "
                    "metadata for a library — irreversible (requires "
                    "library_id)."
                ),
            },
            "library_id": {
                "type": "integer",
                "description": (
                    "Required for refresh_library and empty_trash. The Plex "
                    "library section ID (e.g., 1 for Movies, 2 for TV "
                    "Shows). If you don't know the ID, call the Plex MCP "
                    "tool library_list first to enumerate sections."
                ),
            },
            "confirmed": {
                "type": "boolean",
                "description": (
                    "MUST be true for the action to actually fire. Set to "
                    "true ONLY after the user has confirmed the destructive "
                    "action in plain language. The tool returns a "
                    "confirmation-required notice if false or omitted."
                ),
            },
        },
        "required": ["action"],
    },
}


# --- Action implementations -----------------------------------------------


def _do_restart_plex(client: PlexLaptopClient) -> str:
    """Restart Plex Media Server on the laptop.

    Probes for a Windows service first (some users install the open-source
    PlexService wrapper which runs Plex as a real service). If no service
    exists, falls back to process kill + relaunch via `cmd /c start`, THEN
    verifies the relaunch survived by re-checking `Get-Process` after a
    settle delay.

    Why verification matters (real bug caught in M27 testing): Plex Media
    Server in tray-app mode expects an interactive user session — HKCU
    registry, the desktop, etc. An SSH-spawned launch lives in Session 0
    (the non-interactive system context) and the new process can either
    exit immediately or fail to fully initialize. Without verification,
    Jarvis would report a successful restart even when Plex was actually
    just dead — and the user has to manually relaunch.

    With verification, if Plex doesn't come back up we return an honest
    error message explaining the Session-0 limitation and the workarounds
    (interactive login, or install PlexService for true service-managed
    restarts).
    """
    # Compact PowerShell — emits short status codes that Python translates
    # into user-facing prose. The previous "verbose error in PowerShell"
    # version pushed the base64-encoded command past Windows' 8191-char
    # cmd.exe limit and got rejected with "command line too long".
    script = f"""
# Probe known PMS wrapper service names. cjmurph/PmsService registers as
# 'PlexService'; some forks/older installers use other names. Hardcoded
# 'Plex Media Server' was the original assumption (e.g., custom service
# registration via sc.exe), kept as a fallback. We pick the FIRST running
# match — Update / Tuner services are intentionally not in this list.
$svc = $null
foreach ($n in 'PlexService', 'Plex Media Server', 'PlexMediaServerService', 'PmsService') {{
    $candidate = Get-Service -Name $n -ErrorAction SilentlyContinue
    if ($candidate) {{ $svc = $candidate; break }}
}}
if ($svc) {{
    Restart-Service -Name $svc.Name -Force
    $mode = 'service'
    $pid_ = 0
}} else {{
    Stop-Process -Name 'Plex Media Server' -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    & cmd /c 'start "" "{_PLEX_EXE_PATH}"'
    Start-Sleep -Seconds 5
    $alive = Get-Process -Name 'Plex Media Server' -ErrorAction SilentlyContinue
    if (-not $alive) {{ Write-Output 'STATUS:FAIL_PROCESS_DEAD'; exit 0 }}
    $mode = 'process'
    $pid_ = $alive.Id
}}
$serving = $false
for ($i = 0; $i -lt 6; $i++) {{
    Start-Sleep -Seconds 2
    try {{
        $r = Invoke-WebRequest -Uri 'http://localhost:32400/identity' `
            -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) {{ $serving = $true; break }}
    }} catch {{}}
}}
if ($serving) {{
    Write-Output "STATUS:OK_${{mode}}:${{pid_}}"
}} else {{
    Write-Output "STATUS:FAIL_NO_API_${{mode}}:${{pid_}}"
}}
"""
    try:
        # retry=False: these are non-idempotent mutating actions — a silent
        # auto-retry on a connection drop could double-fire (see run()'s note).
        rc, out, err = client.run(_ps_encoded(script), retry=False)
    except Exception as exc:
        return f"Plex laptop unreachable: {exc}"
    if rc != 0:
        return f"Plex restart script error: {(err or '').strip() or f'rc={rc}'}"

    # Find the STATUS line (last line of stdout that starts with "STATUS:").
    status = ""
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("STATUS:"):
            status = line[len("STATUS:"):]
    return _format_restart_status(status)


# Mapping of short status codes from the restart script to user-facing
# prose. Lives in Python (not PowerShell) so the encoded command stays
# small and so the messages are easier to tune.
_RESTART_FAIL_PROCESS_DEAD = (
    "Plex was stopped but the SSH-launched relaunch did NOT survive. "
    "Plex Media Server in tray-app mode needs an interactive user session "
    "(Session 0 from SSH cannot fully initialize it). Workarounds: log "
    "into the laptop interactively to relaunch Plex, or install PlexService "
    "— it runs Plex as a real Windows service WITHOUT reinstalling Plex "
    "(your data folder and library metadata stay intact)."
)
_RESTART_FAIL_API_SERVICE = (
    "Plex service was restarted but the HTTP API is not responding on "
    "localhost:32400 after 12 seconds. The service shows running but Plex "
    "isn't serving. Check Plex's own log for startup errors (use "
    "plex_logs_tail) or wait a moment and retry."
)
_RESTART_FAIL_API_PROCESS = (
    "Plex Media Server.exe is running (PID {pid}) but the HTTP API is not "
    "responding on localhost:32400 after 12 seconds. This is the known "
    "Session-0 limitation: the process starts but cannot fully initialize "
    "without an interactive desktop. Workarounds: log into the laptop "
    "interactively, or install PlexService (no reinstall needed; metadata "
    "stays intact)."
)


def _format_restart_status(status: str) -> str:
    """Translate a STATUS:<code> tail from the restart script into the
    user-facing message. Unknown codes fall through with the raw status
    so a future PowerShell change isn't silently swallowed."""
    if status == "OK_service:0":
        return "Plex service restarted, API responding on localhost:32400."
    if status.startswith("OK_process:"):
        return (
            f"Plex relaunched successfully (PID {status.split(':', 1)[1]}, "
            f"API responding on localhost:32400). Tray icon will not "
            f"reappear until next interactive login."
        )
    if status == "FAIL_PROCESS_DEAD":
        return _RESTART_FAIL_PROCESS_DEAD
    if status.startswith("FAIL_NO_API_service"):
        return _RESTART_FAIL_API_SERVICE
    if status.startswith("FAIL_NO_API_process:"):
        return _RESTART_FAIL_API_PROCESS.format(pid=status.split(":", 1)[1])
    return f"Plex restart returned unexpected status: {status or '(empty)'}."


def _do_refresh_library(client: PlexLaptopClient, library_id: int, token: str) -> str:
    """Trigger a library re-scan via Plex's local HTTP API.

    Endpoint: GET /library/sections/{id}/refresh — Plex's documented
    refresh trigger. Asynchronous on Plex's side; this returns as soon as
    the request is acknowledged (HTTP 200), not when the scan finishes.
    """
    # Token is interpolated into the PowerShell body, then base64-encoded
    # by _ps_encoded — same trust model as having Plex itself store the
    # token in its data folder on disk. Never logged.
    script = f"""
try {{
    $resp = Invoke-WebRequest -Uri '{_PLEX_API_BASE}/library/sections/{library_id}/refresh' `
        -Method Get `
        -Headers @{{ 'X-Plex-Token' = '{token}' }} `
        -UseBasicParsing `
        -TimeoutSec 10 `
        -ErrorAction Stop
    Write-Output ("Library {library_id} refresh requested (HTTP " + $resp.StatusCode + "). The scan runs asynchronously on Plex's side.")
}} catch {{
    Write-Error $_.Exception.Message
    exit 1
}}
"""
    try:
        # retry=False: these are non-idempotent mutating actions — a silent
        # auto-retry on a connection drop could double-fire (see run()'s note).
        rc, out, err = client.run(_ps_encoded(script), retry=False)
    except Exception as exc:
        return f"Plex laptop unreachable: {exc}"
    if rc != 0:
        return f"Library refresh failed: {(err or '').strip() or 'unknown'}"
    return (out or "").strip() or f"Library {library_id} refresh requested."


def _do_empty_trash(client: PlexLaptopClient, library_id: int, token: str) -> str:
    """Empty Plex's trash for a specific library — IRREVERSIBLE.

    Endpoint: PUT /library/sections/{id}/emptyTrash — Plex's documented
    trash purge. After this fires, items previously in the library that
    no longer have associated media files are permanently removed from
    the database (along with watch history, ratings, etc.).
    """
    script = f"""
try {{
    $resp = Invoke-WebRequest -Uri '{_PLEX_API_BASE}/library/sections/{library_id}/emptyTrash' `
        -Method Put `
        -Headers @{{ 'X-Plex-Token' = '{token}' }} `
        -UseBasicParsing `
        -TimeoutSec 10 `
        -ErrorAction Stop
    Write-Output ("Library {library_id} trash emptied (HTTP " + $resp.StatusCode + ").")
}} catch {{
    Write-Error $_.Exception.Message
    exit 1
}}
"""
    try:
        # retry=False: these are non-idempotent mutating actions — a silent
        # auto-retry on a connection drop could double-fire (see run()'s note).
        rc, out, err = client.run(_ps_encoded(script), retry=False)
    except Exception as exc:
        return f"Plex laptop unreachable: {exc}"
    if rc != 0:
        return f"Empty trash failed: {(err or '').strip() or 'unknown'}"
    return (out or "").strip() or f"Library {library_id} trash emptied."


# --- Top-level executor ----------------------------------------------------


def execute_plex_action(client: PlexLaptopClient, params: dict) -> str:
    """Dispatch a plex_action call. Always returns a string — never raises.

    Server-side confirmation gate fires here, BEFORE any SSH call. A
    misbehaving prompt that emits `confirmed=false` (or omits it) gets a
    confirmation-required notice instead of an executed action.
    """
    action = (params.get("action") or "").lower().strip()
    library_id = params.get("library_id")
    confirmed = bool(params.get("confirmed"))

    valid_actions = {"restart_plex_server", "refresh_library", "empty_trash"}
    if action not in valid_actions:
        return (
            f"Unknown plex_action '{action}'. "
            f"Valid: {', '.join(sorted(valid_actions))}."
        )

    if not confirmed:
        # Server-side gate. The system prompt also instructs Claude to
        # confirm in plain language first; this layer makes the gate
        # robust against prompt regressions or hallucinated tool calls.
        readable = action.replace("_", " ")
        return (
            f"plex_action requires explicit user confirmation. Ask the user "
            f"to confirm '{readable}'"
            + (f" for library {library_id}" if library_id is not None else "")
            + ", then call this tool again with confirmed=true."
        )

    if action == "restart_plex_server":
        return _do_restart_plex(client)

    # Both API-backed actions need a library_id and the token.
    if library_id is None:
        return f"plex_action {action} requires the library_id parameter."
    try:
        library_id_int = int(library_id)
    except (TypeError, ValueError):
        return f"plex_action library_id must be an integer (got {library_id!r})."

    token = os.getenv("PLEX_TOKEN", "").strip()
    if not token:
        return (
            "plex_action needs PLEX_TOKEN set in .env to call the Plex API "
            "for refresh/trash operations."
        )

    if action == "refresh_library":
        return _do_refresh_library(client, library_id_int, token)
    return _do_empty_trash(client, library_id_int, token)
