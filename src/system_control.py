"""System control tool — allowlisted Windows actions.

Ten actions, each individually scoped:

  open_app          launch an app from a small allowlist
  lock_workstation  lock the screen (Win+L equivalent)
  volume_set        set system master volume 0-100
  volume_mute       mute system master
  volume_unmute     unmute system master
  screen_off        power off the display(s)
  kill_process      terminate a named process — requires confirmed=true (M23)
  flush_dns         flush the DNS resolver cache — requires confirmed=true
                    AND admin (M40)
  restart_service   restart a Windows service by name — requires
                    confirmed=true AND admin AND a service name matching
                    ^[A-Za-z0-9._-]{1,128}$ (M40)
  stop_service      stop a Windows service by name — same gates as
                    restart_service (M75)
  start_service     start a Windows service by name — same gates as
                    restart_service (M75)
  dhcp_cycle        release + renew DHCP lease on all adapters — requires
                    confirmed=true AND admin (M42)

Security posture (M23 + M40 + M42 — the same shape applies to every
mutating verb added in the future):

- No arbitrary command exec. Every action is a fixed, named operation.
- No arbitrary app launch. open_app resolves names against `_APP_ALIASES`;
  unknown names return an error, not a Popen of attacker-controlled text.
- Confirmation contract enforced HERE, not in the system prompt. Every
  mutating verb checks `confirmed=true` before any subprocess fires. On
  the first call without it, the tool returns a "needs confirmation"
  string Claude paraphrases to the user; only after the user agrees does
  Claude call again with `confirmed=true`. Server-side gating means a
  misbehaving prompt or a hallucinated tool call can't bypass it.
- Elevation gate enforced HERE for actions that require it (flush_dns,
  restart_service, dhcp_cycle). `_IS_ADMIN` is cached at import via
  `autostart.is_admin()`; the gate fails fast with a "needs admin"
  message rather than discovering Access Denied from subprocess output.
  Aligns with the "Jarvis stays Jarvis, not Ultron" least-privilege
  principle — see the feedback memory of the same name.
- Per-verb input validation where the verb takes free-form input.
  restart_service's service-name regex is the only one currently;
  kill_process accepts any process name (psutil filters by exact match).
  Validation IS the security boundary — same principle as pc_shell.
- No shell composition. Every subprocess goes through
  `subprocess.run([list, of, args])` with `shell=False`. No f-strings
  into shell commands, no os.system(), no Popen(string, shell=True).
- Bounded. Per-action timeouts (20s flush_dns, 30s restart_service,
  30s+30s dhcp_cycle). Timeouts surface as readable strings, not hangs.
- Defensive. pycaw / ctypes / subprocess failures all become readable
  error strings — never raise through to the agentic loop.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
import time
from typing import Any

import psutil


# --- Elevation status ------------------------------------------------------
# Admin status is fixed for the life of a process — Windows decides at
# launch whether you're elevated and never changes its mind mid-session.
# So cache it once at module import. The mutating verbs (M40 — flush_dns,
# restart_service) check this and fail loudly rather than discovering the
# Access Denied from subprocess output.
#
# M41: single source of truth for `is_admin()` lives in `src.autostart`
# (alongside the elevated-relaunch helper that the tray uses). Importing
# at module top is cheap — autostart has no expensive deps.
from src.autostart import is_admin as _is_admin  # noqa: E402

_IS_ADMIN: bool = _is_admin()


# --- Anthropic tool definition ---------------------------------------------
SYSTEM_CONTROL_TOOL = {
    "name": "system_control",
    "description": (
        "Control THIS Windows PC with a fixed allowlist of safe actions: "
        "open an app, lock the workstation, set or mute system volume, "
        "turn off the display, kill a running process, flush the DNS "
        "resolver cache, restart / stop / start a Windows service, or "
        "cycle the DHCP lease (release + renew). "
        "IMPORTANT for the mutating actions (kill_process, flush_dns, "
        "restart_service, stop_service, start_service, dhcp_cycle): you "
        "MUST first ask the user to "
        "confirm in plain language ('Confirm: flush the DNS cache?' / "
        "'Confirm: restart the Spooler service?' / 'Confirm: stop the "
        "Spooler service?' / 'Confirm: release "
        "and renew the DHCP lease? This briefly drops network "
        "connectivity.'), wait for their explicit yes, THEN call this "
        "tool with confirmed=true. flush_dns, restart_service, "
        "stop_service, start_service, and "
        "dhcp_cycle additionally require Jarvis to be running as "
        "Administrator — the tool returns a clear error if not, which "
        "you should relay to the user. Other actions (open_app, lock, "
        "volume, screen_off) are low-impact and can run without explicit "
        "confirmation, but always announce what you're about to do."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "open_app", "lock_workstation",
                    "volume_set", "volume_mute", "volume_unmute",
                    "screen_off", "kill_process",
                    "flush_dns", "restart_service", "stop_service",
                    "start_service", "dhcp_cycle",
                ],
            },
            "target": {
                "type": "string",
                "description": (
                    "For open_app: app name (edge, chrome, firefox, notepad, "
                    "calculator, file explorer, vs code, terminal, task "
                    "manager, settings, control panel, powershell, cmd). "
                    "For kill_process: process name ('chrome.exe' or 'chrome'). "
                    "For restart_service / stop_service / start_service: the "
                    "Windows service name as it "
                    "appears in services.msc (e.g. 'Spooler', 'PlexService', "
                    "'WSearch'). Must match [A-Za-z0-9._-]{1,128}. "
                    "Ignored for other actions."
                ),
            },
            "value": {
                "type": "integer",
                "description": (
                    "For volume_set: integer 0-100. Ignored for other actions."
                ),
            },
            "confirmed": {
                "type": "boolean",
                "description": (
                    "Required true for kill_process, flush_dns, "
                    "restart_service, stop_service, start_service, and "
                    "dhcp_cycle — only set this "
                    "AFTER the user has explicitly confirmed in "
                    "conversation. The tool rejects those actions "
                    "without it. Ignored for other actions."
                ),
            },
        },
        "required": ["action"],
    },
}


# Allowlisted apps. Maps voice-friendly names → executable names. We launch
# via `cmd /c start "" <exe>`, which respects Windows' App Paths registry
# so apps like Edge / Chrome resolve even when they're not in PATH.
_APP_ALIASES: dict[str, str] = {
    "edge": "msedge",
    "microsoft edge": "msedge",
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "powershell": "powershell",
    "pwsh": "pwsh",
    "windows terminal": "wt",
    "terminal": "wt",
    "command prompt": "cmd",
    "cmd": "cmd",
    "task manager": "taskmgr",
    "control panel": "control",
    "settings": "ms-settings:",
    "spotify": "spotify",
    "discord": "discord",
}


def _do_open_app(target: str) -> str:
    if not target:
        return "App name required for open_app."
    key = target.lower().strip()
    exe = _APP_ALIASES.get(key)
    if not exe:
        known = sorted(set(_APP_ALIASES.keys()))
        # Cap the suggestion list — Claude only needs a hint, not the full set.
        sample = ", ".join(known[:12])
        return (
            f"App '{target}' not in allowlist. Known apps include: {sample}, "
            f"and {len(known) - 12} more."
        )
    # cmd /c start "" <exe> is the cleanest Windows-native launch:
    # - respects App Paths registry (Edge/Chrome resolve without PATH)
    # - ms-settings: protocol works the same way
    # - returns immediately (start spawns + detaches)
    # We pass a list (not a shell string) so quoting is handled by Python.
    # `exe` only ever comes from `_APP_ALIASES`, never user input — no
    # injection surface here.
    try:
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", exe],
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return f"Opened {target}."
    except OSError as exc:
        return f"Could not open {target}: {exc}"


def _do_lock_workstation() -> str:
    """ctypes against user32.LockWorkStation — returns nonzero on success.
    Equivalent to Win+L, no UAC, no permissions issue."""
    try:
        ok = ctypes.windll.user32.LockWorkStation()
        return "Workstation locked." if ok else "Lock request failed."
    except (AttributeError, OSError) as exc:
        return f"Could not lock workstation: {exc}"


def _get_audio_endpoint():
    """Return an `IAudioEndpointVolume` interface for the default playback
    device.

    Two non-obvious requirements landed during M23 voice testing:

    1. **COM init per thread.** pycaw uses `comtypes`, which requires
       `CoInitialize()` to have been called on the current OS thread before
       any audio device call. The agentic loop in `src/llm.py` runs in a
       worker thread (text_input_loop / listen_loop) that may not have
       touched COM yet — without this, calls fail with
       `[WinError -2147221008] CoInitialize has not been called`.
       The call is idempotent: returns `S_FALSE` if already initialized,
       which we ignore. We do NOT call `CoUninitialize()` — letting the
       thread retain its COM space is correct for a long-running
       worker.
    2. **pycaw 20251023+ API.** `AudioUtilities.GetSpeakers()` now returns
       an `AudioDevice` wrapper instead of the raw `IMMDevice` COM
       interface. The wrapper exposes `.EndpointVolume` (lazy-cached) which
       does the `.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)`
       dance internally. The older `IMMDevice.Activate(...)` pattern raises
       `'AudioDevice' object has no attribute 'Activate'` on this version.

    Lazy import keeps the rest of system_control working if pycaw / comtypes
    ever fail to import (wedged build, missing audio service).
    """
    import comtypes  # type: ignore
    from pycaw.pycaw import AudioUtilities  # type: ignore

    try:
        comtypes.CoInitialize()
    except OSError:
        # RPC_E_CHANGED_MODE if a different threading model was already
        # selected on this thread — fine, we just don't own the init.
        pass

    return AudioUtilities.GetSpeakers().EndpointVolume


def _do_volume_set(value: Any) -> str:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return "volume_set requires an integer 0-100 in `value`."
    v = max(0, min(100, v))
    try:
        vol = _get_audio_endpoint()
        # pycaw uses a 0.0-1.0 scalar (linear, not dB).
        vol.SetMasterVolumeLevelScalar(v / 100.0, None)
        # Also unmute if previously muted — "set volume to 50" implies audible.
        try:
            vol.SetMute(0, None)
        except Exception:  # noqa: BLE001 — best-effort unmute
            pass
        return f"Volume set to {v}%."
    except ImportError:
        return "Volume control unavailable (pycaw not installed)."
    except Exception as exc:  # noqa: BLE001 — pycaw raises OSError, comtypes errors, etc.
        print(
            f"[system_control] volume_set failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return f"Could not set volume: {exc}"


def _do_volume_mute() -> str:
    try:
        _get_audio_endpoint().SetMute(1, None)
        return "Muted."
    except ImportError:
        return "Volume control unavailable (pycaw not installed)."
    except Exception as exc:  # noqa: BLE001
        print(
            f"[system_control] volume_mute failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return f"Could not mute: {exc}"


def _do_volume_unmute() -> str:
    try:
        _get_audio_endpoint().SetMute(0, None)
        return "Unmuted."
    except ImportError:
        return "Volume control unavailable (pycaw not installed)."
    except Exception as exc:  # noqa: BLE001
        print(
            f"[system_control] volume_unmute failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return f"Could not unmute: {exc}"


def _do_screen_off() -> str:
    """Broadcast WM_SYSCOMMAND with SC_MONITORPOWER, lParam=2 (off).
    This is the same call the Windows shell uses for 'turn off display' — no
    permissions, no UAC, no extra deps."""
    WM_SYSCOMMAND = 0x0112
    SC_MONITORPOWER = 0xF170
    HWND_BROADCAST = 0xFFFF
    MONITOR_OFF = 2
    try:
        ctypes.windll.user32.SendMessageW(
            HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_OFF
        )
        return "Display turned off."
    except (AttributeError, OSError) as exc:
        return f"Could not turn off display: {exc}"


def _do_kill_process(target: str, confirmed: bool) -> str:
    """Terminate processes matching `target` by name. Requires confirmed=true.

    Server-side enforcement of the confirmation contract. If Claude (or some
    future caller) ever forgets to ask first, the worst that can happen is
    the tool returns a "needs confirmation" string — no process gets killed.
    """
    if not target:
        return "Process name required for kill_process."
    if not confirmed:
        return (
            f"kill_process requires explicit user confirmation. Ask the user "
            f"to confirm terminating '{target}', then call this tool again "
            f"with confirmed=true."
        )

    name = target.lower().strip()
    name_with_exe = name if name.endswith(".exe") else name + ".exe"

    killed: list[int] = []
    denied = 0
    for p in psutil.process_iter(["pid", "name"]):
        try:
            pname = (p.info.get("name") or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if pname != name and pname != name_with_exe:
            continue
        try:
            p.terminate()
            killed.append(p.info.get("pid"))
        except psutil.AccessDenied:
            denied += 1
        except psutil.NoSuchProcess:
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[system_control] terminate failed: {exc}", file=sys.stderr)

    if not killed and denied == 0:
        return f"No process named '{target}' found."
    parts = []
    if killed:
        parts.append(f"Terminated {len(killed)} process(es) named '{target}'.")
    if denied:
        parts.append(f"{denied} process(es) refused (likely needs admin).")
    return " ".join(parts)


def _do_flush_dns(confirmed: bool) -> str:
    """Run `ipconfig.exe /flushdns`. Requires confirmed=true AND admin.

    Why ipconfig and not the PowerShell `Clear-DnsClientCache` cmdlet:
    they do effectively the same thing on Windows 10/11 (both invoke the
    DnsApi flush via the same RPC), but ipconfig.exe is older, smaller,
    and always present — `Clear-DnsClientCache` lives in a PowerShell
    module that has occasionally broken across Windows updates. ipconfig
    is the "always works" path.
    """
    if not confirmed:
        return (
            "flush_dns requires explicit user confirmation. Ask the user "
            "to confirm flushing the DNS resolver cache, then call this "
            "tool again with confirmed=true."
        )
    if not _IS_ADMIN:
        return (
            "Flushing the DNS cache requires Jarvis to be running as "
            "Administrator, sir. Restart Jarvis from an elevated prompt "
            "and try again."
        )

    started = time.monotonic()
    try:
        result = subprocess.run(
            ["ipconfig.exe", "/flushdns"],
            capture_output=True, text=True, timeout=20, shell=False,
        )
    except subprocess.TimeoutExpired:
        print("[system_control] flush_dns timed out after 20s", file=sys.stderr)
        return "DNS flush timed out after 20 seconds."
    except (OSError, FileNotFoundError) as exc:
        print(f"[system_control] flush_dns spawn failed: {exc}", file=sys.stderr)
        return f"Could not run ipconfig: {exc}"

    elapsed = time.monotonic() - started
    rc = result.returncode
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    print(
        f"[system_control] flush_dns exit={rc} in {elapsed:.2f}s",
        file=sys.stderr,
    )
    if rc != 0:
        # Non-zero exit despite us being admin is unusual — surface
        # whatever ipconfig said so Claude can paraphrase to the user.
        return f"DNS flush failed (exit {rc}). {stderr or stdout}".strip()
    return "DNS resolver cache flushed."


def _do_dhcp_cycle(confirmed: bool) -> str:
    """Release + renew all DHCP-managed adapters in sequence. Requires
    confirmed=true AND admin.

    Why one combined verb rather than separate `dhcp_release` and
    `dhcp_renew`: in practice you almost never want one without the
    other. Release without renew leaves the host offline; renew without
    release first won't dislodge a stuck lease. The combined cycle IS
    the SRE move — splitting it adds two tool calls' worth of friction
    for no real diagnostic gain.

    Failure handling is deliberately ordered: if release fails, we DO
    NOT proceed to renew. Renewing on top of an uncertain network state
    risks compounding the breakage. We surface the release error and
    let the user decide.

    Timeouts: 30s per subprocess. Typical release is <1s, typical renew
    is 1-5s, but a slow / unreachable DHCP server can legitimately push
    renew toward Windows's own internal 60s ceiling. 30s catches the
    "DHCP server is down" case before the user gets bored of waiting,
    while still allowing a normal-ish slow renew to complete.
    """
    if not confirmed:
        return (
            "dhcp_cycle requires explicit user confirmation. Warn the "
            "user this will briefly drop network connectivity on all "
            "DHCP-managed adapters, then ask them to confirm. Once "
            "they agree, call this tool again with confirmed=true."
        )
    if not _IS_ADMIN:
        return (
            "Cycling the DHCP lease requires Jarvis to be running as "
            "Administrator, sir. Restart Jarvis from an elevated "
            "prompt and try again."
        )

    # --- Phase 1: release -------------------------------------------------
    started = time.monotonic()
    try:
        rel = subprocess.run(
            ["ipconfig.exe", "/release"],
            capture_output=True, text=True, timeout=30, shell=False,
        )
    except subprocess.TimeoutExpired:
        print("[system_control] dhcp_cycle release timed out after 30s",
              file=sys.stderr)
        return "ipconfig /release timed out after 30 seconds — network state uncertain."
    except (OSError, FileNotFoundError) as exc:
        print(f"[system_control] dhcp_cycle release spawn failed: {exc}",
              file=sys.stderr)
        return f"Could not run ipconfig /release: {exc}"

    rel_elapsed = time.monotonic() - started
    rel_rc = rel.returncode
    print(
        f"[system_control] dhcp_cycle release exit={rel_rc} in {rel_elapsed:.2f}s",
        file=sys.stderr,
    )
    if rel_rc != 0:
        msg = (rel.stderr or rel.stdout or "no output").strip()
        return f"ipconfig /release failed (exit {rel_rc}). {msg}"

    # --- Phase 2: renew ---------------------------------------------------
    started = time.monotonic()
    try:
        ren = subprocess.run(
            ["ipconfig.exe", "/renew"],
            capture_output=True, text=True, timeout=30, shell=False,
        )
    except subprocess.TimeoutExpired:
        print("[system_control] dhcp_cycle renew timed out after 30s",
              file=sys.stderr)
        # Release already succeeded — the network is in "no lease" state
        # right now. Surface that explicitly so the user / Claude knows
        # to retry rather than assume connectivity is back.
        return (
            "Released DHCP lease successfully, but ipconfig /renew "
            "timed out after 30 seconds. The host has no active lease "
            "right now — try again, or run /renew manually."
        )
    except (OSError, FileNotFoundError) as exc:
        print(f"[system_control] dhcp_cycle renew spawn failed: {exc}",
              file=sys.stderr)
        return f"Released lease, but renew spawn failed: {exc}"

    ren_elapsed = time.monotonic() - started
    ren_rc = ren.returncode
    print(
        f"[system_control] dhcp_cycle renew exit={ren_rc} in {ren_elapsed:.2f}s",
        file=sys.stderr,
    )
    if ren_rc != 0:
        msg = (ren.stderr or ren.stdout or "no output").strip()
        return f"Released DHCP lease, but renew failed (exit {ren_rc}). {msg}"

    return "DHCP lease cycled — released and renewed on all adapters."


# Service-name validator. Windows service names are alphanumeric plus a
# handful of separators in practice (Spooler, PlexService, WSearch,
# Themes, RpcSs, w3svc, MSSQL$SQLEXPRESS — that $ is the only common
# special char and we exclude it deliberately to avoid PowerShell
# expansion surprises; users with named SQL instances can request that
# later). 128-char cap mirrors the documented Windows SCM limit.
_SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


# The three service verbs differ ONLY in the PowerShell cmdlet, the
# success/timeout wording, and the confirmation gerund — everything else
# (regex validation, confirmation gate, admin gate, subprocess plumbing,
# 30s timeout, error passthrough) is identical. (action) → (cmdlet,
# past-tense verb, gerund, short verb). M75 — the third service verb is the
# one that earned the rule-of-three extraction (same philosophy as the M43
# http_util pull-out: extract on the tool that makes it three).
_SERVICE_CMDLETS: dict[str, tuple[str, str, str, str]] = {
    "restart_service": ("Restart-Service", "Restarted", "restarting", "restart"),
    "stop_service":    ("Stop-Service",    "Stopped",   "stopping",   "stop"),
    "start_service":   ("Start-Service",   "Started",   "starting",   "start"),
}


def _do_service_action(action: str, target: str, confirmed: bool) -> str:
    """Run a Windows service verb (`Restart-Service` / `Stop-Service` /
    `Start-Service`) via PowerShell. Requires confirmed=true AND admin AND a
    service name that matches the regex.

    PowerShell (rather than `net stop`/`net start` or `sc.exe`): the
    *-Service cmdlets propagate errors as exceptions with usable messages,
    and the `-ErrorAction Stop` + try/catch idiom lets us exit non-zero on
    failure so we can distinguish "service not found" from "denied" from
    "succeeded."

    Deliberately NO `-Force` on Stop-Service: a service with running
    dependents will fail with a clear "Cannot stop ... because it has
    dependent services" message that we surface, rather than silently
    stopping the dependents too (a bigger blast radius the user should opt
    into explicitly — the "Jarvis stays Jarvis, not Ultron" least-privilege
    principle). Same reason Start-Service is left to surface its own
    "dependency failed to start" errors.
    """
    cmdlet, past, gerund, verb = _SERVICE_CMDLETS[action]
    if not target:
        return f"Service name required for {action}."
    if not _SERVICE_NAME_RE.match(target):
        # Reject anything with spaces, quotes, semicolons, pipes,
        # backticks, ampersands, $, etc. before it ever reaches PowerShell.
        # Validation IS the security boundary — same principle as pc_shell.
        return (
            f"Service name '{target}' contains invalid characters. "
            f"Names must match {_SERVICE_NAME_RE.pattern}."
        )
    if not confirmed:
        return (
            f"{action} requires explicit user confirmation. Ask the "
            f"user to confirm {gerund} the '{target}' service, then call "
            f"this tool again with confirmed=true."
        )
    if not _IS_ADMIN:
        return (
            f"{gerund.capitalize()} a Windows service requires Jarvis to be "
            "running as Administrator, sir. Restart Jarvis from an elevated "
            "prompt and try again."
        )

    # -NoProfile skips $PROFILE.ps1 (faster, deterministic). The *-Service
    # cmdlets by default don't fail loudly when the service is missing or
    # denied — `-ErrorAction Stop` promotes those to terminating errors that
    # our try/catch surfaces, and `exit 1` makes the subprocess exit non-zero
    # so we know to look at stderr.
    ps_script = (
        f"try {{ {cmdlet} -Name '{target}' -ErrorAction Stop; "
        f"Write-Output '{past} {target}.' }} "
        f"catch {{ Write-Error $_.Exception.Message; exit 1 }}"
    )

    started = time.monotonic()
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True, text=True, timeout=30, shell=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[system_control] {action} '{target}' timed out after 30s",
            file=sys.stderr,
        )
        return f"{gerund.capitalize()} '{target}' timed out after 30 seconds."
    except (OSError, FileNotFoundError) as exc:
        print(
            f"[system_control] {action} '{target}' spawn failed: {exc}",
            file=sys.stderr,
        )
        return f"Could not run PowerShell: {exc}"

    elapsed = time.monotonic() - started
    rc = result.returncode
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    print(
        f"[system_control] {action} '{target}' exit={rc} in {elapsed:.2f}s",
        file=sys.stderr,
    )
    if rc != 0:
        # PowerShell error wraps the SCM error — typical messages: "Cannot
        # find any service with service name 'foo'.", "Cannot open <name>
        # service on computer '.'." (the denied-access shape), "Cannot stop
        # ... because it has dependent services". Pass it through so the user
        # knows whether the name was wrong or whether something else blocked it.
        msg = stderr or stdout or "no output"
        return f"Could not {verb} '{target}': {msg}"
    return stdout or f"{past} {target}."


def execute_system_control_tool(params: dict) -> str:
    """Run the tool. Always returns a string — never raises."""
    action = (params.get("action") or "").lower().strip()
    target = (params.get("target") or "").strip()
    value = params.get("value")
    confirmed = bool(params.get("confirmed"))

    try:
        if action == "open_app":
            return _do_open_app(target)
        if action == "lock_workstation":
            return _do_lock_workstation()
        if action == "volume_set":
            return _do_volume_set(value)
        if action == "volume_mute":
            return _do_volume_mute()
        if action == "volume_unmute":
            return _do_volume_unmute()
        if action == "screen_off":
            return _do_screen_off()
        if action == "kill_process":
            return _do_kill_process(target, confirmed)
        if action == "flush_dns":
            return _do_flush_dns(confirmed)
        if action in _SERVICE_CMDLETS:
            return _do_service_action(action, target, confirmed)
        if action == "dhcp_cycle":
            return _do_dhcp_cycle(confirmed)
        return f"Unknown action '{action}'."
    except Exception as exc:  # noqa: BLE001 — defensive last-resort net
        print(f"[system_control] {action} raised: {exc}", file=sys.stderr)
        return f"System control error in '{action}': {exc}"
