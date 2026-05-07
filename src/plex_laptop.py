"""Remote Plex laptop tools (M24).

Three voice-grade diagnostic tools that reach the Plex laptop over SSH:

  plex_logs_tail        Last N lines of Plex Media Server.log
  plex_logs_search      Regex search inside Plex Media Server.log
  plex_laptop_health    CPU/RAM/disk/network on the remote box

Read-only by hard contract. No service restarts, no log rotation, no file
deletes — that's M25's territory if we ever build it. M24 is purely about
"tell me what's happening over there" without ever touching state.

Architecture in three observations:

1. paramiko, persistent connection, one auto-reconnect. Subprocess'ing
   `ssh.exe` per call would add 200-400ms of handshake to every tool turn,
   which compounds across multi-iteration agentic loops. paramiko is sync
   (matches src/llm.py's stream loop), pure-Python (cryptography is the only
   binary dep, already pulled by anthropic), and well-trodden — it's what
   Ansible uses underneath. Connection is opened lazily on first call and
   held until close(); one auto-reconnect handles idle timeout / wifi blip.

2. Base64-encoded PowerShell via `-EncodedCommand`. The path through
   Python → ssh client → cmd.exe → powershell.exe is four layers of
   quote-handling. Base64'ing the PowerShell script and passing it via
   -EncodedCommand makes the script body opaque to every shell along the
   way — only powershell.exe decodes it. This is Microsoft's documented
   pattern for piping arbitrary scripts; it avoids the entire class of
   "why is my $env:LOCALAPPDATA not expanding" bugs.

3. Host-key trust from the user's ~/.ssh/known_hosts (load_system_host_keys
   + RejectPolicy). The setup phase captured the laptop's host key
   interactively; from now on, if the laptop's host key changes for any
   reason, paramiko rejects the connection rather than silently trusting a
   new identity (TOFU). Failing closed is the right default for a
   programmatic client that can't ask the user.

Defensive contract — same as sports/weather/games/pc_diagnostics: never
raises, always returns a readable string. Connection failures, command
non-zero exits, parse errors all surface as voice-friendly error strings
that Claude can apologize through.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from typing import Any

import paramiko


# --- Constants -------------------------------------------------------------

# How long paramiko waits for SSH banner / auth handshake. LAN-local connect
# usually completes in <500ms; 8s is generous safety for a laptop that's
# just been woken from sleep.
_CONNECT_TIMEOUT = 8.0

# Per-command exec timeout. Tail/search are nearly instant; health uses
# Get-Counter which can sample for a couple seconds. 30s caps the rare hung
# command without making slow calls feel hung to the user.
_COMMAND_TIMEOUT = 30.0

# Voice budget: a tool result longer than this gets truncated. Claude
# summarizes for TTS anyway — sending 50KB of log lines is just waste.
_RESULT_MAX_CHARS = 4000

# Default location of the Plex Media Server log on the laptop. Confirmed in
# M24 setup via Resolve-Path. Plex stores logs under the user that runs the
# server process; for a personal install that's `youruser`. Override via env
# (PLEX_LAPTOP_LOG_PATH) if you ever change the install or use a non-default
# data path.
DEFAULT_LOG_PATH = (
    r"C:\Users\youruser\AppData\Local\Plex Media Server\Logs\Plex Media Server.log"
)


# --- Anthropic tool definitions --------------------------------------------

PLEX_LOGS_TAIL_TOOL = {
    "name": "plex_logs_tail",
    "description": (
        "Read the last N lines of the Plex Media Server log on the Plex "
        "laptop. Use for 'what's Plex doing right now?', 'is Plex okay?', "
        "'show me the latest activity'. Returns raw log lines."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "lines": {
                "type": "integer",
                "description": (
                    "Number of trailing lines to return. Default 40, capped "
                    "at 200 so the voice reply stays digestible."
                ),
            },
        },
    },
}

PLEX_LOGS_SEARCH_TOOL = {
    "name": "plex_logs_search",
    "description": (
        "Search the Plex Media Server log for lines matching a regex "
        "pattern (case-insensitive). Returns the most recent matches. Use "
        "for diagnostic queries: 'any transcoder errors today?', 'any 401 "
        "auth failures?', 'why did the stream buffer?'. The pattern is a "
        ".NET regex — alternation works ('transcode|error|warning')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Regex pattern. Case-insensitive. Examples: 'error', "
                    "'transcode.*fail', 'unauthori[sz]ed', '4\\d\\d'."
                ),
            },
            "max_matches": {
                "type": "integer",
                "description": (
                    "Maximum number of matching lines to return (most recent "
                    "first). Default 15, capped at 50."
                ),
            },
        },
        "required": ["pattern"],
    },
}

PLEX_LAPTOP_HEALTH_TOOL = {
    "name": "plex_laptop_health",
    "description": (
        "Read-only system health on the Plex laptop (the REMOTE machine). "
        "Use for 'how is the Plex box doing?', 'is the Plex laptop "
        "drowning?', 'what's the disk space on Plex?'. This is the remote "
        "equivalent of pc_diagnostics — for THIS PC use pc_diagnostics, "
        "for the Plex laptop use this tool. NEVER modifies state."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["overview", "cpu", "memory", "disk", "network"],
                "description": (
                    "overview = quick CPU/RAM/disk/uptime summary; "
                    "cpu = CPU% + top processes by CPU time; "
                    "memory = RAM% + top processes by working set; "
                    "disk = free space per drive; "
                    "network = interface IPs and adapter info."
                ),
            },
        },
        "required": ["mode"],
    },
}


# --- PlexLaptopClient ------------------------------------------------------


class PlexLaptopClient:
    """Synchronous SSH client with persistent connection + one auto-reconnect.

    Construct with the connection params; the actual TCP+SSH handshake is
    deferred until the first `run()` call so a dead host doesn't block app
    startup. After that, the connection is held open across calls — voice
    queries don't pay the handshake cost per turn.

    Thread-safe: a lock serializes exec_command, since paramiko.Channel is
    not concurrent-safe. The agentic loop is already serialized today, but
    the lock is cheap insurance against future parallelism.
    """

    def __init__(
        self,
        host: str,
        user: str,
        key_path: str,
        log_path: str = DEFAULT_LOG_PATH,
    ) -> None:
        if not host or not user or not key_path:
            raise ValueError("host, user, and key_path are required")
        self._host = host
        self._user = user
        self._key_path = os.path.expanduser(key_path)
        self._log_path = log_path
        self._client: paramiko.SSHClient | None = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def log_path(self) -> str:
        return self._log_path

    @property
    def host(self) -> str:
        return self._host

    def _connect(self) -> None:
        """Open a fresh SSH connection. Caller must hold self._lock."""
        if self._closed:
            raise RuntimeError("PlexLaptopClient is closed")

        c = paramiko.SSHClient()
        # Trust root: known_hosts captured during M24 setup. RejectPolicy
        # means a host-key mismatch fails closed instead of silently trusting
        # a new identity — the right default for a programmatic client.
        c.load_system_host_keys()
        c.set_missing_host_key_policy(paramiko.RejectPolicy())

        # Load the ed25519 private key. We pass it explicitly via pkey=
        # rather than rely on look_for_keys to avoid paramiko picking up
        # other keys in ~/.ssh; we want a single, deterministic key path.
        try:
            pkey = paramiko.Ed25519Key.from_private_key_file(self._key_path)
        except (paramiko.SSHException, OSError) as exc:
            raise RuntimeError(
                f"failed to load SSH key from {self._key_path}: {exc}"
            ) from exc

        c.connect(
            hostname=self._host,
            username=self._user,
            pkey=pkey,
            timeout=_CONNECT_TIMEOUT,
            banner_timeout=_CONNECT_TIMEOUT,
            auth_timeout=_CONNECT_TIMEOUT,
            allow_agent=False,
            look_for_keys=False,
        )
        # Keepalive packets every 30s so we notice dead connections quickly
        # rather than discovering it on the next exec_command timeout.
        transport = c.get_transport()
        if transport is not None:
            transport.set_keepalive(30)
        self._client = c

    def _safe_close_client(self) -> None:
        """Tear down the connection without raising. Caller holds self._lock."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def run(self, command: str, timeout: float = _COMMAND_TIMEOUT) -> tuple[int, str, str]:
        """Execute `command` on the laptop. Returns (exit_code, stdout, stderr).

        Auto-reconnects ONCE on a dropped connection. Raises on the second
        consecutive failure — caller is expected to catch and return a
        voice-friendly error string.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("PlexLaptopClient is closed")

            last_exc: Exception | None = None
            for attempt in (1, 2):
                if self._client is None:
                    try:
                        self._connect()
                    except Exception as exc:
                        # Connect itself failed — no point retrying immediately.
                        raise

                try:
                    _stdin, stdout, stderr = self._client.exec_command(
                        command, timeout=timeout
                    )
                    out = stdout.read().decode("utf-8", errors="replace")
                    err = stderr.read().decode("utf-8", errors="replace")
                    rc = stdout.channel.recv_exit_status()
                    return rc, out, err
                except (paramiko.SSHException, OSError, EOFError) as exc:
                    last_exc = exc
                    print(
                        f"[plex-laptop] command failed (attempt {attempt}): {exc}",
                        file=sys.stderr,
                    )
                    self._safe_close_client()
                    # Loop will reconnect on attempt 2.

            assert last_exc is not None
            raise last_exc

    def close(self) -> None:
        """Idempotent. Closes any active connection and prevents future use."""
        with self._lock:
            self._closed = True
            self._safe_close_client()


# --- PowerShell helpers ----------------------------------------------------


def _ps_encoded(script: str) -> str:
    """Wrap a PowerShell script for transmission via SSH using -EncodedCommand.

    Why this matters: the script crosses Python → ssh client → remote cmd.exe
    → powershell.exe, four layers of quote-handling. EncodedCommand is opaque
    to every layer except powershell.exe, which decodes the Base64 UTF-16-LE
    payload into the script body. Sidesteps every quoting bug.

    Microsoft's documented format: Base64 of UTF-16-LE-encoded source. Note
    UTF-16-LE specifically (not UTF-8); .NET's default string encoding.

    Prepending $ProgressPreference = SilentlyContinue is defensive: cmdlets
    that lazy-load a module on first use (Get-NetAdapter, Get-NetIPAddress,
    others from NetTCPIP / NetSecurity) emit a "Preparing modules for first
    use" progress record that, over a remote channel, gets serialized as
    CLIXML and corrupts our JSON output. Setting it once at the top kills
    the noise across all our scripts at zero cost.
    """
    full = "$ProgressPreference = 'SilentlyContinue'\n" + script
    encoded = base64.b64encode(full.encode("utf-16-le")).decode("ascii")
    return (
        f"powershell.exe -NoProfile -NonInteractive -EncodedCommand {encoded}"
    )


def _truncate_voice(text: str) -> str:
    """Cap output at _RESULT_MAX_CHARS so we don't blow Claude's tool-result
    budget for nothing. Marker tells Claude there's more if needed."""
    if len(text) > _RESULT_MAX_CHARS:
        return text[:_RESULT_MAX_CHARS] + "\n… [truncated]"
    return text


def _bytes_to_human(b: float) -> str:
    """14.2 GB beats 15234234234 B for TTS."""
    val = float(b)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} PB"


def _format_uptime(days: int, hours: int, minutes: int) -> str:
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


# --- Tool executors --------------------------------------------------------


def execute_plex_logs_tail(client: PlexLaptopClient, params: dict) -> str:
    """Return the last N lines of Plex Media Server.log."""
    try:
        lines = int(params.get("lines") or 40)
    except (TypeError, ValueError):
        lines = 40
    lines = max(1, min(200, lines))

    log_path = client.log_path
    # Single-quoted path inside PowerShell — no escape needed because Windows
    # paths don't contain single quotes. Get-Content's -Tail reads from the
    # end without scanning the whole file (efficient for large logs).
    script = (
        f"$path = '{log_path}'\n"
        f"if (-not (Test-Path -LiteralPath $path)) {{\n"
        f"    Write-Output 'LOG_NOT_FOUND'\n"
        f"    exit 0\n"
        f"}}\n"
        f"Get-Content -LiteralPath $path -Tail {lines} -Encoding UTF8\n"
    )

    try:
        rc, out, err = client.run(_ps_encoded(script))
    except Exception as exc:
        return f"Plex laptop unreachable: {exc}"

    if rc != 0:
        return f"Plex log read failed (rc={rc}): {(err or '').strip() or 'unknown'}"
    if "LOG_NOT_FOUND" in out and len(out.strip()) <= 20:
        return f"Plex log not found at {log_path} on the laptop."
    cleaned = out.strip("\r\n")
    if not cleaned:
        return "Plex log is empty (or no recent entries)."
    return _truncate_voice(cleaned)


def execute_plex_logs_search(client: PlexLaptopClient, params: dict) -> str:
    """Regex-search Plex Media Server.log; return most recent matches."""
    pattern = (params.get("pattern") or "").strip()
    if not pattern:
        return "Search requires a regex pattern."
    try:
        max_matches = int(params.get("max_matches") or 15)
    except (TypeError, ValueError):
        max_matches = 15
    max_matches = max(1, min(50, max_matches))

    log_path = client.log_path
    # Single quotes around the pattern preserve regex metacharacters; we have
    # to escape any literal single quotes the user passed by doubling them
    # (PowerShell's single-quote-in-single-quote escape).
    safe_pattern = pattern.replace("'", "''")

    script = (
        f"$path = '{log_path}'\n"
        f"if (-not (Test-Path -LiteralPath $path)) {{\n"
        f"    Write-Output 'LOG_NOT_FOUND'\n"
        f"    exit 0\n"
        f"}}\n"
        f"Select-String -LiteralPath $path -Pattern '{safe_pattern}' "
        f"-Encoding UTF8 -ErrorAction SilentlyContinue | "
        f"Select-Object -Last {max_matches} | "
        f"ForEach-Object {{ \"$($_.LineNumber): $($_.Line)\" }}\n"
    )

    try:
        rc, out, err = client.run(_ps_encoded(script))
    except Exception as exc:
        return f"Plex laptop unreachable: {exc}"

    if rc != 0:
        return f"Plex log search failed (rc={rc}): {(err or '').strip() or 'unknown'}"
    if "LOG_NOT_FOUND" in out and len(out.strip()) <= 20:
        return f"Plex log not found at {log_path} on the laptop."

    cleaned = out.strip("\r\n")
    if not cleaned:
        return f"No matches for /{pattern}/ in the Plex log."
    return _truncate_voice(cleaned)


def execute_plex_laptop_health(client: PlexLaptopClient, params: dict) -> str:
    """Return remote system health for the requested mode."""
    mode = (params.get("mode") or "overview").lower().strip()
    valid = {"overview", "cpu", "memory", "disk", "network"}
    if mode not in valid:
        return f"Unknown health mode '{mode}'. Use: {', '.join(sorted(valid))}."

    script = _HEALTH_SCRIPTS[mode]

    try:
        rc, out, err = client.run(_ps_encoded(script))
    except Exception as exc:
        return f"Plex laptop unreachable: {exc}"

    if rc != 0 or not out.strip():
        # Empty stdout with rc=0 usually means PowerShell hit an exception
        # routed to the error stream. Surface stderr for diagnostic value.
        msg = (err or "no output").strip()
        return f"Plex laptop {mode} query failed: {msg}"

    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        return f"Plex laptop {mode} returned unparseable output: {exc}"

    formatter = _HEALTH_FORMATTERS[mode]
    try:
        return formatter(data)
    except (KeyError, TypeError, ValueError) as exc:
        return f"Plex laptop {mode} formatting failed: {exc}"


# --- Health PowerShell scripts + formatters --------------------------------
#
# Each script computes data on the laptop and emits compact JSON via
# ConvertTo-Json -Compress. The corresponding formatter parses the JSON and
# composes a voice-friendly string. Keeping formatting in Python (one
# language, one place) is easier than hand-formatting PowerShell strings.


_HEALTH_OVERVIEW_PS = r"""
$os = Get-CimInstance Win32_OperatingSystem
$cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
$disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$boot = $os.LastBootUpTime
$uptime = New-TimeSpan -Start $boot -End (Get-Date)
$out = [ordered]@{
    cpu_pct      = [int]$cpu
    mem_total    = [long]($os.TotalVisibleMemorySize * 1024)
    mem_free     = [long]($os.FreePhysicalMemory * 1024)
    disk_total   = [long]$disk.Size
    disk_free    = [long]$disk.FreeSpace
    uptime_days  = [int]$uptime.Days
    uptime_hrs   = [int]$uptime.Hours
    uptime_mins  = [int]$uptime.Minutes
    hostname     = $env:COMPUTERNAME
}
$out | ConvertTo-Json -Compress
"""


def _format_overview(d: dict) -> str:
    mem_used = d["mem_total"] - d["mem_free"]
    mem_pct = round(100 * mem_used / d["mem_total"]) if d["mem_total"] else 0
    disk_used = d["disk_total"] - d["disk_free"]
    disk_pct = round(100 * disk_used / d["disk_total"]) if d["disk_total"] else 0
    uptime = _format_uptime(d["uptime_days"], d["uptime_hrs"], d["uptime_mins"])
    return (
        f"{d['hostname']}: CPU {d['cpu_pct']}%, "
        f"RAM {mem_pct}% ({_bytes_to_human(mem_used)} of {_bytes_to_human(d['mem_total'])}), "
        f"C: {disk_pct}% used ({_bytes_to_human(d['disk_free'])} free of "
        f"{_bytes_to_human(d['disk_total'])}). "
        f"Uptime {uptime}."
    )


_HEALTH_CPU_PS = r"""
$cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
$cores = (Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfLogicalProcessors
$top = Get-Process | Where-Object { $_.CPU -ne $null } |
    Sort-Object -Property CPU -Descending |
    Select-Object -First 5 |
    ForEach-Object { @{ name = $_.ProcessName; cpu_sec = [math]::Round($_.CPU, 1); ws = [long]$_.WorkingSet64 } }
$out = [ordered]@{
    cpu_pct = [int]$cpu
    cores   = [int]$cores
    top     = @($top)
}
$out | ConvertTo-Json -Compress -Depth 4
"""


def _format_cpu(d: dict) -> str:
    lines = [f"CPU {d['cpu_pct']}% across {d['cores']} logical cores."]
    top = d.get("top") or []
    if top:
        lines.append("Top processes by CPU seconds (cumulative since process start):")
        for p in top:
            lines.append(f"  {p['name']}: {p['cpu_sec']}s, {_bytes_to_human(p['ws'])} RAM")
    return "\n".join(lines)


_HEALTH_MEMORY_PS = r"""
$os = Get-CimInstance Win32_OperatingSystem
$mem_total = $os.TotalVisibleMemorySize * 1024
$mem_free = $os.FreePhysicalMemory * 1024
$top = Get-Process |
    Sort-Object -Property WorkingSet64 -Descending |
    Select-Object -First 5 |
    ForEach-Object { @{ name = $_.ProcessName; ws = [long]$_.WorkingSet64 } }
$out = [ordered]@{
    mem_total = [long]$mem_total
    mem_free  = [long]$mem_free
    top       = @($top)
}
$out | ConvertTo-Json -Compress -Depth 4
"""


def _format_memory(d: dict) -> str:
    mem_used = d["mem_total"] - d["mem_free"]
    mem_pct = round(100 * mem_used / d["mem_total"]) if d["mem_total"] else 0
    lines = [
        f"RAM {mem_pct}% used "
        f"({_bytes_to_human(mem_used)} of {_bytes_to_human(d['mem_total'])}), "
        f"{_bytes_to_human(d['mem_free'])} free."
    ]
    top = d.get("top") or []
    if top:
        lines.append("Top processes by working set:")
        for p in top:
            lines.append(f"  {p['name']}: {_bytes_to_human(p['ws'])}")
    return "\n".join(lines)


_HEALTH_DISK_PS = r"""
$drives = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
    ForEach-Object { @{ id = $_.DeviceID; total = [long]$_.Size; free = [long]$_.FreeSpace } }
@($drives) | ConvertTo-Json -Compress -Depth 4
"""


def _format_disk(d: Any) -> str:
    if isinstance(d, dict):
        d = [d]
    if not d:
        return "No fixed drives detected on Plex laptop."
    lines = []
    for v in d:
        total = v.get("total") or 0
        free = v.get("free") or 0
        used = total - free
        pct = round(100 * used / total) if total else 0
        lines.append(
            f"{v.get('id', '?')} {pct}% used "
            f"({_bytes_to_human(used)} of {_bytes_to_human(total)}, "
            f"{_bytes_to_human(free)} free)"
        )
    return "\n".join(lines)


# Uses the .NET NetworkInterface API rather than Get-NetAdapter / Get-NetIPAddress
# because the NetTCPIP module's CIM provider (MSFT_NetAdapter under
# ROOT/StandardCimv2) is broken on this laptop with HRESULT 0x80041010
# "Invalid class". Old-style Win32_* CIM classes (used by other modes) live
# in ROOT/CIMV2 and are unaffected. The .NET API has no WMI dependency at
# all, so this path is robust to any future provider rot.
_HEALTH_NETWORK_PS = r"""
$rows = @(
    [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
        Where-Object { $_.OperationalStatus -eq 'Up' -and $_.NetworkInterfaceType -ne 'Loopback' } |
        ForEach-Object {
            $ip4 = $_.GetIPProperties().UnicastAddresses |
                Where-Object {
                    $_.Address.AddressFamily -eq 'InterNetwork' -and
                    -not $_.Address.ToString().StartsWith('169.254.')
                } |
                Select-Object -First 1
            [ordered]@{
                name       = $_.Name
                desc       = $_.Description
                type       = $_.NetworkInterfaceType.ToString()
                speed_mbps = if ($_.Speed -gt 0) { [long]($_.Speed / 1000000) } else { 0 }
                ip         = if ($ip4) { $ip4.Address.ToString() } else { 'no-ipv4' }
            }
        }
)
ConvertTo-Json $rows -Compress -Depth 4
"""


def _format_network(d: Any) -> str:
    if isinstance(d, dict):
        d = [d]
    if not d:
        return "No active physical network interfaces on Plex laptop."
    lines = []
    for r in d:
        speed = r.get("speed_mbps") or 0
        speed_str = f"{speed} Mbps" if speed else "unknown speed"
        lines.append(
            f"{r.get('name', '?')}: {r.get('ip', '?')} ({speed_str}) — {r.get('desc', '?')}"
        )
    return "\n".join(lines)


_HEALTH_SCRIPTS: dict[str, str] = {
    "overview": _HEALTH_OVERVIEW_PS,
    "cpu": _HEALTH_CPU_PS,
    "memory": _HEALTH_MEMORY_PS,
    "disk": _HEALTH_DISK_PS,
    "network": _HEALTH_NETWORK_PS,
}

_HEALTH_FORMATTERS = {
    "overview": _format_overview,
    "cpu": _format_cpu,
    "memory": _format_memory,
    "disk": _format_disk,
    "network": _format_network,
}
