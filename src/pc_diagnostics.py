"""PC diagnostics tool — read-only Windows system telemetry.

Eight modes covering the natural voice-query distribution:

  overview   "How is my PC doing?"           — CPU/RAM/disk/uptime one-liner
  cpu        "What's eating my CPU?"          — % + top-CPU processes
  memory     "What's eating my RAM?"          — % + top-memory processes
  disk       "How much disk space?"           — per-volume free / total / %
  processes  "What's running?"                — top N by CPU or memory
  services   "Is the Plex service running?"   — service list + status filter
  events     "Any recent errors?"             — System log Errors/Warnings (24h)
  network    "What's my IP / network usage?"  — interface IPs + throughput

Same defensive contract as sports/weather/games/plex_mcp: never raises,
always returns a readable string. Voice replies need to be short, so each
mode caps output at limit=10 (max 25) and uses voice-friendly phrasing
("44% used" not "0.4421 ratio").

Read-only by design. Nothing in this module modifies system state — actions
live in src/system_control.py with separate guardrails.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import psutil


# --- Anthropic tool definition ---------------------------------------------
PC_DIAGNOSTICS_TOOL = {
    "name": "pc_diagnostics",
    "description": (
        "Read-only telemetry about THIS Windows PC: CPU, memory, disk, "
        "running processes, services, network, and recent system event-log "
        "entries. Use for any 'how is my PC doing', 'what's slowing me down', "
        "'check disk space', 'what's eating my CPU', 'is service X running', "
        "or 'any recent errors' question. This tool ONLY reads — it never "
        "modifies state. For actions (lock, mute, kill a process, open an "
        "app), use system_control instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": [
                    "overview", "cpu", "memory", "disk",
                    "processes", "services", "events", "network",
                ],
                "description": (
                    "overview = quick health summary (CPU/RAM/disk/uptime); "
                    "cpu = CPU% + top-CPU processes; "
                    "memory = RAM% + top-memory processes; "
                    "disk = free space per drive; "
                    "processes = top N processes (sort with sort_by); "
                    "services = Windows services, optionally filtered; "
                    "events = recent System log errors/warnings (last 24h); "
                    "network = interface IPs and throughput counters."
                ),
            },
            "sort_by": {
                "type": "string",
                "enum": ["cpu", "memory"],
                "description": (
                    "For mode=processes: sort by 'cpu' (default) or 'memory'."
                ),
            },
            "filter": {
                "type": "string",
                "description": (
                    "For mode=services: substring matched against the service "
                    "name (case-insensitive). For mode=events: 'errors' (default), "
                    "'warnings', or 'both'."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Max items returned. Default 10, capped at 25 so voice "
                    "replies stay short."
                ),
            },
        },
        "required": ["mode"],
    },
}


_DEFAULT_LIMIT = 10
_MAX_LIMIT = 25
# psutil.cpu_percent and psutil.Process.cpu_percent both need a baseline read
# followed by a sample-window read. 0.5s is short enough not to drag voice
# latency and long enough to give meaningful per-process numbers.
_CPU_SAMPLE_SEC = 0.5
# PowerShell event-log queries can hang briefly on busy systems; cap them.
_EVENT_QUERY_TIMEOUT_SEC = 8.0


def _clamp_limit(limit: Any) -> int:
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT
    return max(1, min(_MAX_LIMIT, n))


def _bytes_to_human(b: float) -> str:
    """14.2 GB beats 15234234234 B for TTS."""
    val = float(b)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} PB"


def _format_uptime() -> str:
    delta = datetime.now() - datetime.fromtimestamp(psutil.boot_time())
    total_min = int(delta.total_seconds() / 60)
    days, rem = divmod(total_min, 60 * 24)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


_CPU_CORES = psutil.cpu_count(logical=True) or 1

# Pseudo-processes that always top a CPU sort but aren't useful answers to
# "what's eating my CPU". System Idle Process represents *idle* time, not
# load; on a 12-core box it routinely shows 1100%+ in psutil.
_PROCESS_BLOCKLIST = {
    "system idle process",
}


def _sample_top_processes(sort_by: str, limit: int) -> list[tuple[str, float, float]]:
    """Sample (name, cpu%, mem%) for every process, sorted by either field.

    Two-pass sampling: psutil.Process.cpu_percent() returns 0 on first call
    — it needs a baseline. Prime every iter, sleep, sample. Adds ~0.5s
    latency but voice users won't notice; the LLM round-trip is longer.

    CPU normalization: psutil reports per-process CPU as a percentage of ONE
    core (top/ps convention). On multicore systems a multithreaded process
    can show 200%, 400%+ which confuses voice users expecting "% of total
    CPU". Divide by logical core count so a process pegging the whole CPU
    reads as 100%, matching Task Manager's default view.
    """
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            p.cpu_percent()  # prime baseline
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(_CPU_SAMPLE_SEC)
    out: list[tuple[str, float, float]] = []
    for p in procs:
        try:
            name = p.info.get("name") or f"pid {p.info.get('pid')}"
            if name.lower() in _PROCESS_BLOCKLIST:
                continue
            cpu_per_core = p.cpu_percent()
            cpu = cpu_per_core / _CPU_CORES
            mem = p.memory_percent()
            out.append((name, cpu, mem))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if sort_by == "memory":
        out.sort(key=lambda x: x[2], reverse=True)
    else:
        out.sort(key=lambda x: x[1], reverse=True)
    return out[:limit]


def _do_overview() -> str:
    cpu = psutil.cpu_percent(interval=_CPU_SAMPLE_SEC)
    mem = psutil.virtual_memory()
    try:
        c = psutil.disk_usage("C:\\")
        c_str = f"C: {c.percent}% used ({_bytes_to_human(c.used)} of {_bytes_to_human(c.total)})"
    except OSError:
        c_str = "C: unavailable"
    return (
        f"CPU {cpu:.0f}%, RAM {mem.percent:.0f}% "
        f"({_bytes_to_human(mem.used)} of {_bytes_to_human(mem.total)}), "
        f"{c_str}. Uptime {_format_uptime()}."
    )


def _do_cpu(limit: int) -> str:
    cpu = psutil.cpu_percent(interval=_CPU_SAMPLE_SEC)
    cores = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    freq = psutil.cpu_freq()
    freq_str = f"{freq.current/1000:.1f} GHz" if freq else "freq unavailable"
    top = _sample_top_processes("cpu", limit)
    lines = [f"CPU {cpu:.0f}% across {cores} logical cores ({physical} physical), {freq_str}."]
    if top:
        lines.append("Top processes by CPU:")
        for name, c, _m in top:
            if c > 0.05:
                lines.append(f"  {name}: {c:.1f}%")
    return "\n".join(lines)


def _do_memory(limit: int) -> str:
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    top = _sample_top_processes("memory", limit)
    lines = [
        f"RAM {mem.percent:.0f}% used "
        f"({_bytes_to_human(mem.used)} of {_bytes_to_human(mem.total)}), "
        f"{_bytes_to_human(mem.available)} available. "
        f"Swap {swap.percent:.0f}% ({_bytes_to_human(swap.used)} of {_bytes_to_human(swap.total)})."
    ]
    if top:
        lines.append("Top processes by memory:")
        for name, _c, m in top:
            if m > 0.1:
                lines.append(f"  {name}: {m:.1f}%")
    return "\n".join(lines)


def _do_disk() -> str:
    """Per-volume free space. Skip CD-ROM/removable drives that aren't ready
    (psutil raises on those). Net result: every fixed volume the user cares
    about, none of the noise."""
    lines = []
    for part in psutil.disk_partitions(all=False):
        # `cdrom` opts means a removable, often-empty optical drive. Skip.
        if "cdrom" in (part.opts or "") or part.fstype == "":
            continue
        try:
            u = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        lines.append(
            f"{part.device} {u.percent}% used "
            f"({_bytes_to_human(u.used)} of {_bytes_to_human(u.total)}, "
            f"{_bytes_to_human(u.free)} free)"
        )
    if not lines:
        return "No fixed drives detected."
    return "\n".join(lines)


def _do_processes(sort_by: str, limit: int) -> str:
    top = _sample_top_processes(sort_by, limit)
    if not top:
        return "No processes found."
    header = f"Top {len(top)} processes by {sort_by or 'cpu'}:"
    lines = [header]
    for name, c, m in top:
        lines.append(f"  {name}: cpu {c:.1f}%, ram {m:.1f}%")
    return "\n".join(lines)


def _do_services(filter_str: str, limit: int) -> str:
    """List Windows services, optionally substring-filtered. We bias toward
    'running' services (those are the ones a user usually asks about) and
    only show stopped ones if the filter matches them — otherwise the list
    is hundreds of entries long.

    Per-service try/except is mandatory: a Windows install commonly has at
    least one service whose registered config is stale or whose binary is
    missing, and `as_dict()` will raise OSError (WinError 2 from
    QueryServiceConfig2W). Skip those rather than aborting the whole call.
    """
    needle = (filter_str or "").lower().strip()
    rows: list[tuple[str, str, str]] = []
    try:
        iterator = psutil.win_service_iter()
    except (AttributeError, psutil.AccessDenied) as exc:
        return f"Service enumeration unavailable: {exc}"
    for svc in iterator:
        try:
            info = svc.as_dict()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            # OSError covers WinError 2 (missing binary), 5 (access denied
            # at the per-service level), 1060 (service does not exist), and
            # the rare WinError 1115 (system shutting down).
            continue
        name = info.get("name", "")
        display = info.get("display_name", "")
        status = info.get("status", "")
        if needle:
            hay = f"{name} {display}".lower()
            if needle not in hay:
                continue
        else:
            # No filter — only show running services to keep voice short.
            if status != "running":
                continue
        rows.append((name, display, status))

    if not rows:
        if needle:
            return f"No services matching '{filter_str}'."
        return "No running services found."
    rows.sort(key=lambda x: (x[2] != "running", x[0]))
    rows = rows[:limit]
    return "\n".join(f"{n}: {s} ({d})" for n, d, s in rows)


def _do_events(filter_str: str, limit: int) -> str:
    """Recent System log errors/warnings via PowerShell Get-WinEvent.

    Why PowerShell over pywin32: pywin32 adds a heavy install (~80 MB) and
    Win32 event log APIs are a mouthful. Get-WinEvent is built into Windows
    and gives us structured JSON output with ConvertTo-Json. The 8s timeout
    handles the rare case where the log is huge or the channel is busy.
    """
    sel = (filter_str or "errors").lower().strip()
    if sel == "warnings":
        levels = "3"
    elif sel == "both":
        levels = "1,2,3"
    else:
        levels = "1,2"  # 1=Critical, 2=Error

    # The truncation in PowerShell ([Math]::Min(160, ...)) keeps each message
    # short enough to read aloud — a typical Event Log message can be many
    # paragraphs of stack-traceish prose otherwise.
    ps_cmd = (
        f"Get-WinEvent -FilterHashtable @{{LogName='System'; Level={levels}; "
        f"StartTime=(Get-Date).AddHours(-24)}} -MaxEvents {limit} "
        f"-ErrorAction SilentlyContinue | "
        f"Select-Object @{{Name='time';Expression={{$_.TimeCreated.ToString('HH:mm')}}}}, "
        f"@{{Name='level';Expression={{$_.LevelDisplayName}}}}, "
        f"@{{Name='source';Expression={{$_.ProviderName}}}}, "
        f"@{{Name='id';Expression={{$_.Id}}}}, "
        f"@{{Name='msg';Expression={{if ($_.Message) "
        f"{{$_.Message.Substring(0,[Math]::Min(160,$_.Message.Length))}} else {{''}}}}}} | "
        f"ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=_EVENT_QUERY_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return "Event log query timed out."
    except FileNotFoundError:
        return "PowerShell not available — cannot query Event Log."
    except OSError as exc:
        return f"Event log query failed: {exc}"

    out = proc.stdout.strip()
    if proc.returncode != 0 and not out:
        err = (proc.stderr or "").strip().splitlines()
        return "Event log query returned no data" + (
            f" (stderr: {err[0]})" if err else "."
        )
    if not out:
        return f"No {sel} events in the last 24 hours."
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        return f"Could not parse event log output: {exc}"

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return "Event log returned unexpected shape."

    lines = []
    for e in data:
        t = e.get("time", "??:??")
        lvl = e.get("level", "?")
        src = e.get("source", "?")
        eid = e.get("id", "?")
        msg = (e.get("msg") or "").replace("\r\n", " ").replace("\n", " ")
        lines.append(f"{t} {lvl} {src} ({eid}): {msg}")
    return "\n".join(lines)


def _do_network(limit: int) -> str:
    """Active interface IPs + throughput. Skip loopback and inactive
    interfaces — they add noise without value for voice."""
    try:
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
    except OSError as exc:
        return f"Network info unavailable: {exc}"

    lines: list[str] = []
    for iface, addr_list in addrs.items():
        st = stats.get(iface)
        if st is None or not st.isup:
            continue
        # Skip loopback regardless of the interface name (varies across
        # Windows builds): it's the only interface that resolves to 127.x.
        # Pick the first non-loopback IPv4 — Windows interfaces commonly
        # carry IPv6 link-local + APIPA addresses we don't want for voice.
        ipv4 = next(
            (
                a.address
                for a in addr_list
                if a.family == socket.AF_INET
                and not (a.address or "").startswith("127.")
            ),
            None,
        )
        if not ipv4:
            continue
        speed = f"{st.speed} Mbps" if st.speed else "unknown speed"
        lines.append(f"{iface}: {ipv4} ({speed})")

    # Aggregate I/O counters since boot.
    try:
        io = psutil.net_io_counters()
        lines.append(
            f"Total since boot: {_bytes_to_human(io.bytes_recv)} down, "
            f"{_bytes_to_human(io.bytes_sent)} up"
        )
    except OSError:
        pass

    if not lines:
        return "No active network interfaces."
    return "\n".join(lines[:limit])


def execute_pc_diagnostics_tool(params: dict) -> str:
    """Run the tool. Always returns a string — never raises."""
    mode = (params.get("mode") or "overview").lower().strip()
    sort_by = (params.get("sort_by") or "cpu").lower().strip()
    filter_str = (params.get("filter") or "").strip()
    limit = _clamp_limit(params.get("limit"))

    try:
        if mode == "overview":
            return _do_overview()
        if mode == "cpu":
            return _do_cpu(limit)
        if mode == "memory":
            return _do_memory(limit)
        if mode == "disk":
            return _do_disk()
        if mode == "processes":
            return _do_processes(sort_by, limit)
        if mode == "services":
            return _do_services(filter_str, limit)
        if mode == "events":
            return _do_events(filter_str, limit)
        if mode == "network":
            return _do_network(limit)
        return f"Unknown mode '{mode}'."
    except Exception as exc:
        # Defensive net — psutil errors on edge cases (Hyper-V containers,
        # inaccessible processes mid-iter). Don't let a tool exception kill
        # the turn.
        print(f"[pc_diagnostics] {mode} raised: {exc}", file=sys.stderr)
        return f"Diagnostics error in mode '{mode}': {exc}"
