"""pc_shell — read-only Windows shell tool for SRE-grade investigation.

The "let Jarvis actually investigate" upgrade. The natural companion to
M28's read_local_file (which reads a specific file): pc_shell runs a
specific *command* from a strict allowlist. Together they give Claude
read-only diagnostic depth on the live host instead of forcing every
question through the static pc_diagnostics snapshot.

  ipconfig                show interface configuration (optional /all, /displaydns)
  Get-NetAdapter          list network adapters and their link state
  Get-NetRoute            show the IP routing table (optional family filter)
  Get-NetTCPConnection    list TCP connections (optional state/port filter)
  netstat                 classic TCP/UDP listing (flags allowlisted)
  arp                     show the ARP cache (-a)
  ping                    connectivity test (count 1-10)
  tracert                 path trace (max_hops 1-30)
  Test-NetConnection      modern ping + optional port test
  Resolve-DnsName         DNS lookup (record type optional)
  nslookup                classic DNS lookup with optional server
  Get-Process             list processes (optional name filter)
  Get-Service             list services (optional name/status filter)
  Get-WinEvent            recent Windows event-log entries
  Get-CimInstance         WMI/CIM query against a small class allowlist
  systeminfo              one-shot host summary
  Get-ChildItem           directory listing (path validated, no recurse)
  Get-HotFix              installed Windows updates

Security posture — every choice in this module exists for a reason:

1. **No arbitrary command path.** A `command` enum gates dispatch; an
   unknown value short-circuits before any process spawns. There is no
   "free-form command" parameter. Adding a verb is a code change, not a
   prompt change.
2. **No shell composition, ever.** Every command goes through
   `subprocess.run([list, of, args])` with `shell=False`. No f-strings
   into shell commands, no os.system(), no Popen(string, shell=True). A
   user-controlled hostname like "google.com; rm -rf /" can never reach
   a shell parser because there is no shell parser in this code path.
3. **Strict per-verb input validation.** Hostnames match a hostname
   regex; ports are int + range-checked; class names match a small CIM
   allowlist; paths get a denylist + traversal check. Validation IS the
   security boundary; the subprocess invocation just consumes already-
   validated values.
4. **Read-only by design.** No verb in the allowlist mutates host
   state. ipconfig's /release, /renew, /flushdns and /registerdns are
   deliberately NOT exposed — those belong in system_control if/when
   wanted, behind confirmation gates. Per the diag-vs-action memory:
   read tools and write tools stay separate.
5. **Bounded.** 15s hard timeout per invocation (subprocess.run
   timeout=15), output truncated to 8 KB / 200 lines (whichever is
   smaller). Runaway `ping -t` or a `Get-WinEvent` that streams
   millions of records can't blow up the context window or hang the
   agentic loop.
6. **Audited.** Every invocation logs to stderr (→ jarvis.log) with
   verb, args, exit code, and output size. "What investigative commands
   has Jarvis run lately?" is one `grep '[pc_shell]'` away.
7. **Defensive contract.** Like every other tool in src/, never raises.
   subprocess errors, timeouts, missing executables — all become
   readable strings.

What pc_shell deliberately is NOT:
- An interactive REPL — every call is one-shot.
- A way to run arbitrary PowerShell — `Invoke-Expression`, `iex`,
  `&{...}`, dot-sourcing are all unreachable.
- A replacement for read_local_file — Get-ChildItem LISTS files;
  read_local_file READS file contents. Complementary, not overlapping.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


# --- Anthropic tool definition ---------------------------------------------
# Single tool with `command` enum + flat `args` object. Claude doesn't
# reliably follow oneOf/anyOf conditional schemas in tool input, so per-
# command parameter shape is documented in the description and enforced
# inside the executor — same pattern as system_control. Optional args
# unused by a given verb are simply ignored, not an error.
PC_SHELL_TOOL = {
    "name": "pc_shell",
    "description": (
        "Run ONE command from a fixed, read-only Windows investigation "
        "allowlist on THIS PC. Use to investigate dynamically when "
        "pc_diagnostics isn't specific enough — ping a host, look up DNS, "
        "check a service, search the event log, inspect a TCP connection, "
        "list processes by name, etc. Always read-only — this tool cannot "
        "modify any system state. Output is truncated; invoke again with "
        "different args if you need more depth. Per-command argument "
        "shapes (pass via `args` object):\n"
        "  ipconfig:            verbose:bool, display_dns:bool\n"
        "  Get-NetAdapter:      (no args)\n"
        "  Get-NetRoute:        family:'IPv4'|'IPv6' (optional)\n"
        "  Get-NetTCPConnection: state:'Listen'|'Established'|'TimeWait' (optional), local_port:int (optional)\n"
        "  netstat:             flags:str (subset of 'anob', e.g. 'ano')\n"
        "  arp:                 (no args)\n"
        "  ping:                target:str (required), count:int 1-10 (optional, default 4)\n"
        "  tracert:             target:str (required), max_hops:int 1-30 (optional)\n"
        "  Test-NetConnection:  target:str (required), port:int 1-65535 (optional)\n"
        "  Resolve-DnsName:     target:str (required), type:'A'|'AAAA'|'CNAME'|'MX'|'NS'|'TXT' (optional)\n"
        "  nslookup:            target:str (required), server:str (optional)\n"
        "  Get-Process:         name:str (optional)\n"
        "  Get-Service:         name:str (optional), status:'Running'|'Stopped' (optional)\n"
        "  Get-WinEvent:        log_name:str (required, e.g. 'System', 'Application'), max_events:int 1-200 (optional, default 20), level:'Error'|'Warning'|'Information' (optional)\n"
        "  Get-CimInstance:     class_name:str (required, from allowlist: Win32_OperatingSystem, Win32_ComputerSystem, Win32_PhysicalMemory, Win32_LogicalDisk, Win32_Processor, Win32_NetworkAdapter, Win32_BIOS, Win32_VideoController)\n"
        "  systeminfo:          (no args)\n"
        "  Get-ChildItem:       path:str (required, no recursion)\n"
        "  Get-HotFix:          (no args)\n"
        "Briefly announce what you're checking, then call — no confirmation needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": [
                    "ipconfig", "Get-NetAdapter", "Get-NetRoute",
                    "Get-NetTCPConnection", "netstat", "arp",
                    "ping", "tracert", "Test-NetConnection",
                    "Resolve-DnsName", "nslookup",
                    "Get-Process", "Get-Service", "Get-WinEvent",
                    "Get-CimInstance", "systeminfo",
                    "Get-ChildItem", "Get-HotFix",
                ],
            },
            "args": {
                "type": "object",
                "description": (
                    "Command-specific arguments. See the tool description "
                    "for the shape per command. Unused fields are ignored."
                ),
                "additionalProperties": True,
            },
        },
        "required": ["command"],
    },
}


# --- Bounds ----------------------------------------------------------------
# 20s wall clock per call. Generous enough for the slowest realistic case
# (tracert at max_hops=30 with -w 1500 ≈ 45s worst-case theoretically, but
# real blackhole traces typically settle in 10-15s; ping -n 10 ≈ 10s). The
# original "Test-NetConnection cmdlet at 30-40s" worst case was dodged by
# replacing _h_test_netconnection's PS body with a TcpClient probe that
# returns in <3s — see that handler's docstring for the rationale. 20s ×
# 5 agentic iterations = 100s worst-case turn budget, comfortable for voice.
_TIMEOUT_SECONDS = 20

# 8 KB / 200 lines is enough for a netstat snapshot or a 20-event log dump.
# Claude can re-invoke with narrower filters (specific port, specific log
# level) if it needs more.
_MAX_OUTPUT_BYTES = 8 * 1024
_MAX_OUTPUT_LINES = 200

# Common PowerShell preamble — no profile (faster, deterministic), no
# interactive prompts, no execution-policy bypass (we never run scripts
# from disk; everything is -Command with inline statements).
_PS_BASE = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]


# --- Input validation ------------------------------------------------------
# Hostname/IP validator: alphanumeric, dots, hyphens, colons (for IPv6), and
# percent (for IPv6 zone IDs). Rejects ANY shell metacharacter — ; & | ` $ <
# > * ? ( ) [ ] { } ' " \ space — which is the actual safety property we
# want. We trust the OS resolver to decide whether the result is a valid
# hostname; this regex only enforces "no injection surface".
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9._%:-]+$")

# Process names: alphanumeric, dot, underscore, dash, asterisk (wildcard).
# Both 'chrome' and 'chrome.exe' and 'chrome*' valid.
_PROCESS_NAME_RE = re.compile(r"^[A-Za-z0-9._*-]+$")

# Event-log names. Standard ones: System, Application, Security, Setup. Also
# allow "Microsoft-Windows-*" channel names (alphanumeric + dash + slash +
# space — slash for "Microsoft-Windows-Diagnostics/Operational" style).
_LOG_NAME_RE = re.compile(r"^[A-Za-z0-9 /_-]+$")

# CIM class allowlist. WMI is enormous; an open class_name parameter is a
# DoS vector (some classes are slow + huge). The allowlist covers the
# classes Claude actually reaches for in SRE investigation. Add as needed.
_CIM_CLASS_ALLOWLIST = {
    "Win32_OperatingSystem", "Win32_ComputerSystem", "Win32_PhysicalMemory",
    "Win32_LogicalDisk", "Win32_Processor", "Win32_NetworkAdapter",
    "Win32_BIOS", "Win32_VideoController",
}

# Path traversal denylist for Get-ChildItem. Read-only LISTING is far less
# sensitive than read_local_file's content reads — file names alone are
# rarely secret. We still refuse the same private-key locations defensively;
# the rest of the filesystem is fair game for a directory listing.
_PATH_DENY_PREFIXES = {
    # User credential stores
    ".ssh", "credentials",
    # Windows registry hives  (full path matches)
}


def _validate_target(value: Any, field: str = "target") -> tuple[str | None, str | None]:
    """Return (validated_value, error_string). Either is None.

    The validation IS the security boundary against shell injection. After
    this returns a clean string, the subprocess argv can safely contain it.
    """
    if not isinstance(value, str) or not value.strip():
        return None, f"{field} must be a non-empty string."
    v = value.strip()
    if len(v) > 253:
        return None, f"{field} too long (max 253 chars)."
    if not _HOSTNAME_RE.match(v):
        return None, (
            f"{field} contains characters I won't pass through "
            f"(only letters, digits, dots, hyphens, colons, percent allowed)."
        )
    return v, None


def _validate_int(value: Any, field: str, lo: int, hi: int) -> tuple[int | None, str | None]:
    if value is None:
        return None, None  # absent is fine; caller decides if required
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None, f"{field} must be an integer."
    if v < lo or v > hi:
        return None, f"{field} must be between {lo} and {hi}."
    return v, None


# --- Subprocess execution + output handling --------------------------------

def _run(argv: list[str], log_label: str) -> str:
    """Execute argv with timeout + size cap; return formatted text.

    All verbs funnel through this. `argv` is a list — `shell=False` is the
    default and we don't override it. The audit log entry is emitted in
    finally so a timeout/exception still leaves a trace.
    """
    exe_name = argv[0]
    # Resolve once for the audit log so "command not found" reads better than
    # WinError 2.
    if shutil.which(exe_name) is None and not Path(exe_name).exists():
        return f"{exe_name} is not available on this system."

    truncated = False
    output = ""
    exit_code: int | str = "?"
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TIMEOUT_SECONDS,
            shell=False,
        )
        exit_code = proc.returncode
        # Merge stdout + stderr — PS writes warnings/non-fatal errors to
        # stderr but Claude still wants to see them ("Test-NetConnection:
        # Name resolution failed" is the meaningful part of a failed call).
        combined = (proc.stdout or "") + (proc.stderr or "")
        output, truncated = _truncate(combined)
    except subprocess.TimeoutExpired:
        return f"{log_label} timed out after {_TIMEOUT_SECONDS}s."
    except OSError as exc:
        return f"{log_label} failed to start: {exc}"
    finally:
        # Audit BEFORE returning so even errored invocations leave a trace.
        # argv[0] is the exe; argv[1:] is the parameter list — useful in the
        # log for "what exactly did Jarvis ask for here".
        print(
            f"[pc_shell] {log_label} argv={argv[1:]} exit={exit_code} "
            f"bytes={len(output)}{' TRUNCATED' if truncated else ''}",
            file=sys.stderr,
        )

    if not output.strip():
        return f"{log_label} returned no output (exit code {exit_code})."
    if truncated:
        output += (
            f"\n\n[output truncated to {_MAX_OUTPUT_BYTES} bytes / "
            f"{_MAX_OUTPUT_LINES} lines — re-run with narrower filters "
            f"if you need more]"
        )
    return output


def _truncate(text: str) -> tuple[str, bool]:
    """Cap text by both byte size AND line count, whichever hits first.

    Line cap matters for `Get-WinEvent` or `netstat -ano` which produce
    many short lines; byte cap matters for `systeminfo` or `Get-CimInstance`
    which produce fewer but longer ones. Either alone leaves a gap.
    """
    if not text:
        return text, False
    lines = text.splitlines()
    truncated = False
    if len(lines) > _MAX_OUTPUT_LINES:
        lines = lines[:_MAX_OUTPUT_LINES]
        truncated = True
    capped = "\n".join(lines)
    if len(capped.encode("utf-8")) > _MAX_OUTPUT_BYTES:
        # Trim from the end. Decode in case we cut a multi-byte char.
        capped = capped.encode("utf-8")[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
        truncated = True
    return capped, truncated


# --- Per-verb handlers -----------------------------------------------------
# Each takes the args dict and returns text. Validation happens FIRST; only
# already-validated values flow into argv.
#
# Native-exe verbs (ping, tracert, nslookup, arp, netstat, ipconfig,
# systeminfo) get clean argv-list passing via subprocess — Windows'
# CommandLineToArgvW tokenizes each element separately, no shell parser
# touches the input.
#
# PowerShell verbs are trickier: `powershell.exe -Command "<text>"` does NOT
# bind any tokens after the command string to `$args` (a common mistake — I
# made it on first pass and the smoke test caught it). The reliable pattern
# is to INLINE validated values into a SINGLE-QUOTED PS string. Single
# quotes in PowerShell are literal: no `$` expansion, no operator parsing,
# only `''` escapes a literal apostrophe — and every regex we validate
# strings against (_HOSTNAME_RE, _PROCESS_NAME_RE, _LOG_NAME_RE) explicitly
# REFUSES the apostrophe character. So the validated value can never break
# out of the surrounding quotes. Validation is the security boundary;
# inlining is just composition after the boundary has been crossed.

def _h_ipconfig(args: dict) -> str:
    argv = ["ipconfig"]
    if args.get("verbose"):
        argv.append("/all")
    elif args.get("display_dns"):
        argv.append("/displaydns")
    return _run(argv, "ipconfig")


def _h_get_netadapter(_args: dict) -> str:
    return _run(_PS_BASE + ["Get-NetAdapter | Format-Table -AutoSize"], "Get-NetAdapter")


def _h_get_netroute(args: dict) -> str:
    family = args.get("family")
    if family is not None and family not in ("IPv4", "IPv6"):
        return "family must be 'IPv4' or 'IPv6' (or omitted)."
    if family:
        cmd = f"Get-NetRoute -AddressFamily {family} | Format-Table -AutoSize"
    else:
        cmd = "Get-NetRoute | Format-Table -AutoSize"
    return _run(_PS_BASE + [cmd], "Get-NetRoute")


def _h_get_nettcpconnection(args: dict) -> str:
    state = args.get("state")
    if state is not None and state not in ("Listen", "Established", "TimeWait", "CloseWait", "SynSent"):
        return "state must be one of Listen, Established, TimeWait, CloseWait, SynSent."
    port, err = _validate_int(args.get("local_port"), "local_port", 1, 65535)
    if err:
        return err

    parts = ["Get-NetTCPConnection"]
    if state:
        parts.append(f"-State {state}")
    if port is not None:
        parts.append(f"-LocalPort {port}")
    parts.append("| Format-Table -AutoSize")
    return _run(_PS_BASE + [" ".join(parts)], "Get-NetTCPConnection")


def _h_netstat(args: dict) -> str:
    flags = args.get("flags", "ano")
    if not isinstance(flags, str):
        return "flags must be a string."
    # netstat flags are single letters. Allowlist exactly the SRE-useful ones.
    if not re.fullmatch(r"[anob]+", flags):
        return "flags must be a subset of 'anob' (e.g. 'ano', 'an')."
    return _run(["netstat", f"-{flags}"], f"netstat -{flags}")


def _h_arp(_args: dict) -> str:
    return _run(["arp", "-a"], "arp -a")


def _h_ping(args: dict) -> str:
    target, err = _validate_target(args.get("target"))
    if err:
        return err
    count, err = _validate_int(args.get("count"), "count", 1, 10)
    if err:
        return err
    if count is None:
        count = 4
    return _run(["ping", "-n", str(count), target], f"ping {target}")


def _h_tracert(args: dict) -> str:
    target, err = _validate_target(args.get("target"))
    if err:
        return err
    max_hops, err = _validate_int(args.get("max_hops"), "max_hops", 1, 30)
    if err:
        return err
    argv = ["tracert"]
    if max_hops is not None:
        argv += ["-h", str(max_hops)]
    # -d skips reverse DNS (faster, plenty for an investigative trace).
    # -w 1500 caps each hop's wait at 1.5s so a fully-blackholed trace at
    # max_hops=30 stays inside the 30s subprocess timeout (default -w is
    # 4000ms which would push past the budget).
    argv += ["-d", "-w", "1500", target]
    return _run(argv, f"tracert {target}")


def _h_test_netconnection(args: dict) -> str:
    """Fast TCP/ICMP probe.

    NOT the cmdlet of the same name — the PS `Test-NetConnection` cmdlet
    does TCP + ICMP + reverse-DNS + (sometimes) route-trace in a single
    call, which clocks in at ~30-40s on hosts with slow ICMP or rDNS
    paths (seen on this dev host during M33 smoke testing). For voice/
    agentic use the rich output is rarely worth the wait — "is the port
    open?" should answer in seconds, not half a minute.

    With port: `System.Net.Sockets.TcpClient.ConnectAsync` with a 3s
    wall-clock timeout. Returns clean SUCCESS / FAILED text.

    Without port: `Test-Connection` (the ICMP-only cmdlet) with 2
    pings — fast and well-bounded. ICMP-blocked hosts come back as a
    clear failure rather than hanging.
    """
    target, err = _validate_target(args.get("target"))
    if err:
        return err
    port, err = _validate_int(args.get("port"), "port", 1, 65535)
    if err:
        return err

    if port is not None:
        # Multi-line PS literal: single-quoted target survives interpolation;
        # the regex already forbade apostrophes so it can't break out.
        cmd = (
            f"$c = New-Object System.Net.Sockets.TcpClient; "
            f"try {{ "
            f"  $task = $c.ConnectAsync('{target}', {port}); "
            f"  if ($task.Wait(3000) -and $c.Connected) {{ "
            f"    'TCP {target}:{port} - SUCCESS' "
            f"  }} else {{ "
            f"    'TCP {target}:{port} - FAILED (timeout or refused after 3s)' "
            f"  }} "
            f"}} catch {{ "
            f"  \"TCP {target}:{port} - ERROR: $($_.Exception.Message)\" "
            f"}} finally {{ $c.Close() }}"
        )
        return _run(_PS_BASE + [cmd], f"tcp-probe {target}:{port}")

    # ICMP-only path: Test-Connection is the fast cmdlet (Test-NetConnection
    # without -Port still calls into the slow probe chain). 2 pings, terse
    # output via -Quiet (returns True/False) — wrapped to emit clean text.
    cmd = (
        f"if (Test-Connection -ComputerName '{target}' -Count 2 -Quiet) "
        f"{{ 'ICMP {target} - SUCCESS' }} else "
        f"{{ 'ICMP {target} - FAILED (no reply or host unreachable)' }}"
    )
    return _run(_PS_BASE + [cmd], f"icmp-probe {target}")


def _h_resolve_dnsname(args: dict) -> str:
    target, err = _validate_target(args.get("target"))
    if err:
        return err
    rtype = args.get("type")
    if rtype is not None and rtype not in ("A", "AAAA", "CNAME", "MX", "NS", "TXT", "PTR", "SOA"):
        return "type must be one of A, AAAA, CNAME, MX, NS, TXT, PTR, SOA."
    if rtype:
        cmd = f"Resolve-DnsName -Name '{target}' -Type {rtype}"
    else:
        cmd = f"Resolve-DnsName -Name '{target}'"
    return _run(_PS_BASE + [cmd], f"Resolve-DnsName {target}")


def _h_nslookup(args: dict) -> str:
    target, err = _validate_target(args.get("target"))
    if err:
        return err
    argv = ["nslookup", target]
    server = args.get("server")
    if server:
        srv_clean, err = _validate_target(server, "server")
        if err:
            return err
        argv.append(srv_clean)
    return _run(argv, f"nslookup {target}")


def _h_get_process(args: dict) -> str:
    name = args.get("name")
    if name is not None:
        if not isinstance(name, str) or not _PROCESS_NAME_RE.match(name):
            return "name must be alphanumeric (dots/underscores/dashes/asterisks ok)."
        cmd = f"Get-Process -Name '{name}' | Format-Table -AutoSize"
        return _run(_PS_BASE + [cmd], f"Get-Process {name}")
    return _run(_PS_BASE + ["Get-Process | Format-Table -AutoSize"], "Get-Process")


def _h_get_service(args: dict) -> str:
    name = args.get("name")
    status = args.get("status")
    if status is not None and status not in ("Running", "Stopped", "Paused", "Starting", "Stopping"):
        return "status must be Running, Stopped, Paused, Starting, or Stopping."
    if name is not None and (not isinstance(name, str) or not _PROCESS_NAME_RE.match(name)):
        return "name must be alphanumeric (dots/underscores/dashes/asterisks ok)."

    pipeline = ["Get-Service"]
    if name:
        pipeline.append(f"-Name '{name}'")
    if status:
        pipeline.append(f"| Where-Object Status -eq '{status}'")
    pipeline.append("| Format-Table -AutoSize")
    cmd = " ".join(pipeline)
    return _run(_PS_BASE + [cmd], "Get-Service")


def _h_get_winevent(args: dict) -> str:
    log_name = args.get("log_name")
    if not isinstance(log_name, str) or not log_name or not _LOG_NAME_RE.match(log_name):
        return (
            "log_name required (e.g. 'System', 'Application', 'Security'). "
            "Allowed chars: alphanumerics, space, /, _, -."
        )
    max_events, err = _validate_int(args.get("max_events"), "max_events", 1, 200)
    if err:
        return err
    if max_events is None:
        max_events = 20
    level = args.get("level")
    level_map = {"Critical": 1, "Error": 2, "Warning": 3, "Information": 4}
    if level is not None and level not in level_map:
        return "level must be Critical, Error, Warning, or Information."

    # FilterHashtable is the SAFE Get-WinEvent invocation — keys are a
    # fixed enum, values flow in as already-validated literals.
    # log_name regex forbids quotes; inlining in single quotes is safe.
    filter_parts = [f"LogName='{log_name}'"]
    if level is not None:
        filter_parts.append(f"Level={level_map[level]}")
    filter_str = "; ".join(filter_parts)
    cmd = (
        f"Get-WinEvent -FilterHashtable @{{{filter_str}}} -MaxEvents {max_events} "
        f"-ErrorAction Stop | "
        f"Format-Table TimeCreated, Id, LevelDisplayName, ProviderName, Message "
        f"-Wrap -AutoSize"
    )
    return _run(_PS_BASE + [cmd], f"Get-WinEvent {log_name}")


def _h_get_ciminstance(args: dict) -> str:
    class_name = args.get("class_name")
    if not isinstance(class_name, str) or class_name not in _CIM_CLASS_ALLOWLIST:
        allowed = ", ".join(sorted(_CIM_CLASS_ALLOWLIST))
        return f"class_name must be one of: {allowed}."
    cmd = f"Get-CimInstance -ClassName {class_name} | Format-List *"
    return _run(_PS_BASE + [cmd], f"Get-CimInstance {class_name}")


def _h_systeminfo(_args: dict) -> str:
    return _run(["systeminfo"], "systeminfo")


def _h_get_childitem(args: dict) -> str:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return "path required (an absolute or ~-relative directory path)."
    try:
        p = Path(raw_path).expanduser()
    except (ValueError, RuntimeError) as exc:
        return f"Bad path '{raw_path}': {exc}"
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    if not resolved.exists():
        return f"No such path: {resolved}"
    if not resolved.is_dir():
        return f"Not a directory: {resolved}. Use read_local_file for files."

    # Light denylist — refuse listing locations whose mere existence/contents
    # leak credential material (filenames are themselves the signal). The
    # rest of the filesystem is fair game for read-only enumeration.
    name_lower = resolved.name.lower()
    if name_lower in _PATH_DENY_PREFIXES:
        return (
            f"Refusing to list {resolved} — that directory typically "
            f"contains credential material. Name a more specific path."
        )

    # No recurse, no filtering — keep this primitive. Claude can re-invoke
    # with a nested path if needed. Format-Table for compact output; -Force
    # so hidden + system items aren't silently dropped (which would confuse
    # diagnostic queries like "what's in C:\ProgramData\<vendor>").
    # Paths can legally contain apostrophes (rare but legal in NTFS) — refuse
    # rather than escape; far simpler than running the PS doubled-quote dance.
    path_str = str(resolved)
    if "'" in path_str:
        return f"Refusing to list a path containing apostrophes: {resolved}"
    cmd = f"Get-ChildItem -Path '{path_str}' -Force | Format-Table Mode, LastWriteTime, Length, Name -AutoSize"
    return _run(_PS_BASE + [cmd], f"Get-ChildItem {resolved}")


def _h_get_hotfix(_args: dict) -> str:
    return _run(_PS_BASE + ["Get-HotFix | Format-Table -AutoSize"], "Get-HotFix")


# --- Dispatch --------------------------------------------------------------
# Single source of truth: command name → handler. Order here is the order
# Claude sees in error messages — keep it semantic (network first, then
# host, then files/updates).
_HANDLERS: dict[str, Callable[[dict], str]] = {
    "ipconfig":            _h_ipconfig,
    "Get-NetAdapter":      _h_get_netadapter,
    "Get-NetRoute":        _h_get_netroute,
    "Get-NetTCPConnection": _h_get_nettcpconnection,
    "netstat":             _h_netstat,
    "arp":                 _h_arp,
    "ping":                _h_ping,
    "tracert":             _h_tracert,
    "Test-NetConnection":  _h_test_netconnection,
    "Resolve-DnsName":     _h_resolve_dnsname,
    "nslookup":            _h_nslookup,
    "Get-Process":         _h_get_process,
    "Get-Service":         _h_get_service,
    "Get-WinEvent":        _h_get_winevent,
    "Get-CimInstance":     _h_get_ciminstance,
    "systeminfo":          _h_systeminfo,
    "Get-ChildItem":       _h_get_childitem,
    "Get-HotFix":          _h_get_hotfix,
}


def execute_pc_shell(params: dict) -> str:
    """Run the tool. Always returns a string — never raises.

    Dispatch is gated on the `command` enum literally being in _HANDLERS;
    a value not in the enum (or a hallucinated one) returns a clear error
    listing the valid commands rather than attempting anything.
    """
    command = (params.get("command") or "").strip()
    args = params.get("args") or {}
    if not isinstance(args, dict):
        return "`args` must be an object."

    handler = _HANDLERS.get(command)
    if handler is None:
        allowed = ", ".join(_HANDLERS.keys())
        return f"Unknown command '{command}'. Allowed: {allowed}."

    try:
        return handler(args)
    except Exception as exc:  # noqa: BLE001 — defensive last-resort net
        print(f"[pc_shell] '{command}' raised: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return f"pc_shell error in '{command}': {exc}"
