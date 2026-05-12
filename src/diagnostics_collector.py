"""run_pc_diagnostics_collector — deep Windows diagnostic snapshot.

Wraps the user's standalone collector script
(`~/repos/hs-windows-diagnostics/Invoke-HSWindowsDiagnostics.ps1`) — a
PowerShell port of work's `diagnostics.bash` that gathers host /
security / package / event-log telemetry into a `.tar.gz` of ~47 small text
files. This tool runs it, extracts the archive, and hands Claude a file
listing so it can then `read_local_file` the specific bundle files relevant
to the user's question.

Relationship to the M23 `pc_diagnostics` tool:
  - `pc_diagnostics` = "what's happening RIGHT NOW" — live psutil/PowerShell
    queries, no disk artifacts, sub-second.
  - this tool = "deep snapshot, archive, follow-up analysis" — writes ~MB to
    disk, takes 60-90s (quick mode) to a few minutes (full), produces a
    browsable bundle. Keep both; different questions.

Confirmation-gated (the M23 `kill_process` pattern, enforced HERE not just in
the system prompt): the first call without `confirmed=true` returns a "needs
confirmation" string Claude paraphrases to the user; only after the user
agrees does Claude call again with `confirmed=true`. The bundle isn't
destructive, but it writes hundreds of KB and takes a minute-plus — worth a
one-line "Confirm: run full diagnostics?" first.

No startup wiring — like `pc_diagnostics`' Get-WinEvent call, this just shells
out when invoked. If the collector script isn't installed, the tool says so
plainly rather than crashing.

Defensive contract, same as the rest of the toolkit: never raises, always
returns a readable string.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from src.memory import default_base_dir


# --- Anthropic tool definition ---------------------------------------------
RUN_PC_DIAGNOSTICS_COLLECTOR_TOOL = {
    "name": "run_pc_diagnostics_collector",
    "description": (
        "Run a DEEP diagnostic snapshot of THIS Windows PC: collects host info "
        "(OS, hardware, network, processes, services, users, scheduled tasks), "
        "security posture (Defender, BitLocker status, firewall, UAC, Secure "
        "Boot, TPM), installed packages and Windows updates, and recent event "
        "logs — into a bundle of small text files. Returns the bundle's "
        "location plus a file listing; use read_local_file afterward to read "
        "the specific files relevant to the question. "
        "Use this for 'run a full diagnostic', 'deep system check', 'collect "
        "everything for a support ticket', or follow-up troubleshooting that "
        "needs more than the live pc_diagnostics snapshot. "
        "IMPORTANT: this writes hundreds of KB to disk and takes 60-90 seconds "
        "(longer in full mode). You MUST first ask the user to confirm in plain "
        "language ('Confirm: run a full diagnostics collection? It takes about "
        "a minute.'), wait for their explicit yes, THEN call with confirmed=true."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "quick": {
                "type": "boolean",
                "description": (
                    "true (default) = quick mode: skips the slow collectors "
                    "(firewall-rule enumeration), caps event logs at 500/log — "
                    "finishes in ~60-90s, right for ad-hoc troubleshooting. "
                    "false = full collection: everything, up to 2000 events/log "
                    "— a few minutes, right for a support-ticket bundle."
                ),
            },
            "confirmed": {
                "type": "boolean",
                "description": (
                    "Required true to actually run. Only set this AFTER the "
                    "user has explicitly confirmed in conversation. Without it "
                    "the tool returns a confirmation-required notice and does "
                    "nothing."
                ),
            },
        },
        "required": [],
    },
}


# Where the collector script lives by default — follows the user's "code in
# ~/repos/<name>" convention. Overridable via the DIAGNOSTICS_COLLECTOR_PATH
# env var (read at call time, like games.py's RAWG_API_KEY — it's consumed
# only here, so no need to thread it through Config).
DEFAULT_COLLECTOR_SCRIPT = (
    Path.home() / "repos" / "hs-windows-diagnostics" / "Invoke-HSWindowsDiagnostics.ps1"
)

# Output goes under %LOCALAPPDATA%\Jarvis\diagnostics\ — alongside the log and
# the memory store, not in the script's own default (%LOCALAPPDATA%\HS\...),
# so Jarvis owns the lifecycle (extraction + pruning) of what it produced.
def _diag_dir() -> Path:
    d = default_base_dir() / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Keep this many recent bundles (archive + extracted dir); prune older ones
# before each new run. Bundles are KB-to-low-MB so this is generous.
_KEEP_BUNDLES = 3

# Subprocess timeouts. Quick mode targets <3 min; full can run several. Pad
# generously — a timeout here means a half-written bundle, which is worse than
# waiting a bit longer.
_QUICK_TIMEOUT_SEC = 300
_FULL_TIMEOUT_SEC = 600

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _resolve_script_path() -> Path:
    override = os.getenv("DIAGNOSTICS_COLLECTOR_PATH", "").strip()
    return Path(override).expanduser() if override else DEFAULT_COLLECTOR_SCRIPT


def _subprocess_env() -> dict[str, str]:
    """Copy of the current environment with %SystemRoot%\\System32 pushed to
    the FRONT of PATH. The collector script shells out to `tar` for .tar.gz
    packaging; we need Windows' bsdtar (System32\\tar.exe), not whatever else
    might shadow it on PATH — notably MSYS2/Git-Bash's GNU `tar`, which mangles
    Windows-style absolute paths and exits 2. If Jarvis was launched from a
    Git Bash terminal, that GNU tar would be first on PATH without this."""
    env = dict(os.environ)
    system32 = str(Path(env.get("SystemRoot", r"C:\Windows")) / "System32")
    parts = env.get("PATH", "").split(os.pathsep)
    if not parts or parts[0].lower() != system32.lower():
        env["PATH"] = os.pathsep.join([system32, *parts])
    return env


def _prune_old_bundles(diag_dir: Path) -> None:
    """Best-effort: keep the _KEEP_BUNDLES most-recent archives and their
    extracted siblings; delete the rest. Never raises."""
    try:
        archives = sorted(
            [p for ext in ("*.tar.gz", "*.zip") for p in diag_dir.glob(ext)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for old in archives[_KEEP_BUNDLES:]:
        try:
            extracted = old.parent / (_archive_stem(old) + "-extracted")
            old.unlink(missing_ok=True)
            if extracted.is_dir():
                _rmtree_quiet(extracted)
        except OSError:
            continue


def _rmtree_quiet(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001 — best-effort
        pass


def _archive_stem(archive: Path) -> str:
    """'hs-windiag-host-2026-05-12-093015.tar.gz' -> 'hs-windiag-host-...-093015'.
    Path.stem only strips one suffix, so '.tar.gz' needs two passes."""
    name = archive.name
    for suffix in (".tar.gz", ".tgz", ".zip"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return archive.stem


def _archives_in(diag_dir: Path) -> set[Path]:
    return {p for ext in ("*.tar.gz", "*.zip") for p in diag_dir.glob(ext)}


def _newest_archive(candidates: set[Path]) -> Path | None:
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract(archive: Path) -> tuple[Path | None, str | None]:
    """Extract `archive` to a sibling '<stem>-extracted' dir. Returns
    (extract_dir, error) — exactly one is non-None. Uses tarfile's 'data'
    filter (Python 3.12) to neutralize any path-traversal entries — the
    archive is self-produced and trusted, but defense in depth is cheap."""
    extract_dir = archive.parent / (_archive_stem(archive) + "-extracted")
    if extract_dir.exists():
        _rmtree_quiet(extract_dir)
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        if archive.name.lower().endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_dir)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                try:
                    tf.extractall(extract_dir, filter="data")  # py3.12+
                except TypeError:
                    tf.extractall(extract_dir)  # older Python — no filter kwarg
        return extract_dir, None
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return None, f"could not extract {archive.name}: {exc}"


def _list_bundle_files(extract_dir: Path) -> tuple[str, list[str]]:
    """Walk the extracted bundle and return (base_dir, sorted_relative_paths).

    The bundle commonly nests one level (hs-windiag-<host>-<ts>/host/os.txt,
    ...), so base_dir is the deepest common ancestor — keeping the per-file
    entries short ('host/cpu.txt', not a 190-char absolute path). The caller
    builds read_local_file paths as base_dir + os.sep + relative."""
    abs_files: list[str] = []
    for root, _dirs, files in os.walk(extract_dir):
        for fn in files:
            abs_files.append(str(Path(root) / fn))
    abs_files.sort()
    if not abs_files:
        return str(extract_dir), []
    base = os.path.commonpath(abs_files) if len(abs_files) > 1 else os.path.dirname(abs_files[0])
    rels = [os.path.relpath(f, base) for f in abs_files]
    return base, rels


def execute_run_pc_diagnostics_collector(params: dict) -> str:
    """Run the tool. Always returns a string — never raises."""
    quick = params.get("quick")
    quick = True if quick is None else bool(quick)
    confirmed = bool(params.get("confirmed"))

    mode = "quick" if quick else "full"

    if not confirmed:
        return (
            f"run_pc_diagnostics_collector requires explicit user confirmation. "
            f"Ask the user to confirm running a {mode} diagnostics collection "
            f"(it writes a bundle to disk and takes "
            f"{'about a minute' if quick else 'a few minutes'}), then call this "
            f"tool again with confirmed=true."
        )

    script = _resolve_script_path()
    if not script.is_file():
        return (
            f"Diagnostics collector script not found at {script}. "
            f"Expected it at ~/repos/hs-windows-diagnostics/ — clone or point "
            f"DIAGNOSTICS_COLLECTOR_PATH at it. (The live pc_diagnostics tool "
            f"still works for a quick snapshot.)"
        )

    try:
        diag_dir = _diag_dir()
    except OSError as exc:
        return f"Could not create the diagnostics output directory: {exc}"

    _prune_old_bundles(diag_dir)  # best-effort, never raises

    # Snapshot existing archives so we can identify the one THIS run produces
    # — guards against extracting a stale bundle from a prior run if the script
    # somehow succeeds without writing a new archive.
    archives_before = _archives_in(diag_dir)

    cmd = [
        "powershell.exe", "-NoProfile", "-NonInteractive",
        "-ExecutionPolicy", "Bypass",  # the script may be unsigned; scoped to this call only
        "-File", str(script),
        "-OutDir", str(diag_dir),
        "-NoSplit",     # always one archive — we extract it, never reassemble parts
        "-NoUnicode",   # ASCII-only progress UI → clean captured output
        "-MaxEvents", "500" if quick else "2000",
    ]
    if quick:
        cmd.append("-Quick")

    timeout = _QUICK_TIMEOUT_SEC if quick else _FULL_TIMEOUT_SEC
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
            env=_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        return (
            f"Diagnostics collection timed out after {timeout // 60} minutes. "
            f"The bundle may be incomplete. Try {mode} mode again, or use the "
            f"live pc_diagnostics tool."
        )
    except FileNotFoundError:
        return "PowerShell not available — cannot run the diagnostics collector."
    except OSError as exc:
        return f"Could not launch the diagnostics collector: {exc}"

    elapsed = time.monotonic() - started

    # The script exits 1 when it produced no output files; surface a stderr
    # tail so the user has something to act on.
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        hint = f" Last line: {tail[-1]}" if tail else ""
        print(
            f"[diagnostics_collector] script exited {proc.returncode} after {elapsed:.0f}s",
            file=sys.stderr,
        )
        return (
            f"The diagnostics collector exited with an error (code "
            f"{proc.returncode}) after {elapsed:.0f}s.{hint}"
        )

    new_archives = _archives_in(diag_dir) - archives_before
    archive = _newest_archive(new_archives)
    if archive is None:
        print("[diagnostics_collector] ran clean but no new archive found", file=sys.stderr)
        return (
            f"The collector ran for {elapsed:.0f}s but I couldn't find the "
            f"bundle it produced in {diag_dir}."
        )

    extract_dir, err = _extract(archive)
    if err is not None or extract_dir is None:
        print(f"[diagnostics_collector] extract failed: {err}", file=sys.stderr)
        return (
            f"Collected the bundle ({archive}) but couldn't extract it: {err}. "
            f"The archive is on disk if you want to open it manually."
        )

    base_dir, rel_files = _list_bundle_files(extract_dir)
    print(
        f"[diagnostics_collector] {mode} bundle: {archive.name}, "
        f"{len(rel_files)} files extracted, {elapsed:.0f}s",
        file=sys.stderr,
    )

    listing = "\n".join(f"  {r}" for r in rel_files) if rel_files else "  (no files?)"
    return (
        f"Diagnostics bundle collected ({mode} mode) in {elapsed:.0f}s.\n"
        f"Bundle root: {base_dir}\n"
        f"(Archive at {archive} if you want it intact.)\n\n"
        f"Files below are RELATIVE to the bundle root above — to read one, call "
        f"read_local_file with the bundle root + path separator + the relative "
        f"path (e.g. {os.path.join(base_dir, rel_files[0]) if rel_files else base_dir}):\n"
        f"{listing}\n\n"
        f"Orientation: events/ holds recent errors+warnings (system.txt, "
        f"application.txt, setup.txt, windows_update.txt, "
        f"diagnostics_performance.txt, security.txt); host/ has OS / hardware / "
        f"network / processes / services / users / scheduled tasks; security/ "
        f"has Defender, BitLocker status, firewall, UAC, Secure Boot, TPM, "
        f"audit policy, certs; packages/ has installed software and Windows "
        f"updates; BUNDLE_INFO.md has the run metadata. Read the files relevant "
        f"to the user's question, then summarize — don't read all of them."
    )
