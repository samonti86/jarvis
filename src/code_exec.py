"""Code execution tool — run Python in an isolated, throwaway sandbox (M50).

The flagship-risk tool, and the one designed against the project's
least-privilege spine the hardest. Every other powerful tool is
*allowlisted* — pc_shell is 18 fixed verbs, system_control is a fixed action
set. Code execution CANNOT be allowlisted: "run arbitrary code" is the whole
point; constrain *what* it does and it stops being a code interpreter.

So the security model shifts — you can't constrain what the code does, you
constrain WHERE it runs. The container is the security boundary:

  - **Ephemeral.** A fresh container per call (`podman run --rm`), destroyed
    the instant the run ends. Nothing persists between calls.
  - **No network** (`--network=none`). The code cannot exfiltrate, phone
    home, or pip-install anything — the image contents are the whole world.
  - **No host filesystem.** No bind mounts. The container cannot see, read,
    or write a single file on the user's machine (the M50 v1 decision —
    "fully isolated"; scoped mounts are a deliberate, confirmation-gated v2).
  - **Resource-capped** (`--memory`, `--cpus`, `--pids-limit`) + a hard
    wall-clock kill. A runaway — infinite loop, fork bomb, memory blow-up —
    cannot starve the host (the M44 OOM arc is why a hard memory cap on
    anything that runs code is non-negotiable).
  - **No new privileges** inside the container; runs as a non-root user.

Get the container right and arbitrary code is genuinely safe — the blast
radius is *zero*, the container is thrown away. That zero-blast-radius is
also why run_code needs NO confirmation prompt (unlike the M40 mutating
verbs): the M40 gate guards against unwanted *host* changes, and there is no
host effect to guard. The sandbox REPLACES the confirmation gate. Every run
is still logged to stderr/jarvis.log for the audit trail (the same
transparency mechanism pc_shell uses) — transparency, not gating.

run_code is in llm._RESTRICTED_DENY: a phone-origin turn (M48.2a) can never
trigger code execution, sandbox or not. The principle holds regardless of
the isolation.

Opt-in by dependency: there is no .env flag. If Podman isn't installed, the
tool simply reports "unavailable" — installing Podman IS the deliberate
opt-in.

Defensive contract, identical to the other tools: the public entry point
never raises and always returns a readable string. Podman missing, the
machine not running, the image not built, a timeout, a nonzero exit — all
become voice-friendly strings, never an exception into the listen loop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid


# --- Anthropic tool definition ---------------------------------------------
CODE_EXEC_TOOL = {
    "name": "run_code",
    "description": (
        "Execute Python code in an isolated, single-use sandbox container "
        "and get its output back. The sandbox is ephemeral (destroyed after "
        "each run), has NO network access, NO access to the user's files or "
        "machine, ships with Python 3.12 + numpy + pandas, and is killed "
        "after 30 seconds. Use this for genuine PROGRAMMING tasks no other "
        "tool covers: parsing or transforming data the user has given you, "
        "multi-step algorithmic or simulation work, generating structured "
        "output (CSV, JSON, tables). Do NOT use it for: a computational "
        "question with a clean answer (that is wolfram_query), looking up "
        "facts (web_search), or anything that needs to touch the user's "
        "actual files or system (the sandbox cannot see them — there is no "
        "tool for that yet). Write code that prints its result to stdout; "
        "you get stdout, stderr, and the exit code back, and you can fix "
        "and re-run if it errors."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "The Python 3.12 source to execute. Runs in a clean "
                    "sandbox with numpy + pandas available, NO network, and "
                    "NO access to the user's files. Print results to stdout "
                    "— stdout is what you get back."
                ),
            },
        },
        "required": ["code"],
    },
}


# The container runtime + image. Podman by deliberate M50 choice (rootless +
# daemonless = the cleaner posture for running untrusted code). To switch to
# Docker, change _RUNTIME to "docker" — the run / rm syntax used here is
# identical between the two.
_RUNTIME = "podman"
_IMAGE = "jarvis-code-sandbox"   # built from sandbox/Containerfile

# Sandbox limits — each is load-bearing, not decoration:
#   --network=none            code cannot exfiltrate or phone home
#   --memory / --cpus         a runaway cannot starve the host (cf. M44 OOM)
#   --pids-limit              fork-bomb ceiling
#   --security-opt=...        no privilege escalation inside the container
#   --rm                      the container is destroyed after the run
_MEM_LIMIT = "512m"
_CPU_LIMIT = "1"
_PIDS_LIMIT = "128"
_TIMEOUT_SEC = 30            # hard wall-clock kill
_MAX_CODE_LEN = 20_000       # generous; defends against absurd input
_MAX_OUTPUT_LEN = 8_000      # truncate huge output (matches pc_shell's cap)


# Jarvis is a long-lived, usually autostarted process. A winget install of
# Podman writes the new directory into the *registry* PATH — but a process
# already running (or restarted by the tray, which inherits the old
# process's environment) keeps a STALE os.environ['PATH'] and so cannot see
# `podman`: subprocess would raise FileNotFoundError even though Podman is
# installed and the machine is up. So resolve the runtime binary OURSELVES
# rather than trust PATH — try PATH, then a registry-refreshed PATH, then
# known install locations. Resolved once and cached; only an absolute hit is
# cached, so installing Podman mid-session is still picked up on a later call.
_runtime_path: str | None = None


def _resolve_runtime() -> str:
    """Locate the container-runtime executable, PATH-independently. Returns
    an absolute path when found, else the bare name (so the FileNotFoundError
    branch still yields the clean 'not installed' message)."""
    found = shutil.which(_RUNTIME)
    if found:
        return found
    if sys.platform == "win32":
        # Re-read PATH from the registry — a winget install lands there, but
        # this process may be holding a pre-install copy of the environment.
        try:
            import winreg  # noqa: PLC0415 — Windows-only; import locally
            extra: list[str] = []
            for hive, sub in (
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, "Environment"),
            ):
                try:
                    with winreg.OpenKey(hive, sub) as key:
                        extra.append(winreg.QueryValueEx(key, "Path")[0])
                except OSError:
                    pass
            if extra:
                merged = os.pathsep.join([os.environ.get("PATH", "")] + extra)
                found = shutil.which(_RUNTIME, path=merged)
                if found:
                    return found
        except Exception:  # noqa: BLE001 — a registry hiccup must not crash
            pass
        # Known install locations as the final fallback.
        for cand in (
            os.path.expandvars(r"%ProgramFiles%\RedHat\Podman\podman.exe"),
            r"C:\Program Files\RedHat\Podman\podman.exe",
        ):
            if os.path.isfile(cand):
                return cand
    return _RUNTIME


def _runtime() -> str:
    """Cached accessor for the resolved runtime executable path."""
    global _runtime_path
    if _runtime_path is None:
        resolved = _resolve_runtime()
        if os.path.isabs(resolved):
            _runtime_path = resolved      # cache only a genuine hit
        return resolved
    return _runtime_path


def _truncate(text: str, limit: int = _MAX_OUTPUT_LEN) -> str:
    """Cap a stream so a runaway print loop can't flood the prompt."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…(truncated — {len(text) - limit} more chars)"


def _force_remove(name: str) -> None:
    """Best-effort cleanup of a timed-out container. `--rm` only fires on a
    clean exit; when we kill podman on timeout the container can linger, so
    remove it by name explicitly."""
    try:
        subprocess.run(
            [_runtime(), "rm", "-f", name],
            capture_output=True, timeout=10,
        )
    except Exception:  # noqa: BLE001 — cleanup is best-effort
        pass


def execute_code_tool(params: dict) -> str:
    """Run Python in the sandbox. Always returns a string for Claude — never
    raises. Podman missing / machine down / image unbuilt / timeout / nonzero
    exit all become readable strings."""
    code = params.get("code") or ""
    if not code.strip():
        return "No code was provided to run, sir."
    if len(code) > _MAX_CODE_LEN:
        return (
            f"That code is {len(code)} characters — past the "
            f"{_MAX_CODE_LEN}-char sandbox limit. Trim it down, sir."
        )

    # Audit trail: every code run is logged to stderr (jarvis.log) — the same
    # transparency mechanism pc_shell uses. The console also shows code=yes.
    print(
        f"[code] running {len(code)} chars in {_RUNTIME} sandbox:\n"
        f"{'-' * 50}\n{code}\n{'-' * 50}",
        file=sys.stderr,
    )

    name = f"jarvis-codex-{uuid.uuid4().hex[:12]}"
    cmd = [
        _runtime(), "run", "--rm", "-i",
        "--network=none",
        f"--memory={_MEM_LIMIT}",
        f"--cpus={_CPU_LIMIT}",
        f"--pids-limit={_PIDS_LIMIT}",
        "--security-opt=no-new-privileges",
        "--name", name,
        _IMAGE,
        "python", "-",          # read the script from stdin
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        # The runtime binary isn't on PATH at all.
        return (
            f"Code execution is unavailable, sir — '{_RUNTIME}' isn't "
            f"installed or isn't on PATH. Install Podman to enable it."
        )
    except subprocess.TimeoutExpired:
        # podman was killed; force-remove the container so it doesn't linger.
        _force_remove(name)
        return (
            f"The code ran past the {_TIMEOUT_SEC}-second sandbox limit and "
            f"was killed, sir — most likely an infinite loop or a task too "
            f"heavy for the sandbox."
        )
    except Exception as exc:  # noqa: BLE001 — never crash the listen loop
        return f"The sandbox failed to run, sir: {type(exc).__name__}."

    out = _truncate((proc.stdout or "").strip())
    err = _truncate((proc.stderr or "").strip())

    # Distinguish a runtime-level failure (podman could not even start the
    # container) from the code itself erroring. podman exits 125 for its own
    # failures; the stderr text pins down which actionable cause it was.
    if proc.returncode != 0:
        low = err.lower()
        if any(s in low for s in (
            "no such image", "image not known", "unable to find image",
        )):
            return (
                "Code execution is unavailable, sir — the sandbox image "
                "isn't built yet. Build it once with: "
                "podman build -t jarvis-code-sandbox sandbox/"
            )
        if any(s in low for s in (
            "cannot connect", "podman machine", "connection refused",
            "is the podman service running",
        )):
            return (
                "Code execution is unavailable, sir — the Podman machine "
                "isn't running. Start Podman Desktop, or run "
                "`podman machine start`."
            )

    # Normal path: hand Claude the exit code + both streams, so it can
    # summarize a success or read a traceback and iterate on the next turn.
    parts = [f"Exit code: {proc.returncode}"]
    parts.append(f"--- stdout ---\n{out if out else '(empty)'}")
    if err:
        parts.append(f"--- stderr ---\n{err}")
    return "\n".join(parts)
