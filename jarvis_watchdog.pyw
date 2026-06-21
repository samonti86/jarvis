"""Crash-resilience watchdog launcher (M65) — the production hardening
companion to `jarvis.pyw`.

Same role as jarvis.pyw — Windows hands .pyw files to pythonw.exe (no
console), the file makes sure the venv's Python is in use, and then it
launches Jarvis. The DIFFERENCE: this launcher does NOT run main() in
this process. It spawns `python main.py` as a CHILD and watches its exit
code. The child crashes ⇒ this respawns it. The child quits cleanly ⇒
this exits too.

Why this is needed
------------------
Today `jarvis.pyw` imports + calls main() inline. If main() crashes hard
(an uncaught exception bubbles out of a daemon thread's parent, a
segfault, a torch / dlib native fault), the process dies and nothing
revives it. The user notices only when they try to talk to a
non-responding Jarvis. As Jarvis grows the cost of an unnoticed crash
gets higher — silent miss of a security alarm, a missed proactive
calendar reminder, a homelab monitor that stopped watching at 3 am.

Cooperation with the tray restart paths (M32 / M41 / M64)
---------------------------------------------------------
The tray's "Restart Jarvis" used to fire `autostart.relaunch()` which
spawns a new detached process — that works fine standalone but would
race with the watchdog (the watchdog respawns AND the relaunch spawns
⇒ two instances fighting over the audio device). The contract:

  * main.py sees `JARVIS_WATCHDOG=1` in its env (set by THIS launcher).
  * On a "normal" restart it skips `autostart.relaunch()` and exits 42.
  * The watchdog reads 42 as "respawn, please."
  * Exit 0 = user quit, watchdog exits too.
  * Any other exit code = crash, watchdog respawns with backoff/cap.

The "elevated" restart path (M41) still uses `ShellExecuteW("runas")`
and exits 0 — the elevated child becomes unwatched (documented
limitation, acceptable since elevated mode is rare and user-driven).

Rapid-restart cap
-----------------
A bug that crashes Jarvis at startup would otherwise spin forever. The
watchdog tracks the last _RESTART_HISTORY restart timestamps; if all of
them are within _RESTART_WINDOW seconds, give up and log fatally. The
user can read jarvis.log to see what happened.

Why not Windows Services
------------------------
A Windows Service is the "proper" answer for "always running, restart
on crash." But: installing requires admin, complicates the autostart
path, and Jarvis intentionally runs as the user (so it can see the user's
screen, mic, etc.). A pythonw watchdog is the lowest-friction option;
upgrade to a service if/when the user actually wants it.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_VENV_PYTHONW = _HERE / "venv" / "Scripts" / "pythonw.exe"
_VENV_PYTHON = _HERE / "venv" / "Scripts" / "python.exe"  # console-attached fallback
# Spawn jarvis.pyw (not main.py) — it owns the stdout/stderr → log redirect
# and the `--- Jarvis started ---` session marker. Skipping it under pythonw
# means main.py runs headless (sys.stdout is None ⇒ prints are silent), and
# M60's "errors since session start" count loses its anchor. Both processes
# appending to the same jarvis.log is fine (Windows O_APPEND, line-buffered).
_CHILD_LAUNCHER = _HERE / "jarvis.pyw"

# How many recent restarts to retain. If the most recent _RESTART_HISTORY
# starts ALL happened within the last _RESTART_WINDOW seconds, the
# watchdog gives up. 3 in 60s is a generous-but-still-clearly-broken
# crash loop signature; tune via env if needed.
_RESTART_HISTORY = 3
_RESTART_WINDOW = 60.0

# Sentinel — main.py exits with this code to mean "watchdog, please
# respawn me." Any other non-zero exit is treated as a crash (also
# respawned, just with the cap engaged). Plain old 0 = clean quit.
_RESTART_EXIT_CODE = 42


def _ensure_correct_interpreter() -> None:
    """Re-exec under the venv's pythonw if we're not already there. Same
    pattern as jarvis.pyw — Windows can hand .pyw to the system Python."""
    if not _VENV_PYTHONW.exists():
        return
    if Path(sys.executable).resolve() == _VENV_PYTHONW.resolve():
        return
    subprocess.Popen([str(_VENV_PYTHONW), str(_HERE / "jarvis_watchdog.pyw")])
    sys.exit(0)


def _redirect_to_logfile() -> Path:
    """Mirror of jarvis.pyw's log redirect so watchdog events land in the
    same jarvis.log alongside main's logs. The watchdog's own lines are
    tagged `[watchdog]` for grep."""
    from src.logfile import rotate_if_needed, TimestampStream

    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "jarvis.log"
    rotate_if_needed(log_path)
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    stamped = TimestampStream(f)  # per-line timestamps (see src/logfile.py)
    sys.stdout = stamped
    sys.stderr = stamped
    print(f"\n--- Jarvis watchdog started "
          f"{datetime.now().isoformat(timespec='seconds')} ---")
    return log_path


def _spawn_child() -> subprocess.Popen:
    """Spawn one Jarvis instance via jarvis.pyw as our child. Sets
    JARVIS_WATCHDOG=1 so main's restart path knows to exit-42 instead of
    spawning. jarvis.pyw handles its own stdout/stderr → log redirect +
    session marker, so the child's prints land in jarvis.log alongside
    the watchdog's. NOT detached — we WANT to wait on it."""
    env = os.environ.copy()
    env["JARVIS_WATCHDOG"] = "1"
    return subprocess.Popen(
        [str(_VENV_PYTHONW), str(_CHILD_LAUNCHER)],
        cwd=str(_HERE),
        env=env,
    )


def _is_crash_looping(restart_history: list[float]) -> bool:
    """True iff the most recent _RESTART_HISTORY restarts ALL happened
    within _RESTART_WINDOW seconds. Pure decision — testable in isolation."""
    if len(restart_history) < _RESTART_HISTORY:
        return False
    oldest = restart_history[-_RESTART_HISTORY]
    return (time.time() - oldest) < _RESTART_WINDOW


def watchdog_loop() -> int:
    """Spawn → wait → decide, until the child quits cleanly or we cap. The
    return value becomes the watchdog process's own exit code: 0 on a clean
    handoff (user quit or elevated takeover), 1 on a crash-loop give-up."""
    restart_history: list[float] = []

    while True:
        print(f"[watchdog] spawning jarvis.pyw "
              f"(restart history: {len(restart_history)})")
        try:
            child = _spawn_child()
        except (OSError, FileNotFoundError) as exc:
            print(f"[watchdog] spawn failed: {exc} — giving up")
            return 1

        try:
            rc = child.wait()
        except KeyboardInterrupt:
            # Ctrl+C in a dev console — propagate to the child and exit.
            print("[watchdog] KeyboardInterrupt — terminating child")
            try:
                child.terminate()
            except OSError:
                pass
            return 0

        if rc == 0:
            print("[watchdog] child exited cleanly (user quit) — done")
            return 0

        if rc == _RESTART_EXIT_CODE:
            # The "restart me" signal from M32/M64 — respawn immediately,
            # don't count this against the crash budget (it's user-driven).
            print(f"[watchdog] child requested restart (exit {rc}) — "
                  f"respawning immediately")
            continue

        # Unexpected exit → crash. Add to history, check cap.
        restart_history.append(time.time())
        if _is_crash_looping(restart_history):
            print(f"[watchdog] CRASH LOOP — {_RESTART_HISTORY} restarts "
                  f"within {_RESTART_WINDOW:.0f}s, last exit code {rc}. "
                  f"Giving up.", file=sys.stderr)
            return 1
        # Brief backoff so we don't hammer a CPU-eating tight crash loop;
        # 2s is enough that a flaky network/race usually clears by retry.
        print(f"[watchdog] child crashed (exit {rc}) — respawning in 2s")
        time.sleep(2.0)


def main() -> int:
    """Entry point. Side effects (interpreter check, log redirect) are
    contained here so importing this module for tests is safe — only the
    invocation under `python jarvis_watchdog.pyw` runs them."""
    _ensure_correct_interpreter()
    log_path = _redirect_to_logfile()
    os.environ["JARVIS_LOG_PATH"] = str(log_path)
    return watchdog_loop()


if __name__ == "__main__":
    sys.exit(main())
