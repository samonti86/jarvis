"""Autostart on Windows login via shell:startup shortcut.

Creates / removes a .lnk in the user's Startup folder pointing at this
checkout's venv pythonw + jarvis.pyw. PowerShell + WScript.Shell COM is the
canonical Windows tool for authoring shortcuts; stdlib Python has no native
support and we don't want a heavyweight new dep (pywin32 is overkill for
two ComObject calls).

The shortcut points at venv pythonw — same self-relaunch logic that lives
in jarvis.pyw would catch a wrong-interpreter case anyway, but pointing
straight at the venv is one less hop on every login.

Toggling is idempotent: enable() overwrites any existing shortcut so
relocating the project just requires toggling off / on to refresh paths.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHONW = _PROJECT_ROOT / "venv" / "Scripts" / "pythonw.exe"
_LAUNCHER = _PROJECT_ROOT / "jarvis.pyw"

_STARTUP_DIR = (
    Path(os.environ.get("APPDATA", str(Path.home())))
    / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
)
_SHORTCUT_PATH = _STARTUP_DIR / "Jarvis.lnk"


def is_enabled() -> bool:
    """True if the autostart shortcut exists at the canonical Startup path."""
    return _SHORTCUT_PATH.exists()


def shortcut_path() -> Path:
    """Where the shortcut lives. Useful for logging / diagnostics."""
    return _SHORTCUT_PATH


def enable() -> None:
    """Create or overwrite the autostart shortcut. Refreshes paths on every
    call, so re-toggling fixes a stale shortcut after the project moves."""
    if not _VENV_PYTHONW.exists():
        raise RuntimeError(f"venv pythonw not found at {_VENV_PYTHONW}")
    if not _LAUNCHER.exists():
        raise RuntimeError(f"jarvis.pyw not found at {_LAUNCHER}")

    _STARTUP_DIR.mkdir(parents=True, exist_ok=True)

    # Build a PowerShell here-doc that uses WScript.Shell to author the .lnk.
    # Path values are interpolated into single-quoted PowerShell strings, so
    # backslashes don't need doubling and embedded $ won't be expanded.
    script = (
        f"$WshShell = New-Object -ComObject WScript.Shell\n"
        f"$Shortcut = $WshShell.CreateShortcut('{_SHORTCUT_PATH}')\n"
        f"$Shortcut.TargetPath = '{_VENV_PYTHONW}'\n"
        f"$Shortcut.Arguments = '\"{_LAUNCHER}\"'\n"
        f"$Shortcut.WorkingDirectory = '{_PROJECT_ROOT}'\n"
        f"$Shortcut.Description = 'Jarvis voice assistant — silent background launcher'\n"
        f"$Shortcut.Save()\n"
    )

    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "(no stderr)"
        raise RuntimeError(f"autostart shortcut creation failed: {stderr}")


def disable() -> None:
    """Remove the autostart shortcut. Idempotent — silent if it's not there."""
    try:
        _SHORTCUT_PATH.unlink()
    except FileNotFoundError:
        pass
