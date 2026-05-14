"""Windows shortcuts + process relaunch.

Three jobs in this module:

1. **Autostart shortcut** — a .lnk in the user's Startup folder so Jarvis
   comes up at login. Toggleable from the tray menu.
2. **Desktop shortcut** — a .lnk on the user's Desktop that they can
   double-click or right-click → 'Pin to taskbar'. Windows blocked
   programmatic taskbar pinning in 2018, so this is as close as we get.
3. **Relaunch** — spawn a fresh detached jarvis.pyw process. Used by the
   tray's 'Restart Jarvis' menu after the current process shuts down
   cleanly.

PowerShell + WScript.Shell COM is the canonical Windows tool for authoring
shortcuts; stdlib Python has no native support and pywin32 would be
overkill for a few ComObject calls.

Both shortcuts point at venv pythonw — same self-relaunch logic in
jarvis.pyw would catch a wrong-interpreter case anyway, but pointing
straight at the venv is one less hop on launch.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PYTHONW = _PROJECT_ROOT / "venv" / "Scripts" / "pythonw.exe"
_LAUNCHER = _PROJECT_ROOT / "jarvis.pyw"
_ICON_PATH = _PROJECT_ROOT / "assets" / "jarvis.ico"

_STARTUP_DIR = (
    Path(os.environ.get("APPDATA", str(Path.home())))
    / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
)
_SHORTCUT_PATH = _STARTUP_DIR / "Jarvis.lnk"

# USERPROFILE is the standard Windows env var for the home folder. We
# resolve Desktop relative to it rather than via [Environment]::GetFolderPath
# to keep this module Python-only and avoid an extra PS call. Note: if the
# user has OneDrive redirecting Desktop, ~/Desktop may not exist — we mkdir
# the parent in _write_shortcut so the error message is clearer.
_DESKTOP_DIR = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
_DESKTOP_SHORTCUT_PATH = _DESKTOP_DIR / "Jarvis.lnk"


# Windows process-creation flags (subprocess.CREATE_* constants exist but
# duplicating the literals here keeps this module portable when imported
# from a non-Windows path during tests).
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def is_enabled() -> bool:
    """True if the autostart shortcut exists at the canonical Startup path."""
    return _SHORTCUT_PATH.exists()


def shortcut_path() -> Path:
    """Where the autostart shortcut lives. Useful for logging / diagnostics."""
    return _SHORTCUT_PATH


def desktop_shortcut_path() -> Path:
    """Where the desktop shortcut lives. The file may or may not exist —
    use `has_desktop_shortcut()` to check."""
    return _DESKTOP_SHORTCUT_PATH


def has_desktop_shortcut() -> bool:
    return _DESKTOP_SHORTCUT_PATH.exists()


def _write_shortcut(
    target_path: Path,
    description: str,
    icon_path: Path | None = None,
) -> None:
    """Author a .lnk via PowerShell + WScript.Shell. Shared by autostart
    and desktop variants — they differ only in destination + description +
    whether to embed an icon.

    Single-quoted PS strings sidestep `$` expansion and backslash escaping;
    paths flow through verbatim. The optional IconLocation pins a specific
    .ico (',0' selects the first icon resource in the file)."""
    if not _VENV_PYTHONW.exists():
        raise RuntimeError(f"venv pythonw not found at {_VENV_PYTHONW}")
    if not _LAUNCHER.exists():
        raise RuntimeError(f"jarvis.pyw not found at {_LAUNCHER}")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    icon_line = (
        f"$Shortcut.IconLocation = '{icon_path},0'\n" if icon_path is not None else ""
    )
    script = (
        f"$WshShell = New-Object -ComObject WScript.Shell\n"
        f"$Shortcut = $WshShell.CreateShortcut('{target_path}')\n"
        f"$Shortcut.TargetPath = '{_VENV_PYTHONW}'\n"
        f"$Shortcut.Arguments = '\"{_LAUNCHER}\"'\n"
        f"$Shortcut.WorkingDirectory = '{_PROJECT_ROOT}'\n"
        f"$Shortcut.Description = '{description}'\n"
        f"{icon_line}"
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
        raise RuntimeError(f"shortcut creation failed at {target_path}: {stderr}")


def enable() -> None:
    """Create or overwrite the autostart shortcut. Refreshes paths on every
    call, so re-toggling fixes a stale shortcut after the project moves."""
    _write_shortcut(
        _SHORTCUT_PATH,
        description="Jarvis voice assistant — silent background launcher",
    )


def disable() -> None:
    """Remove the autostart shortcut. Idempotent — silent if it's not there."""
    try:
        _SHORTCUT_PATH.unlink()
    except FileNotFoundError:
        pass


def _generate_icon() -> Path:
    """Lazily render assets/jarvis.ico if it doesn't exist; return the path.

    Design: cyan circle (#7dd3fc, matching the console header palette) with
    a dark 'J' centered. Saved as a multi-resolution .ico — Windows picks
    the right size for each context (16px taskbar, 32px alt-tab, 48px file
    explorer, 256px high-DPI). PIL handles the resampling.

    Cached on disk: subsequent calls short-circuit without re-rendering.
    Delete the file to force regeneration (e.g. after a design tweak).
    """
    if _ICON_PATH.exists():
        return _ICON_PATH

    # Local import so module load doesn't pull in PIL on machines that
    # never invoke shortcut creation. Pillow IS a hard dep elsewhere
    # (tray.py, console.py), but the principle of paying only-when-needed
    # is cheap to maintain here.
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([16, 16, 240, 240], fill=(125, 211, 252, 255))  # #7dd3fc

    # Segoe UI Bold ships on Windows 10/11 — won't be missing in practice.
    # The default PIL font is a 10px bitmap that would look pathetic at 256;
    # the fallback is just defensive for non-Windows test runs.
    try:
        font = ImageFont.truetype("segoeuib.ttf", 180)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), "J", font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    # bbox offsets account for fonts whose glyph origin isn't at (0,0) —
    # without subtracting bbox[0]/[1] the J would be drawn slightly off-center.
    x = (256 - text_w) // 2 - bbox[0]
    y = (256 - text_h) // 2 - bbox[1]
    draw.text((x, y), "J", fill=(26, 29, 35, 255), font=font)  # #1a1d23

    _ICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(
        _ICON_PATH,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    return _ICON_PATH


def create_desktop_shortcut() -> Path:
    """Create (or overwrite) a Jarvis.lnk on the user's Desktop with a
    custom icon. Returns the shortcut path so callers can reveal it in
    Explorer. Lazily generates the icon on first use.

    Windows blocks programmatic taskbar pinning (deliberately removed in
    Win10 1809). The realistic flow is therefore: create a discoverable
    .lnk, tell the user to right-click → 'Pin to taskbar'."""
    icon = _generate_icon()
    _write_shortcut(
        _DESKTOP_SHORTCUT_PATH,
        description="Jarvis — voice assistant",
        icon_path=icon,
    )
    return _DESKTOP_SHORTCUT_PATH


def relaunch() -> None:
    """Spawn a new silent-launcher Jarvis detached from this process.

    Always launches via venv pythonw + jarvis.pyw (the production silent
    path). DEV-mode callers running `python main.py` will get a silent
    background instance back — the restart button is intentionally a
    production UX, not a dev workflow.

    DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP fully separate the child
    so it survives our exit; close_fds=True prevents the child from
    inheriting any audio device handles still held by the listen_loop.
    Callers should invoke this AFTER worker.join() so the mic is released
    before the new instance tries to claim it."""
    if not _VENV_PYTHONW.exists() or not _LAUNCHER.exists():
        raise RuntimeError(
            f"relaunch failed: missing {_VENV_PYTHONW} or {_LAUNCHER}"
        )

    subprocess.Popen(
        [str(_VENV_PYTHONW), str(_LAUNCHER)],
        cwd=str(_PROJECT_ROOT),
        close_fds=True,
        creationflags=_DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP,
    )
