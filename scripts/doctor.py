r"""Jarvis setup doctor — "why won't it start?" answered in one command.

WHY THIS EXISTS:
Jarvis has one hard requirement (ANTHROPIC_API_KEY) and ~20 optional
integrations that each degrade to "not configured" rather than failing. That
fail-soft contract is right for a long-running process, but it makes a BROKEN
setup and a MINIMAL setup look identical from the outside: both just quietly
don't do the thing. This prints the difference.

Read-only. Touches no state, opens no audio device, makes no network call, and
never prints a secret's value — only whether it is set and plausibly shaped.

DESIGN CONSTRAINT: this must run BEFORE `pip install -r requirements.txt`,
which is exactly when it's most useful. So nothing heavy is imported at module
level; every dependency check is a guarded probe.

    python scripts\doctor.py            # exit 0 = Jarvis can start
    python scripts\doctor.py --verbose  # also list audio devices
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

OK, WARN, FAIL, INFO = "  OK  ", " WARN ", " FAIL ", " ..   "

_blockers = 0
_warnings = 0


def line(status: str, label: str, detail: str = "") -> None:
    global _blockers, _warnings
    if status == FAIL:
        _blockers += 1
    elif status == WARN:
        _warnings += 1
    print(f"[{status}] {label}" + (f"  --  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _installed(mod: str) -> bool:
    """True if importable WITHOUT importing it (no import side effects)."""
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------
# 1. Interpreter
# --------------------------------------------------------------------------
def check_python() -> None:
    section("Interpreter")
    v = sys.version_info
    if (v.major, v.minor) >= (3, 12):
        line(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        line(FAIL, f"Python {v.major}.{v.minor}.{v.micro}", "3.12+ required")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        line(OK, "running inside a virtualenv", sys.prefix)
    else:
        line(WARN, "not running inside a virtualenv",
             r"expected .\venv\Scripts\python.exe")

    if sys.platform != "win32":
        line(WARN, f"platform is {sys.platform}",
             "Jarvis is Windows-native (WASAPI audio, SAPI TTS, tray)")


# --------------------------------------------------------------------------
# 2. Dependencies
# --------------------------------------------------------------------------
# (import name, pip name, what breaks without it). Ordered core-first.
_CORE = [
    ("anthropic", "anthropic", "the LLM itself - Jarvis cannot start"),
    ("dotenv", "python-dotenv", "reading .env"),
    ("numpy", "numpy", "all audio maths"),
    ("sounddevice", "sounddevice", "microphone capture and playback"),
    ("openwakeword", "openwakeword", "the wake word"),
    ("faster_whisper", "faster-whisper", "speech-to-text"),
    ("edge_tts", "edge-tts", "the primary voice"),
    ("httpx", "httpx", "every web-facing tool"),
]
_UI = [
    ("pystray", "pystray", "tray icon"),
    ("PIL", "Pillow", "tray icon rendering"),
    ("customtkinter", "customtkinter", "the console window"),
]
_OPTIONAL = [
    ("pyttsx3", "pyttsx3", "offline TTS fallback"),
    ("mcp", "mcp", "Plex integration"),
    ("psutil", "psutil", "system diagnostics"),
    ("cv2", "opencv-python-headless", "webcam snapshots"),
    ("ultralytics", "ultralytics", "vision security mode"),
    ("panns_inference", "panns_inference", "acoustic awareness"),
    ("resemblyzer", "resemblyzer", "speaker identification"),
    ("sentence_transformers", "sentence-transformers", "knowledge-base semantic search"),
    ("discord", "discord.py", "the Discord bridge"),
    ("websockets", "websockets", "the phone console"),
    ("paramiko", "paramiko", "remote Plex-host diagnostics"),
    ("feedparser", "feedparser", "the news tool"),
]


def check_deps() -> None:
    section("Core dependencies")
    for mod, pip_name, why in _CORE:
        if _installed(mod):
            line(OK, pip_name)
        else:
            line(FAIL, pip_name, f"missing - breaks {why}")

    section("Interface dependencies")
    for mod, pip_name, why in _UI:
        line(OK if _installed(mod) else WARN, pip_name,
             "" if _installed(mod) else f"missing - no {why}")

    section("Optional dependencies (each degrades one feature)")
    missing = [(p, w) for m, p, w in _OPTIONAL if not _installed(m)]
    present = [p for m, p, w in _OPTIONAL if _installed(m)]
    if present:
        line(OK, f"{len(present)} present", ", ".join(present))
    for pip_name, why in missing:
        line(INFO, pip_name, f"absent - no {why}")


# --------------------------------------------------------------------------
# 3. Configuration
# --------------------------------------------------------------------------
def _load_env() -> dict[str, str]:
    """Parse .env WITHOUT requiring python-dotenv (it may not be installed)."""
    env_path = ROOT / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


# label -> the keys that together enable it
_INTEGRATIONS: list[tuple[str, list[str]]] = [
    ("Weather / briefing / severe-weather alerts", ["JARVIS_HOME_LOCATION"]),
    ("Films & TV (TMDB)", ["TMDB_API_KEY"]),
    ("Games (RAWG)", ["RAWG_API_KEY"]),
    ("Computation (WolframAlpha)", ["WOLFRAM_APP_ID"]),
    ("Calendar (Outlook iCal)", ["OUTLOOK_ICAL_URL"]),
    ("Plex", ["PLEX_URL", "PLEX_TOKEN"]),
    ("Discord push (one-way)", ["DISCORD_WEBHOOK_URL"]),
    ("Discord bot (two-way)", ["DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID"]),
    ("Email alerts (SMTP)", ["SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_TO"]),
    ("Phone / remote console", ["JARVIS_REMOTE_TOKEN"]),
    ("GPU speech-to-text offload", ["STT_SERVER_URL"]),
    ("Security challenge passphrase", ["JARVIS_SECURITY_PASSPHRASE"]),
]


def check_config() -> dict[str, str]:
    section("Configuration")
    env_path = ROOT / ".env"
    if env_path.exists():
        line(OK, ".env present")
    else:
        line(FAIL, ".env missing",
             "copy it:  Copy-Item .env.example .env")

    env = _load_env()
    # A real environment variable wins over the file, mirroring load_dotenv().
    def get(k: str) -> str:
        return os.getenv(k) or env.get(k, "")

    key = get("ANTHROPIC_API_KEY")
    if not key or key.startswith("sk-ant-..."):
        line(FAIL, "ANTHROPIC_API_KEY not set",
             "the one hard requirement - https://console.anthropic.com/settings/keys")
    elif not key.startswith("sk-ant-"):
        line(WARN, "ANTHROPIC_API_KEY set", "does not look like an Anthropic key")
    else:
        line(OK, "ANTHROPIC_API_KEY set", f"sk-ant-...{key[-4:]}")

    section("Optional integrations")
    for label, keys in _INTEGRATIONS:
        have = [k for k in keys if get(k)]
        if len(have) == len(keys):
            line(OK, label)
        elif have:
            missing = [k for k in keys if not get(k)]
            line(WARN, label, f"partially configured - missing {', '.join(missing)}")
        else:
            line(INFO, label, "not configured")
    return env


# --------------------------------------------------------------------------
# 4. Hardware & external tools
# --------------------------------------------------------------------------
def check_hardware(env: dict[str, str]) -> None:
    section("Audio hardware")
    if not _installed("sounddevice"):
        line(WARN, "cannot check audio", "sounddevice not installed")
    else:
        try:
            import sounddevice as sd  # noqa: PLC0415  (deliberate late import)

            devices = sd.query_devices()
            ins = [d for d in devices if d.get("max_input_channels", 0) > 0]
            outs = [d for d in devices if d.get("max_output_channels", 0) > 0]
            line(OK if ins else FAIL, f"{len(ins)} input device(s)",
                 "" if ins else "no microphone - Jarvis cannot listen")
            line(OK if outs else FAIL, f"{len(outs)} output device(s)",
                 "" if outs else "no speaker - Jarvis cannot reply aloud")

            pin = os.getenv("JARVIS_MIC_DEVICE") or env.get("JARVIS_MIC_DEVICE", "")
            if pin:
                matched = [
                    d["name"] for d in ins if pin.lower() in d["name"].lower()
                ] or ([ins[int(pin)]["name"]] if pin.isdigit() and int(pin) < len(ins) else [])
                if matched:
                    line(OK, f"JARVIS_MIC_DEVICE={pin}", f"matches {matched[0]}")
                else:
                    line(WARN, f"JARVIS_MIC_DEVICE={pin}",
                         "matches no device - will fall back to the default mic")
            if VERBOSE:
                for d in ins:
                    print(f"          in : {d['name']}")
                for d in outs:
                    print(f"          out: {d['name']}")
        except Exception as exc:  # noqa: BLE001 - a probe must never crash
            line(WARN, "audio device query failed", str(exc)[:80])

    section("External tools")
    podman = shutil.which("podman")
    line(OK if podman else INFO, "podman",
         podman or "absent - the run_code sandbox is disabled")
    git = shutil.which("git")
    line(OK if git else WARN, "git", git or "absent - self-update is disabled")

    section("Model caches (downloaded on first use)")
    home = Path.home()
    for label, path in [
        ("openWakeWord", home / ".cache" / "openwakeword"),
        ("faster-whisper / HuggingFace", home / ".cache" / "huggingface"),
        ("YOLO weights", ROOT / "yolov8n.pt"),
    ]:
        line(OK if path.exists() else INFO, label,
             str(path) if path.exists() else "not yet downloaded")


def main() -> int:
    print("=" * 70)
    print("Jarvis setup doctor")
    print("=" * 70)
    check_python()
    check_deps()
    env = check_config()
    check_hardware(env)

    print("\n" + "=" * 70)
    if _blockers:
        print(f"  {_blockers} blocker(s), {_warnings} warning(s) - Jarvis will NOT start.")
        print("  Fix the [ FAIL ] lines above. Setup guide: docs/SETUP.md")
    else:
        print(f"  No blockers, {_warnings} warning(s) - Jarvis can start.")
        print("  Run it:  python main.py     (or: pythonw jarvis.pyw  for no console)")
    print("=" * 70)
    return 1 if _blockers else 0


if __name__ == "__main__":
    sys.exit(main())
