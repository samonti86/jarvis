"""M88 Phase 2 de-risk — hands-free talk-over feasibility probe.

THE QUESTION: for full-duplex "talk over Jarvis and he stops" (no wake word),
the omni mic must stay open WHILE Jarvis speaks and we must tell the user's voice
apart from Jarvis's OWN playback bleeding into the mic. Is that separable on this
hardware, or do we need acoustic echo cancellation (AEC)?

This probe MEASURES it instead of guessing (the M58/M72 discipline). It plays a
~20s Jarvis TTS clip through the speakers while recording the mic, and reports
the per-frame RMS distribution. Run it twice at the SAME speaker + mic volume:

  1. SILENT  — let Jarvis talk, you stay quiet.   → the pure ECHO FLOOR
                (everything the mic hears is Jarvis bleeding back in).
  2. TALK    — talk over Jarvis continuously at a normal conversational volume.
                → ECHO + YOUR VOICE.

Then `compare` reads both and prints the verdict:
  - If the TALK distribution sits CLEARLY above the SILENT echo floor, a simple
    energy/envelope gate can detect a barge-in → cheap Phase 2.
  - If they OVERLAP, raw energy can't separate them → Phase 2 needs real AEC
    (Windows comms-mode AEC, or a software echo canceller).

Throwaway diagnostic — NOT a gate test (it needs hardware + a human talking), so
it's named *_probe.py (run_all_tests only picks up *_test.py).

USAGE (quit the running Jarvis first via the tray, so it doesn't hear the probe
or contend for the mic):

    venv\\Scripts\\python.exe scripts\\duplex_echo_probe.py silent
    venv\\Scripts\\python.exe scripts\\duplex_echo_probe.py talk
    venv\\Scripts\\python.exe scripts\\duplex_echo_probe.py compare

Report back the two summary blocks (or the compare verdict).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.audio import resolve_input_device  # noqa: E402
from src.text_to_speech import speak  # noqa: E402

load_dotenv()  # pick up JARVIS_MIC_DEVICE from .env (no full config/API key needed)

# RMS is computed per 50 ms frame on float32 mono [0,1] — the SAME scale as the
# acoustic rms floors (JARVIS_ACOUSTIC_*_RMS_FLOOR ≈ 0.02–0.06), so the numbers
# are directly relatable to the rest of the system.
_SAMPLE_RATE = 16_000
_FRAME = 800  # 50 ms @ 16 kHz

# ~20 s of speech at a steady cadence — long enough to talk over comfortably.
_CLIP = (
    "Right then, sir. This is a calibration pass for hands-free conversation. "
    "I shall keep speaking at a steady, even pace for a short while so we can "
    "measure how my own voice carries back into the microphone. Please follow "
    "the instructions for this run. If this is the silent pass, simply let me "
    "talk and stay quiet. If this is the talk-over pass, speak over me now in a "
    "natural, conversational voice, as though you were interrupting me. Thank "
    "you, sir. That should be quite enough to take a clean measurement."
)


def _out_path(mode: str) -> Path:
    base = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"duplex_{mode}.json"


def _record_run(mode: str) -> None:
    device = resolve_input_device(os.getenv("JARVIS_MIC_DEVICE", "").strip())

    frames: list[float] = []
    stop = threading.Event()

    def capture() -> None:
        def cb(indata, _frames, _time, status):  # noqa: ANN001
            if status:
                print(f"[probe] stream status: {status}", file=sys.stderr)
            mono = indata[:, 0] if indata.ndim > 1 else indata
            frames.append(float(np.sqrt(np.mean(np.square(mono))) + 1e-12))

        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=_FRAME, device=device, callback=cb):
            stop.wait()

    if mode == "talk":
        print("\n>>> TALK over Jarvis NOW — keep talking until he stops. <<<\n")
    else:
        print("\n>>> SILENT pass — let Jarvis talk, stay quiet. <<<\n")

    cap = threading.Thread(target=capture, daemon=True)
    cap.start()
    try:
        speak(_CLIP, language="en")   # blocks for the duration of playback
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] speak failed ({exc}) — is edge-tts/pyttsx3 working?",
              file=sys.stderr)
    finally:
        stop.set()
        cap.join(timeout=2.0)

    arr = np.array(frames, dtype=np.float64)
    # Trim the leading/trailing ~0.5 s so pre/post-playback room silence doesn't
    # drag the floor down (we want the distribution DURING playback).
    if arr.size > 24:
        arr = arr[10:-10]
    if arr.size == 0:
        print("[probe] no audio captured — check the mic device.", file=sys.stderr)
        return

    stats = {
        "mode": mode,
        "device": device,
        "frames": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }
    _out_path(mode).write_text(json.dumps(stats, indent=2))
    _print_stats(stats)
    print(f"\nsaved → {_out_path(mode)}")


def _print_stats(s: dict) -> None:
    print(f"\n--- {s['mode'].upper()} run  (device={s['device']}, "
          f"{s['frames']} frames) ---")
    for k in ("mean", "median", "p90", "p95", "max"):
        print(f"  {k:>6}: {s[k]:.4f}")


def _compare() -> None:
    sp, tp = _out_path("silent"), _out_path("talk")
    if not sp.exists() or not tp.exists():
        print("[probe] need both runs first: 'silent' then 'talk'.",
              file=sys.stderr)
        sys.exit(1)
    s = json.loads(sp.read_text())
    t = json.loads(tp.read_text())
    _print_stats(s)
    _print_stats(t)

    # The decision: does the user's voice (TALK median) clear the echo ceiling
    # (SILENT p95) with margin? If so, a fixed energy gate set between them
    # detects a barge-in without AEC.
    echo_ceiling = s["p95"]
    user_typical = t["median"]
    margin = user_typical - echo_ceiling
    ratio = user_typical / max(echo_ceiling, 1e-6)
    print("\n=== VERDICT ===")
    print(f"  echo ceiling (silent p95): {echo_ceiling:.4f}")
    print(f"  user typical (talk median): {user_typical:.4f}")
    print(f"  margin: {margin:+.4f}   ratio: {ratio:.2f}x")
    if margin > 0.015 and ratio >= 1.8:
        print("  → SEPARABLE. A fixed energy gate (~midpoint) can detect a "
              "barge-in WITHOUT AEC — cheap Phase 2 is viable.")
        mid = (echo_ceiling + user_typical) / 2
        print(f"    suggested barge threshold ≈ {mid:.4f}")
    elif margin > 0:
        print("  → MARGINAL. Some separation, but tight — energy gating would "
              "be false-positive-prone. Consider envelope-referenced gating "
              "(use Jarvis's own TTS amplitude as a reference) or AEC.")
    else:
        print("  → NOT SEPARABLE by energy alone (user voice is buried in the "
              "echo). Phase 2 needs real AEC: Windows comms-mode capture, or a "
              "software echo canceller (speexdsp / webrtc-audio-processing).")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode in ("silent", "talk"):
        _record_run(mode)
    elif mode == "compare":
        _compare()
    else:
        print(__doc__)
        print("\nGive one of: silent | talk | compare")
        sys.exit(2)


if __name__ == "__main__":
    main()
