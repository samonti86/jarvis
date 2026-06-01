r"""Speaker-ID de-risk probe — does Resemblyzer actually discriminate YOUR voice?

WHY THIS EXISTS
---------------
Before building the whole speaker-ID subsystem (enroll/identify/gate) we answer
the one question that decides whether the model is good enough: on THIS mic and
THIS voice, is "you vs you" measurably tighter than "you vs someone/something
else"? Synthetic noise embeds fine and the latency is ~30 ms steady-state, but
neither proves the embeddings SEPARATE speakers. This is the M58 discipline —
measure the thing that matters before committing to the model.

It must record you speaking on cue, so it is a YOU-RUN instrument: run it in
your own terminal where you can see the live prompts, speak when told, and play
a YouTube clip (or have someone else talk) for the "other" sample.

WHAT IT DOES
------------
1. Records N enrollment clips of you speaking (different sentences each).
2. Builds an enrolled embedding = L2-normalized mean of the clip embeddings
   (the same averaging strategy face_auth uses for the 5-frame face encoding).
3. Records a "you again" clip and an "other voice / background media" clip.
4. Prints cosine similarity (enrolled · clip; higher = more similar) for each,
   the separation margin, a suggested threshold, and a PASS/CHECK verdict.

INTERPRETING THE NUMBERS (Resemblyzer d-vectors, L2-normalized)
  same-speaker   typically ~0.75-0.90
  diff-speaker   typically ~0.00-0.60
  A clean separation (margin >= ~0.20, same >= ~0.75) => the model discriminates
  here; pick a threshold midway. A muddy one => try SpeechBrain ECAPA instead.

USAGE
-----
  venv\Scripts\python.exe scripts\speaker_id_probe.py
  venv\Scripts\python.exe scripts\speaker_id_probe.py --seconds 5 --enroll 3

Pins the same mic as production (JARVIS_MIC_DEVICE) so the measurement reflects
the real capture path. Writes a copy of the result table to
%LOCALAPPDATA%\Jarvis\speaker_probe.txt so you can paste it back verbatim.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from src.audio import resolve_input_device  # noqa: E402

SAMPLE_RATE = 16_000  # Resemblyzer's working rate (and our STT rate)


def _countdown(label: str) -> None:
    print(f"\n>>> {label}")
    for n in (3, 2, 1):
        print(f"      ...{n}", flush=True)
        time.sleep(0.7)
    print("      ● RECORDING — speak now", flush=True)


def _record(seconds: float, device: int | None) -> np.ndarray:
    """Record `seconds` of mono 16 kHz audio, return float32 in [-1, 1]."""
    import sounddevice as sd  # noqa: PLC0415
    frames = int(seconds * SAMPLE_RATE)
    rec = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                 device=device)
    sd.wait()
    print("      ✓ done", flush=True)
    return (rec[:, 0].astype(np.float32) / 32768.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Speaker-ID discrimination probe.")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="seconds per clip (default 4)")
    ap.add_argument("--enroll", type=int, default=3,
                    help="number of enrollment clips (default 3)")
    args = ap.parse_args()

    device = resolve_input_device(os.getenv("JARVIS_MIC_DEVICE", ""))

    print("=" * 64)
    print("  SPEAKER-ID DE-RISK PROBE")
    print("=" * 64)
    print("  You'll record a few short clips. Speak naturally — a full")
    print("  sentence each time (vary the words). For the LAST clip, play a")
    print("  YouTube video or have someone else talk. Ctrl+C aborts.")

    print("\n  Loading the voice encoder (one-time JIT warm, ~5-15 s) ...")
    from resemblyzer import VoiceEncoder, preprocess_wav  # noqa: PLC0415
    enc = VoiceEncoder()
    # Pay the numba/torch JIT now so the first real clip isn't a 14 s outlier.
    enc.embed_utterance(np.zeros(SAMPLE_RATE, dtype=np.float32) + 1e-4)
    print("  Encoder ready.")

    def embed(seconds: float) -> np.ndarray | None:
        raw = _record(seconds, device)
        try:
            wav = preprocess_wav(raw, source_sr=SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            print(f"      ! preprocess failed: {exc}")
            return None
        if wav.size < SAMPLE_RATE // 2:  # < ~0.5 s of voiced audio survived VAD
            print("      ! too little speech detected in that clip")
            return None
        return enc.embed_utterance(wav)

    # --- Enrollment ------------------------------------------------------
    enroll_embs: list[np.ndarray] = []
    try:
        for i in range(args.enroll):
            _countdown(f"ENROLL clip {i + 1}/{args.enroll} — say a sentence "
                       f"({args.seconds:.0f}s)")
            e = embed(args.seconds)
            if e is not None:
                enroll_embs.append(e)
        if len(enroll_embs) < 2:
            print("\n  Not enough usable enrollment clips. Try again in a "
                  "quieter spot / speak the whole time.")
            return 2
        enrolled = np.mean(np.stack(enroll_embs), axis=0)
        enrolled /= np.linalg.norm(enrolled)  # renormalize the mean

        # --- Verification --------------------------------------------------
        _countdown(f"VERIFY (YOU again) — say something DIFFERENT "
                   f"({args.seconds:.0f}s)")
        same = embed(args.seconds)

        _countdown(f"VERIFY (OTHER) — play a YouTube clip / someone else talks "
                   f"({args.seconds:.0f}s)")
        other = embed(args.seconds)
    except KeyboardInterrupt:
        print("\n  Aborted.")
        return 1

    def cos(a: np.ndarray | None) -> float | None:
        return None if a is None else float(np.dot(enrolled, a))

    s_same, s_other = cos(same), cos(other)

    lines = [
        "",
        "=" * 64,
        "  SPEAKER-ID PROBE — RESULT",
        "=" * 64,
        f"  enrollment clips used : {len(enroll_embs)}/{args.enroll}",
        f"  cos(enrolled, YOU)    : {s_same:.3f}" if s_same is not None
        else "  cos(enrolled, YOU)    : (no usable clip)",
        f"  cos(enrolled, OTHER)  : {s_other:.3f}" if s_other is not None
        else "  cos(enrolled, OTHER)  : (no usable clip)",
    ]
    if s_same is not None and s_other is not None:
        margin = s_same - s_other
        thresh = (s_same + s_other) / 2
        lines += [
            f"  separation margin     : {margin:.3f}",
            f"  suggested threshold   : {thresh:.3f}  (midway)",
            "-" * 64,
        ]
        if margin >= 0.20 and s_same >= 0.75:
            lines += [
                "  VERDICT: PASS — Resemblyzer discriminates your voice here.",
                "  Build on it; use a threshold near the suggested value (tune",
                "  conservatively = fail-open, so a borderline YOU still passes).",
            ]
            verdict = 0
        elif margin >= 0.12:
            lines += [
                "  VERDICT: MARGINAL — some separation but not comfortable.",
                "  Re-run (more/longer enroll clips); if it stays muddy, switch",
                "  to SpeechBrain ECAPA (the documented upgrade path).",
            ]
            verdict = 0
        else:
            lines += [
                "  VERDICT: CHECK — poor separation. Either the clips were noisy",
                "  or Resemblyzer isn't enough here; try ECAPA before building.",
            ]
            verdict = 1
    else:
        lines += ["-" * 64,
                  "  VERDICT: INCONCLUSIVE — a verification clip had no speech."]
        verdict = 1
    lines.append("=" * 64)

    report = "\n".join(lines)
    print(report)
    try:
        out = Path(os.getenv("LOCALAPPDATA", ".")) / "Jarvis" / "speaker_probe.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\n  (saved to {out})")
    except Exception:  # noqa: BLE001
        pass
    return verdict


if __name__ == "__main__":
    sys.exit(main())
