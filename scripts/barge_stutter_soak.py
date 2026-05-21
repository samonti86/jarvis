"""M52 — barge-in stutter soak harness.

WHY THIS EXISTS
---------------
M52 barge-in deliberately reintroduces the concurrency the armed-mode
stutter post-mortem fought (docs/MILESTONES.md, project_security_audio_
stutter_gate): a CPU-inference thread — the openWakeWord `BargeInMonitor`
— now runs *during* the Python-fed TTS path. The stutter post-mortem's
mechanism: any CPU/GIL burst overlapping `speak_streaming` starves the
audio buffer, and the instrumented tell is `[audio] status: input
overflow` on the mic InputStream callback (the input-side proxy for the
uninstrumented TTS *output* underrun).

A passive "soak" is the wrong shape here: the barge-in monitor only runs
while Jarvis is mid-reply, so accumulating real exposure means hours of
live voice conversation, judged by ear. This harness instead does what
M44.3's leak_repro.py did for the OOM leak — it is an INSTRUMENT. It runs
the *real* `monitor_for_wake_word` concurrently with *real* back-to-back
`speak_streaming`, continuously, and counts input-overflow events. ~30
min of continuous TTS ≈ many hours of real use in monitor-during-TTS
exposure, and the verdict is a number, not a vibe.

WHAT IT MEASURES
----------------
Isolates exactly what M52 ADDED: the openWakeWord monitor's CPU load
during TTS. (The armed-mode worst case — monitor + YOLO watcher + TTS —
is left to a normal armed session: pre-M52 armed replies already coexist
with the YOLO watcher fine, so M52's delta really is just the monitor.)

  PASS  -> zero `input overflow` events across the run.
  CHECK -> any events: the timeline below shows whether they cluster
           (the post-mortem's stutter signature) or are isolated noise.

This is closing the box rigorously, not chasing a likely bug —
openWakeWord is a tiny ONNX model that caps its own threads, unlike the
uncapped-torch YOLO that actually caused the original stutter.

USAGE
-----
Run from the repo root with the project venv (so the sounddevice /
openWakeWord / edge-tts versions match production exactly). Plays audio
the whole time — you will hear it.

  venv\\Scripts\\python.exe scripts\\barge_stutter_soak.py            # 30 min
  venv\\Scripts\\python.exe scripts\\barge_stutter_soak.py --minutes 5  # quick check

Ctrl+C stops early and still prints the partial report.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

# This harness exercises the REAL src/ code paths, so the repo root must be
# importable (python scripts/x.py puts scripts/ on sys.path, not the root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio import AudioSession  # noqa: E402
from src.text_to_speech import speak_streaming  # noqa: E402
from src.wake_word import monitor_for_wake_word  # noqa: E402


# Filler the harness synthesizes back-to-back. Varied length for a realistic
# synth load; deliberately NO "Jarvis"/"Hey Jarvis" so the played audio can't
# self-trigger the monitor through mic echo.
_SENTENCES = [
    "The afternoon weather is holding clear with a light breeze from the west.",
    "Your calendar shows two meetings before lunch and one review at half past three.",
    "The quarterly figures came in slightly ahead of the earlier projection.",
    "I have queued the report and it should finish processing within the hour.",
    "The northbound route is congested, so the alternate road will be faster today.",
    "Both backup drives reported a clean verification overnight with no errors.",
    "The film opens to strong reviews and a respectable score from most critics.",
    "Rainfall is expected to taper off well before the evening commute begins.",
    "The package cleared the regional facility and is out for delivery this morning.",
    "Server load has stayed comfortably within normal range throughout the day.",
    "The recipe calls for a slow reduction over low heat for about twenty minutes.",
    "Ticket prices tend to settle a little in the final week before the event.",
    "The library extended its weekend hours through the end of the season.",
    "A modest software update is available and should install without a restart.",
    "The garden will need watering tonight given how warm the day turned out.",
    "Traffic cameras show the bridge moving freely in both directions now.",
]
_BATCH = 4  # sentences per speak_streaming call (~one realistic reply)


class _CountingAudioSession(AudioSession):
    """AudioSession that tallies sounddevice callback status flags. An
    `input overflow` is the post-mortem's instrumented tell that a CPU burst
    starved the audio path — the exact thing this soak watches for."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.overflow_count = 0
        self.status_events: list[tuple[float, str]] = []

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            with self._lock:
                self.status_events.append((time.monotonic() - self._t0, str(status)))
                if status.input_overflow:
                    self.overflow_count += 1
        # Still feed the queue so the barge-in monitor has audio to score.
        self._queue.put(indata[:, 0].copy())

    def snapshot(self) -> tuple[int, list[tuple[float, str]]]:
        with self._lock:
            return self.overflow_count, list(self.status_events)


def main() -> int:
    parser = argparse.ArgumentParser(description="M52 barge-in stutter soak.")
    parser.add_argument("--minutes", type=float, default=30.0,
                        help="soak duration in minutes (default 30)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="openWakeWord threshold for the monitor (default 0.5)")
    args = parser.parse_args()

    print(f"[soak] M52 barge-in stutter soak — {args.minutes:.0f} min, "
          f"monitor + back-to-back TTS concurrently. Audio will play.\n")

    session = _CountingAudioSession()
    monitor_stop = threading.Event()
    false_fires = [0]

    def _monitor_runner() -> None:
        """Run the REAL barge-in monitor continuously. monitor_for_wake_word
        returns on a (false) detection or when monitor_stop is set; restart it
        on the former so the openWakeWord CPU load is sustained for the whole
        soak, exit on the latter."""
        while not monitor_stop.is_set():
            # A throwaway interrupt_event — the monitor sets it on detection,
            # but here we only care that detection happened (it returns).
            monitor_for_wake_word(session, threading.Event(), monitor_stop,
                                  threshold=args.threshold)
            if not monitor_stop.is_set():
                false_fires[0] += 1  # openWakeWord fired on echo/ambient

    started = time.monotonic()
    deadline = started + args.minutes * 60.0
    batches = 0
    synth_errors = 0
    speak_event = threading.Event()  # never set — the faithful barge-in TTS path

    try:
        with session:
            monitor = threading.Thread(target=_monitor_runner,
                                       name="SoakMonitor", daemon=True)
            monitor.start()

            idx = 0
            while time.monotonic() < deadline:
                sentences = [_SENTENCES[(idx + i) % len(_SENTENCES)]
                             for i in range(_BATCH)]
                idx += _BATCH
                try:
                    # interrupt_event passed (unset) so the M52 non-blocking
                    # play + poll path runs exactly as in production.
                    speak_streaming(iter(sentences), "en",
                                    interrupt_event=speak_event)
                    batches += 1
                except Exception as exc:  # noqa: BLE001 — soak must not abort
                    synth_errors += 1
                    print(f"[soak] speak_streaming raised: "
                          f"{type(exc).__name__}: {exc}")

                elapsed = time.monotonic() - started
                ovf, _ = session.snapshot()
                print(f"[soak] {elapsed / 60:5.1f}/{args.minutes:.0f} min — "
                      f"{batches} TTS batches, {false_fires[0]} monitor (re)fires, "
                      f"{ovf} input-overflow event(s)")

            monitor_stop.set()
            monitor.join(timeout=3.0)
    except KeyboardInterrupt:
        print("\n[soak] interrupted — stopping and reporting partial result.")
        monitor_stop.set()

    elapsed = time.monotonic() - started
    overflow_count, status_events = session.snapshot()

    print("\n" + "=" * 64)
    print("  M52 BARGE-IN STUTTER SOAK — RESULT")
    print("=" * 64)
    print(f"  duration:               {elapsed / 60:.1f} min")
    print(f"  TTS batches played:     {batches}  (~{batches * _BATCH} sentences)")
    print(f"  speak_streaming errors: {synth_errors}  (edge-tts hiccups, tolerated)")
    print(f"  monitor (re)fires:      {false_fires[0]}  (openWakeWord on echo/ambient)")
    print(f"  est. monitor predicts:  ~{int(elapsed / 0.08):,}  (one per ~80 ms chunk)")
    print(f"  INPUT-OVERFLOW events:  {overflow_count}")
    if status_events:
        print("  --- all callback status events (t in s from start) ---")
        for ts, text in status_events:
            print(f"    {ts:8.1f}s  {text}")
    print("-" * 64)
    if overflow_count == 0:
        print("  VERDICT: PASS — no audio-buffer overflow under the M52")
        print("  monitor-during-TTS concurrency. Barge-in does not stutter")
        print("  the reply path. (Armed-mode stacking still wants a normal")
        print("  armed session — see the M52 entry in docs/MILESTONES.md.)")
        verdict = 0
    else:
        print(f"  VERDICT: CHECK — {overflow_count} input-overflow event(s).")
        print("  Inspect the timeline above: events CLUSTERED with TTS are the")
        print("  post-mortem's stutter signature; isolated singletons are")
        print("  likely system noise. If clustered, JARVIS_BARGE_IN=0 is the")
        print("  escape hatch while the monitor's load is investigated.")
        verdict = 1
    print("=" * 64)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
