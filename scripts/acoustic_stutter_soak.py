r"""M67 — acoustic (PANNs) stutter soak harness, with a fidelity ladder.

WHY THIS EXISTS
---------------
The 2026-05-19 cooperative speech gate (project_security_audio_stutter_gate)
made heavy background CPU bursts and the Python-fed TTS path mutually
exclusive in time: any GIL/CPU burst overlapping `speak_streaming` starves the
audio buffer, and the instrumented tell is `[audio] input overflow` on a mic
InputStream callback (the input-side proxy for the uninstrumented TTS *output*
underrun). M58 acoustic awareness added a SECOND continuous CPU load — the
PANNs Cnn14 `_infer_loop` in src/sound_detector.py — shipped OUTSIDE the gate
AND coupled to armed mode, so the ungated PANNs loop starved the TTS path on
every armed announce: the M67 stutter REGRESSION. M67 re-fixed it by wiring
`announce_speaking` into SoundDetector (it DROPS the inference window while an
announce plays) + capping PANNs torch threads.

This is an INSTRUMENT in the lineage of M44.3's leak_repro.py and M52's
barge_stutter_soak.py: it runs the *real* code paths concurrently with *real*
back-to-back `speak_streaming` and counts input-overflow events. The verdict is
a number, not a vibe.

THE FIDELITY LADDER  (--coloads)
--------------------------------
A first isolated cut (PANNs + TTS only) could NOT reproduce the stutter even
ungated+uncapped — because on this 4-core box a lone PANNs inference (~0.2 s /
2 s window) leaves headroom. The regression is EMERGENT from the fuller armed
concurrency. So the harness layers the real co-loads that run DURING an armed
announce, and you climb the ladder to find which rung finally reproduces it:

  none      PANNs + TTS only. The isolated baseline (showed 0 overflow).
  wakeword  + the always-on openWakeWord main-listen loop on its own mic
            InputStream (it does NOT gate on speech — Jarvis must always be
            able to hear the wake word — so it competes DURING the announce,
            exactly the production during-announce picture).
  full      + a YOLOv8n watcher co-load (synthetic frames, 2 s cadence, the
            real del-results + gc.collect per tick). YOLO is GATED in
            production (defers during the announce), so it mainly establishes
            the armed steady-state / resident-model baseline; included for a
            faithful full-stack reproduction.

KEY: during an armed announce the YOLO watcher DEFERS (gated since 2026-05-19);
the loads actually competing with TTS are the UNGATED ones — openWakeWord
(main listen) + PANNs (ungated, the M67 bug). So `wakeword` is the rung that
should first expose the regression; `full` is the belt-and-braces full stack.

THE A/B  (--gate) — models whether the SPEAKING PATH raises the gate
--------------------------------------------------------------------
The real bug (found live 2026-06-01) was NOT that a consumer ignored the gate —
PANNs and the YOLO watcher are PERMANENT consumers that always defer when the
gate is set. It was that one SPEAKING PATH never SET the gate: proactive
announces (Announcer) raised `announce_speaking`, but TURN REPLIES (TurnRunner)
did not — so a reply spoken while armed left BOTH PANNs and YOLO running through
it, and they stuttered it. So this harness wires PANNs + YOLO as gate consumers
in BOTH arms and toggles only whether the TTS bracket raises the gate:

  --gate off  the pre-fix TURN-REPLY path: the TTS bracket does NOT set
              `announce_speaking`, so every armed consumer keeps running during
              the reply. EXPECT: input-overflow clustered with the TTS batches
              — the stutter signature. (This is the condition that reproduces
              the live bug; the live repro happened WITH torch caps at the
              default 1, so you do NOT need to uncap to see it.)
  --gate on   the FIX (and the announce path): the TTS bracket sets
              `announce_speaking` around playback exactly as TurnRunner now does
              (and the Announcer always did), so PANNs + YOLO defer. EXPECT:
              PASS — zero overflow.

The M67 fix also caps torch threads (JARVIS_ACOUSTIC_THREADS / JARVIS_YOLO_
THREADS, both default 1) — a SECOND, independent mitigation. You can additionally
uncap (export them =0) to stress harder, but it is NOT required to reproduce the
turn-reply stutter. Both effective caps are printed at startup so each result
log is self-documenting.

Run gate=off first to confirm a rung SEES the regression, then gate=on to
certify the fix closes it — proving the instrument before trusting it.

USAGE
-----
Run from the repo root with the project venv (so sounddevice / panns_inference /
torch / ultralytics / openwakeword / edge-tts match production exactly). Plays
audio and opens the production mic the whole time — you will hear it.

  # Reproduce the live turn-reply stutter (caps at default 1 — uncap NOT needed):
  venv\Scripts\python.exe scripts\acoustic_stutter_soak.py --coloads full --gate off --minutes 5

  # Certify the fix closes it (gate raised around playback, as TurnRunner now does):
  venv\Scripts\python.exe scripts\acoustic_stutter_soak.py --coloads full --gate on --minutes 30

  # The PANNs load alone is not enough — YOLO must ALSO be ungated (the live bug):
  ... --coloads none --gate off      # expect ~0 (PANNs-only, insufficient on a 4-core box)
  ... --coloads full --gate off      # both PANNs + YOLO ungated → reproduces

Ctrl+C stops early and still prints the partial report.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# This harness exercises the REAL src/ code paths, so the repo root must be
# importable (python scripts/x.py puts scripts/ on sys.path, not the root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv()  # so JARVIS_MIC_DEVICE matches the production pin
except Exception:  # noqa: BLE001 — .env is optional; default mic is fine
    pass

from src.audio import AudioSession, resolve_input_device  # noqa: E402
import src.sound_detector as sd_mod  # noqa: E402 — to report the import-time torch cap
from src.sound_detector import SoundDetector  # noqa: E402
from src.text_to_speech import speak_streaming  # noqa: E402
from src.wake_word import monitor_for_wake_word  # noqa: E402

_YOLO_POLL_SECONDS = 2.0  # mirror security.py's _POLL_SECONDS

# Filler the harness synthesizes back-to-back. Varied length for a realistic
# synth load; deliberately none of the acoustic alert words (doorbell, knock,
# glass, alarm, …) and no "Jarvis"/"Hey Jarvis" so the played audio can't
# self-trigger an acoustic FIRE or the openWakeWord monitor through mic echo.
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


class _CountingSoundDetector(SoundDetector):
    """SoundDetector that tallies input-overflow on its OWN 32 kHz capture
    callback — the post-mortem's instrumented tell that a CPU burst (the PANNs
    inference) starved the audio path. Overriding `_on_audio` keeps the real
    accumulate→dispatch→infer pipeline intact (we delegate to super after
    counting), so what we measure is the production loop, not a reconstruction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._soak_lock = threading.Lock()
        self._soak_t0 = time.monotonic()
        self.overflow_count = 0
        self.status_events: list[tuple[float, str]] = []

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            with self._soak_lock:
                self.status_events.append(
                    (time.monotonic() - self._soak_t0, "PANNs:" + str(status))
                )
                if status.input_overflow:
                    self.overflow_count += 1
        super()._on_audio(indata, frames, time_info, status)

    def soak_snapshot(self) -> tuple[int, list[tuple[float, str]]]:
        with self._soak_lock:
            return self.overflow_count, list(self.status_events)


class _CountingAudioSession(AudioSession):
    """The main-listen mic stream (feeds the openWakeWord monitor), tallying
    its own input-overflow — the same canary barge_stutter_soak watched."""

    def __init__(self, device: int | None) -> None:
        super().__init__(device=device)
        self._soak_lock = threading.Lock()
        self._soak_t0 = time.monotonic()
        self.overflow_count = 0
        self.status_events: list[tuple[float, str]] = []

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            with self._soak_lock:
                self.status_events.append(
                    (time.monotonic() - self._soak_t0, "listen:" + str(status))
                )
                if status.input_overflow:
                    self.overflow_count += 1
        self._queue.put(indata[:, 0].copy())

    def soak_snapshot(self) -> tuple[int, list[tuple[float, str]]]:
        with self._soak_lock:
            return self.overflow_count, list(self.status_events)


def _yolo_coload(stop: threading.Event, speaking: threading.Event,
                 status: dict) -> None:
    """Replicate the armed watcher's YOLO CPU load: YOLOv8n inference on a
    synthetic frame every ~2 s, with the real per-tick `del results` +
    gc.collect, and the SAME cooperative gate (defer while an announce plays —
    YOLO is gated in production). Caps torch threads exactly as security.py
    does (JARVIS_YOLO_THREADS, default 1). Failures are recorded, never raised."""
    try:
        from ultralytics import YOLO  # noqa: PLC0415
        model = YOLO("yolov8n.pt")
        raw = os.getenv("JARVIS_YOLO_THREADS", "").strip()
        n = int(raw) if raw.lstrip("-").isdigit() else 1
        if n >= 1:
            import torch  # noqa: PLC0415
            torch.set_num_threads(n)
            status["cap"] = f"capped to {n} thread(s)"
        else:
            status["cap"] = "UNCAPPED (all cores)"
        # A fixed synthetic BGR frame at a representative capture size; YOLO
        # resizes to imgsz=640 internally so inference cost is representative.
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}"
        return
    status["ticks"] = 0
    while not stop.is_set():
        # Gated consumer: defer the inference while an announce plays, exactly
        # like SecurityWatcher._wait_for_quiet. So YOLO competes BETWEEN
        # announces, not during them (the production picture).
        while speaking.is_set() and not stop.is_set():
            if stop.wait(0.2):
                return
        if stop.is_set():
            break
        try:
            results = model(frame, verbose=False)
            del results
        except Exception:  # noqa: BLE001 — keep the co-load alive
            pass
        gc.collect()
        status["ticks"] += 1
        stop.wait(_YOLO_POLL_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="M67 acoustic (PANNs) stutter soak.")
    parser.add_argument("--minutes", type=float, default=30.0,
                        help="soak duration in minutes (default 30)")
    parser.add_argument("--gate", choices=("on", "off"), default="on",
                        help="on = production fix (SoundDetector gets the speech "
                             "gate); off = pre-M67 regression repro (ungated PANNs)")
    parser.add_argument("--coloads", choices=("none", "wakeword", "full"),
                        default="full",
                        help="fidelity ladder: none=PANNs+TTS; wakeword=+openWakeWord "
                             "main-listen; full=+YOLO watcher (default full)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="openWakeWord threshold for the main-listen monitor")
    args = parser.parse_args()

    gate_on = args.gate == "on"
    want_wakeword = args.coloads in ("wakeword", "full")
    want_yolo = args.coloads == "full"
    fires = [0]

    def _on_fire(text: str) -> None:
        # No-op announce: the soak tests inference-vs-TTS contention, not
        # alerts. A stray acoustic FIRE (played speech tricking a class) is
        # logged, not spoken — we never want a second TTS path competing here.
        fires[0] += 1
        print(f"[soak] acoustic FIRE (ignored): {text!r}")

    # Resolve the SAME mic the production stack pins so stream/CPU contention is
    # faithful (JARVIS_MIC_DEVICE; blank → Windows default).
    mic_spec = os.getenv("JARVIS_MIC_DEVICE", "")
    device = resolve_input_device(mic_spec)

    # The shared speech-gate Event. PANNs (SoundDetector) and the YOLO co-load
    # are wired as gate CONSUMERS in BOTH arms — they're permanent consumers in
    # production. The A/B toggles only whether the TTS bracket RAISES the gate
    # (gate on = the fix / announce path; gate off = the pre-fix turn-reply path
    # that never raised it, leaving both consumers running through the reply).
    announce_speaking = threading.Event()
    stop = threading.Event()

    detector = _CountingSoundDetector(
        announce=_on_fire,
        discord_webhook_url="",  # no push — we measure contention, not alerts
        device=device,
        speaking_event=announce_speaking,  # permanent consumer in both arms
    )

    acoustic_cap = sd_mod._TORCH_THREADS  # import-time (JARVIS_ACOUSTIC_THREADS)
    acoustic_cap_desc = (f"{acoustic_cap} thread(s)" if acoustic_cap >= 1
                         else "UNCAPPED (all cores)")
    print(f"[soak] M67 acoustic stutter soak — {args.minutes:.0f} min, "
          f"coloads={args.coloads}, gate "
          f"{'ON (fix — TTS raises the gate)' if gate_on else 'OFF (pre-fix turn-reply path — TTS does NOT raise the gate)'}.")
    print(f"[soak] PANNs torch cap: {acoustic_cap_desc}")
    if gate_on and acoustic_cap < 1:
        print("[soak] NOTE: gate ON but PANNs torch uncapped — a non-production "
              "mix; the validation run should leave JARVIS_ACOUSTIC_THREADS at "
              "its default (1).", file=sys.stderr)

    print("[soak] loading PANNs Cnn14 (one-time ~325 MB download if missing) …")
    detector.activate()
    if not detector.is_active():
        print("[soak] SoundDetector failed to activate — model files missing or "
              "mic open failed (see [acoustic] logs above). Aborting.",
              file=sys.stderr)
        return 2
    # activate() emits its own greeting through our announce sink — zero the
    # count so the report reflects only the soak proper.
    fires[0] = 0

    # --- Optional co-loads ------------------------------------------------
    session: _CountingAudioSession | None = None
    monitor_thread: threading.Thread | None = None
    monitor_refires = [0]
    yolo_thread: threading.Thread | None = None
    yolo_status: dict = {}

    if want_wakeword:
        session = _CountingAudioSession(device)
        session.__enter__()

        def _monitor_runner() -> None:
            # Run the REAL openWakeWord main-listen loop continuously. It
            # returns on a (false) detection or when stop is set; restart on the
            # former so the openWakeWord CPU load is sustained for the whole soak.
            while not stop.is_set():
                monitor_for_wake_word(session, threading.Event(), stop,
                                      threshold=args.threshold)
                if not stop.is_set():
                    monitor_refires[0] += 1

        monitor_thread = threading.Thread(target=_monitor_runner,
                                          name="SoakListen", daemon=True)
        monitor_thread.start()
        print("[soak] co-load: openWakeWord main-listen monitor running.")

    if want_yolo:
        yolo_thread = threading.Thread(
            target=_yolo_coload, args=(stop, announce_speaking, yolo_status),
            name="SoakYOLO", daemon=True)
        yolo_thread.start()
        # Give YOLO a moment to load + report its cap/error.
        time.sleep(8.0)
        if yolo_status.get("error"):
            print(f"[soak] co-load: YOLO FAILED to load: {yolo_status['error']} "
                  f"— continuing WITHOUT the YOLO co-load.", file=sys.stderr)
            yolo_thread = None
        else:
            print(f"[soak] co-load: YOLOv8n watcher running "
                  f"({yolo_status.get('cap', '?')}).")

    print("[soak] starting back-to-back TTS. Audio will play.\n")

    started = time.monotonic()
    deadline = started + args.minutes * 60.0
    batches = 0
    synth_errors = 0
    reply_interrupt = threading.Event()  # never set — faithful reply path

    def _total_overflow() -> int:
        ovf, _ = detector.soak_snapshot()
        if session is not None:
            ovf += session.soak_snapshot()[0]
        return ovf

    try:
        idx = 0
        while time.monotonic() < deadline:
            sentences = [_SENTENCES[(idx + i) % len(_SENTENCES)]
                         for i in range(_BATCH)]
            idx += _BATCH
            # gate on  → raise the gate around playback (the fix / announce path).
            # gate off → DON'T raise it (the pre-fix turn-reply path): every armed
            #            consumer keeps running through the reply → reproduces.
            if gate_on:
                announce_speaking.set()
            try:
                speak_streaming(iter(sentences), "en",
                                interrupt_event=reply_interrupt)
                batches += 1
            except Exception as exc:  # noqa: BLE001 — soak must not abort
                synth_errors += 1
                print(f"[soak] speak_streaming raised: {type(exc).__name__}: {exc}")
            finally:
                if gate_on:
                    announce_speaking.clear()

            elapsed = time.monotonic() - started
            print(f"[soak] {elapsed / 60:5.1f}/{args.minutes:.0f} min — "
                  f"{batches} TTS batches, {fires[0]} acoustic fires, "
                  f"{monitor_refires[0]} listen refires, "
                  f"{_total_overflow()} input-overflow event(s)")
    except KeyboardInterrupt:
        print("\n[soak] interrupted — stopping and reporting partial result.")
    finally:
        stop.set()
        detector.shutdown()
        if monitor_thread is not None:
            monitor_thread.join(timeout=3.0)
        if yolo_thread is not None:
            yolo_thread.join(timeout=3.0)
        if session is not None:
            session.__exit__(None, None, None)

    elapsed = time.monotonic() - started
    pann_ovf, pann_events = detector.soak_snapshot()
    listen_ovf, listen_events = (session.soak_snapshot() if session
                                 else (0, []))
    overflow_count = pann_ovf + listen_ovf
    status_events = sorted(pann_events + listen_events)

    print("\n" + "=" * 64)
    print(f"  M67 ACOUSTIC STUTTER SOAK — RESULT  "
          f"(coloads={args.coloads}, gate {args.gate.upper()})")
    print("=" * 64)
    print(f"  duration:               {elapsed / 60:.1f} min")
    print(f"  TTS batches played:     {batches}  (~{batches * _BATCH} sentences)")
    print(f"  speak_streaming errors: {synth_errors}  (edge-tts hiccups, tolerated)")
    print(f"  acoustic fires:         {fires[0]}  (played speech tricking a class)")
    if want_wakeword:
        print(f"  listen monitor refires: {monitor_refires[0]}  (openWakeWord on echo/ambient)")
    if want_yolo:
        print(f"  YOLO co-load ticks:     {yolo_status.get('ticks', 'n/a')}  "
              f"({yolo_status.get('cap', 'not loaded')})")
    print(f"  INPUT-OVERFLOW events:  {overflow_count}  "
          f"(PANNs stream {pann_ovf} + listen stream {listen_ovf})")
    if status_events:
        print("  --- all callback status events (t in s from start) ---")
        for ts, text in status_events:
            print(f"    {ts:8.1f}s  {text}")
    print("-" * 64)
    if gate_on:
        if overflow_count == 0:
            print("  VERDICT: PASS — zero audio-buffer overflow with the speech")
            print(f"  gate RAISED around playback, at coloads={args.coloads}. The armed")
            print("  PANNs + YOLO loops defer, so the spoken reply does NOT stutter")
            print("  (cf. gate off at the same rung, which reproduces it).")
            verdict = 0
        else:
            print(f"  VERDICT: CHECK — {overflow_count} input-overflow event(s)")
            print("  WITH the gate on. Inspect the timeline: events CLUSTERED with")
            print("  the TTS batches are the stutter signature (the gate isn't")
            print("  fully closing the overlap — e.g. an inference already in")
            print("  flight when the announce starts); isolated singletons are")
            print("  likely OS-scheduling noise (the barge soak saw 1 in 30 min).")
            verdict = 1
    else:
        print("  CONTROL RUN (gate OFF — the pre-fix TURN-REPLY path: the TTS")
        print("  bracket never raises the gate, so PANNs + YOLO run through the")
        print("  reply). Overflow CLUSTERED with the TTS batches reproduces the")
        print("  live 2026-06-01 stutter and proves this rung SEES the bug. Note")
        print("  it needs --coloads full: PANNs alone (--coloads none) is not")
        print("  enough on a 4-core box; YOLO must ALSO be ungated. Re-run")
        print("  --gate on to certify the fix closes it.")
        print(f"  ({overflow_count} overflow event(s) observed.)")
        verdict = 0
    print("=" * 64)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
