"""Acoustic awareness (M58) — Jarvis's audio twin of M34 vision security.

The always-on mic listens for one pattern (the wake word); this module makes
Jarvis aware of every OTHER sound he hears — non-speech events worth speaking
up about: a doorbell, a knock, a smoke alarm, glass breaking, a phone ringing,
a kitchen timer, a sink left running. The reactive→proactive jump for the
audio channel. He saw the world (M34 cameras + M30 screen); now he hears it.

Architecture
------------
- **Its own `sounddevice.InputStream`** at 32 kHz mono int16. The existing
  AudioSession is single-reader-by-design (the wake-word loop owns it, with
  M52's barge-in monitor taking over during TTS); adding a third concurrent
  reader would race. Instead, M58 opens a second capture stream — supported
  on Windows/WASAPI for one mic device, negligible cost. Same "subsystem
  owns its own input" pattern security uses for the webcam and the homelab
  monitor uses for SSH. 32 kHz because that's Cnn14's training rate.
- **PANNs Cnn14** (Kong et al., audio tagging on AudioSet — 527 classes).
  We picked it after a measured de-risk: AST via the transformers pipeline
  is the strongest model but ~3.5 s steady-state per inference on this CPU,
  far too slow for a real-time alert loop. Cnn14 is a purpose-built audio-
  tagging CNN: ~325 MB checkpoint, ~50–100 ms CPU inference. Bundled in the
  `panns_inference` pip package.
- **Bypass the broken bundled downloader.** `panns_inference` shells out to
  `wget` for its labels CSV and checkpoint, which fails on Windows. We
  pre-place both via the project's existing `httpx` in `_ensure_files()`,
  then import the package normally. Same atomic-write discipline as the
  evidence saver: stream to .tmp, then `os.replace`.
- **Per-class alert hygiene** (the M56 `_CheckTracker` discipline,
  generalised with a wall-clock cooldown): every monitored class needs N
  consecutive windows above its confidence threshold before an alert fires;
  afterwards a per-class cooldown suppresses repeats. Sharp transients
  (doorbell, knock, glass) use a short sustain; ambient events (running
  water) use a long sustain.
- **The pet-fountain mitigation — a volume guard.** The user's pets have a
  continuously-running water fountain at room ambient levels. Without a
  guard, the `running_water` class would fire 24/7. So that one class is
  gated on *both* (a) classifier confidence and (b) audio RMS above a
  calibrated floor; a sink left on full is measurably louder than the
  fountain trickle. The other six classes are sharp transients naturally
  above ambient — no guard needed. The floor is env-tunable
  (`JARVIS_ACOUSTIC_WATER_RMS_FLOOR`) so we can re-calibrate against the
  actual fountain noise after the first real test.
- **Alerts ride `_announce`** (the WASAPI-safe Announcer path) tagged 🔔,
  plus a Discord push via the generic `send_discord_message` added in M56.
  Same dual-channel as homelab alerts.
- **Opt-in lifecycle** — tray toggle "Acoustic awareness" + `.env` flag
  `JARVIS_ACOUSTIC_MONITOR`. Default OFF. Consistent with security mode and
  homelab monitoring.

Defensive contract: nothing here raises into the audio callback or the
inference worker. A failed model load auto-disarms with an announce (like
SecurityWatcher's YOLO-load failure path); a bad inference logs and continues.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# panns_inference + torch are imported lazily inside _ensure_model(). torch is
# already in-process via ultralytics, but panns_inference's import-time code
# tries to read the labels CSV from ~/panns_data/ — so we pre-place the file
# BEFORE the import happens (see _ensure_files).


# --- Tunables --------------------------------------------------------------
# Read once at import (the M56 pattern). Subsystem-specific knobs live with
# the subsystem (vs. the core Config) — rationale and code stay co-located.

SAMPLE_RATE = 32_000           # PANNs Cnn14 training rate
WINDOW_SECONDS = 2.0           # one classifier inference per window — 2 s
                               # gives the model more context than 1 s
                               # (Cnn14 was trained on 10 s clips) without
                               # blowing the alert-latency budget.
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
BLOCK_SAMPLES = 2560           # 80 ms @ 32 kHz — same callback rate as the
                               # existing AudioSession.

# PANNs data location. The labels CSV path is HARDCODED in
# panns_inference/config.py so we have to use this exact spot.
PANNS_DIR = Path.home() / "panns_data"
LABELS_CSV = PANNS_DIR / "class_labels_indices.csv"
CHECKPOINT = PANNS_DIR / "Cnn14_mAP=0.431.pth"

LABELS_URL = ("http://storage.googleapis.com/us_audioset/youtube_corpus/v1"
              "/csv/class_labels_indices.csv")
CHECKPOINT_URL = ("https://zenodo.org/record/3987831/files/"
                  "Cnn14_mAP%3D0.431.pth?download=1")
CHECKPOINT_MIN_BYTES = 300_000_000   # ~325 MB; reject partial downloads


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_set(name: str) -> set[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


_DISABLED = _env_set("JARVIS_ACOUSTIC_DISABLE")  # per-class kill switch
# Running-water volume floor. Below this RMS the class can't fire — keeps the
# always-on pet fountain (a soft trickle, well below room speech levels) from
# being treated as a sink left on. Default 0.025 is a guessed-conservative
# starting point; expected to need tuning to the actual fountain after the
# first live test (the M44.1 lesson — guess loose, calibrate from a real
# measurement). The other six classes ignore this floor.
_RUNNING_WATER_RMS_FLOOR = _env_float("JARVIS_ACOUSTIC_WATER_RMS_FLOOR", 0.025)


# --- Per-class rules -------------------------------------------------------
# Each ClassRule names ONE logical alert ("doorbell", "smoke_alarm", …) and
# the AudioSet labels its score is aggregated from (the AudioSet ontology has
# 527 leaf labels, many of which are sub-categories of one logical event —
# e.g. "Smoke detector, smoke alarm" and "Fire alarm" both → smoke_alarm).
# Aliases were verified case-insensitively against the real PANNs label set
# during the M58 model de-risk; ALL fourteen aliases below match.

@dataclass(frozen=True)
class ClassRule:
    name: str                              # stable key
    aliases: tuple[str, ...]               # AudioSet labels to aggregate
    threshold: float                       # min summed score per window
    sustain: int                           # consecutive windows required
    cooldown_seconds: float                # silence after a fire
    speak: str                             # what Jarvis says aloud
    push: str                              # Discord one-liner
    needs_volume_guard: bool = False       # gate on RMS too (running_water)
    experimental: bool = False             # logs "experimental" on activate


# The default v1 allowlist — the seven the user confirmed. Thresholds + sustain
# are starting points; expect to tune after the first real test. Cnn14 outputs
# a per-class sigmoid score in [0, 1]; multi-label, so scores DO NOT sum to 1
# across classes — independent per-class probabilities. A strong event
# typically scores 0.3–0.8 on its primary label.
_DEFAULT_RULES: tuple[ClassRule, ...] = (
    ClassRule(
        name="doorbell",
        aliases=("Doorbell", "Ding-dong"),
        threshold=0.30, sustain=1, cooldown_seconds=25.0,
        speak="Sir — that sounded like the doorbell.",
        push="🔔 Doorbell",
    ),
    ClassRule(
        name="knock",
        aliases=("Knock",),
        threshold=0.25, sustain=1, cooldown_seconds=30.0,
        speak="Sir — there's someone at the door.",
        push="🚪 Knock at the door",
    ),
    ClassRule(
        name="smoke_alarm",
        aliases=("Smoke detector, smoke alarm", "Fire alarm"),
        threshold=0.30, sustain=2, cooldown_seconds=60.0,
        speak="Sir — I'm hearing a smoke alarm.",
        push="🚨 Smoke / fire alarm",
    ),
    ClassRule(
        name="glass_break",
        aliases=("Glass", "Shatter"),
        threshold=0.30, sustain=1, cooldown_seconds=45.0,
        speak="Sir — I just heard glass breaking.",
        push="💥 Glass breaking",
    ),
    ClassRule(
        name="phone_ringing",
        aliases=("Telephone bell ringing", "Ringtone", "Telephone"),
        threshold=0.25, sustain=2, cooldown_seconds=30.0,
        speak="Sir — your phone is ringing.",
        push="📞 Phone ringing",
    ),
    ClassRule(
        name="kitchen_timer",
        aliases=("Beep, bleep", "Buzzer"),
        threshold=0.30, sustain=2, cooldown_seconds=60.0,
        speak="Sir — that sounds like a kitchen timer going off.",
        push="⏲ Kitchen timer",
    ),
    ClassRule(
        name="running_water",
        # Long sustain (~8 windows = 16 s) + volume guard: a forgotten sink
        # runs persistently and is measurably louder than the pet fountain.
        # The volume floor is what filters out the always-on fountain, not
        # the score threshold alone.
        aliases=("Water tap, faucet", "Stream"),
        threshold=0.25, sustain=8, cooldown_seconds=600.0,
        speak="Sir — there's been running water for some time. Did you leave a tap on?",
        push="🚿 Running water (sink left on?)",
        needs_volume_guard=True,
        experimental=True,
    ),
)


# --- Per-class state machine ----------------------------------------------
# Generalisation of M56's _CheckTracker — adds a wall-clock cooldown clock
# (M56 had no cooldown because monitoring was edge-only; here a transient
# event firing two windows in a row should still alert exactly once).

class _ClassTracker:
    """Sustain-then-fire alert state machine with cooldown. Pure logic;
    trivially unit-testable. `observe(above_threshold, now)` returns True
    exactly on the window where the class should FIRE; otherwise False."""

    def __init__(self, sustain: int, cooldown_seconds: float) -> None:
        self._sustain = max(1, sustain)
        self._cooldown = max(0.0, cooldown_seconds)
        self._consecutive = 0
        self._cooldown_until = 0.0

    def observe(self, above_threshold: bool, now: float) -> bool:
        if not above_threshold:
            # A miss resets the consecutive counter; cooldown is wall-clock
            # so a streak break during cooldown still preserves the gate.
            self._consecutive = 0
            return False
        self._consecutive += 1
        if self._consecutive < self._sustain:
            return False
        if now < self._cooldown_until:
            # Sustain reached but still in cooldown — keep the streak alive
            # (a transient detection gap during cooldown shouldn't reset us)
            # but stay silent.
            return False
        self._cooldown_until = now + self._cooldown
        return True

    def reset(self) -> None:
        """Clean slate — called on (re)activate so a stale streak from a
        prior arm doesn't bleed into a fresh one."""
        self._consecutive = 0
        self._cooldown_until = 0.0


# --- File pre-placement (bypasses panns_inference's broken wget) ----------

def _ensure_files() -> str | None:
    """Pre-place the labels CSV + the Cnn14 checkpoint in ~/panns_data/.
    Returns None on success, or a human error string on permanent failure
    (the caller surfaces it as an announce). Atomic writes — partial
    downloads can't poison subsequent runs."""
    try:
        PANNS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"couldn't create {PANNS_DIR}: {exc}"

    # Labels CSV — small, fast. Must exist before `import panns_inference`,
    # whose config.py reads it at module-import time.
    if not LABELS_CSV.exists() or LABELS_CSV.stat().st_size < 1000:
        msg = _download(LABELS_URL, LABELS_CSV, expected_min_bytes=1000)
        if msg is not None:
            return f"labels CSV: {msg}"

    # Checkpoint — ~325 MB, one-time. The 3e8-byte sanity check matches
    # panns_inference's own minimum threshold (partial downloads are
    # rejected and re-fetched).
    if (not CHECKPOINT.exists()
            or CHECKPOINT.stat().st_size < CHECKPOINT_MIN_BYTES):
        print(f"[acoustic] downloading Cnn14 checkpoint (~325 MB, one-time) "
              f"to {CHECKPOINT} …", file=sys.stderr)
        msg = _download(CHECKPOINT_URL, CHECKPOINT,
                        expected_min_bytes=CHECKPOINT_MIN_BYTES)
        if msg is not None:
            return f"checkpoint: {msg}"
    return None


def _download(url: str, dest: Path, *, expected_min_bytes: int) -> str | None:
    """Stream a URL to `dest` atomically. Returns None on success or a
    voice-friendly error string on failure. Tmp + os.replace keeps a partial
    download from being mistaken for a complete one on the next run."""
    import httpx  # noqa: PLC0415 — lazy (already a project dep)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
            if resp.status_code >= 400:
                return f"HTTP {resp.status_code}"
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            last_log = time.monotonic()
            with tmp.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    # Log progress every ~10s for the big file.
                    now = time.monotonic()
                    if total > 10_000_000 and now - last_log > 10.0:
                        pct = (100.0 * done / total) if total else 0.0
                        print(f"[acoustic]   … {done/1e6:.1f} / "
                              f"{total/1e6:.1f} MB ({pct:.0f}%)",
                              file=sys.stderr)
                        last_log = now
    except Exception as exc:  # noqa: BLE001 — defensive
        try:
            tmp.unlink()
        except OSError:
            pass
        return f"download failed: {type(exc).__name__}: {exc}"
    if tmp.stat().st_size < expected_min_bytes:
        tmp.unlink()
        return (f"download too small ({tmp.stat().st_size} < "
                f"{expected_min_bytes}) — incomplete?")
    os.replace(tmp, dest)
    return None


# --- The detector ----------------------------------------------------------

@dataclass
class _ResolvedRule:
    """A ClassRule paired with the model-output indices its aliases resolved
    to at startup. Computed once when the model is loaded."""
    rule: ClassRule
    indices: tuple[int, ...]
    tracker: _ClassTracker


class SoundDetector:
    """The acoustic-awareness watcher. Construct always (the tray toggle
    needs the instance); the audio capture + inference loop only runs
    between `activate()` and `deactivate()`."""

    def __init__(
        self,
        announce: Callable[[str], None],
        *,
        discord_webhook_url: str = "",
        rules: tuple[ClassRule, ...] = _DEFAULT_RULES,
        device: int | None = None,
    ) -> None:
        self._announce = announce
        self._discord = (discord_webhook_url or "").strip()
        # Honour the per-class kill switch at construction (reads env once).
        self._rules = tuple(r for r in rules if r.name not in _DISABLED)
        self._device = device              # sounddevice device index; None = default
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None                # sounddevice.InputStream
        # Bounded queue: bursts during a slow inference don't grow unbounded.
        # If the worker falls behind we drop the OLDEST and enqueue the
        # newest — recent audio matters more than ancient backlog.
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        # Audio callback accumulates into this until it has WINDOW_SAMPLES,
        # then dispatches and resets. Single-thread access (only sounddevice's
        # callback touches it) — no lock needed.
        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._dropped = 0

        # Resolved at first activate (model load is heavy). Held across
        # activate/deactivate cycles so the second arm is instant.
        self._model = None
        self._labels: list[str] = []
        self._resolved: list[_ResolvedRule] = []
        self._model_load_failed = False

    # ---------------------------------------------------------------------
    # Public state
    # ---------------------------------------------------------------------

    def is_active(self) -> bool:
        return self._active.is_set()

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def activate(self) -> None:
        if self._active.is_set():
            return
        if not self._ensure_model():
            return  # already auto-announced its own failure
        for r in self._resolved:
            r.tracker.reset()
        self._buf.clear()
        self._buf_len = 0
        try:
            while True:
                self._chunks.get_nowait()
        except queue.Empty:
            pass
        try:
            import sounddevice as sd  # noqa: PLC0415 — lazy
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=BLOCK_SAMPLES,
                device=self._device,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            print(f"[acoustic] mic open failed: {exc}", file=sys.stderr)
            self._safe_announce(
                "I couldn't open the microphone for acoustic awareness, sir."
            )
            return
        self._active.set()
        self._stop.clear()
        names = ", ".join(r.rule.name for r in self._resolved) or "(no classes)"
        print(f"[acoustic] watching: {names}", file=sys.stderr)
        for r in self._resolved:
            if r.rule.experimental:
                print(f"[acoustic] note: '{r.rule.name}' is experimental — "
                      f"may need RMS-floor tuning", file=sys.stderr)
        self._safe_announce("Acoustic awareness on, sir.")
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._infer_loop, name="SoundDetector", daemon=True
            )
            self._thread.start()

    def deactivate(self) -> None:
        if not self._active.is_set():
            return
        self._active.clear()
        self._stop.set()
        self._close_stream()
        print("[acoustic] standing down", file=sys.stderr)
        self._safe_announce("Acoustic awareness off, sir.")

    def shutdown(self) -> None:
        """App-quit: silent wind-down. Daemon thread dies with the process
        anyway; this lets the inference loop exit its wait cleanly."""
        self._active.clear()
        self._stop.set()
        self._close_stream()

    def _close_stream(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop(); s.close()
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Model loading + AudioSet-label resolution
    # ---------------------------------------------------------------------

    def _ensure_model(self) -> bool:
        """Pre-place the data files, import panns_inference, instantiate
        AudioTagging, resolve each rule's aliases to model-output indices.
        Returns False (and auto-disarms with an announce) on permanent
        failure so the user gets a spoken heads-up rather than a silently
        broken activation."""
        if self._model is not None and self._resolved:
            return True
        if self._model_load_failed:
            return False

        # 1. Files in place BEFORE the import (panns_inference's config.py
        # reads the labels CSV at import time — wget-based fetch is broken
        # on Windows so we bypass it).
        err = _ensure_files()
        if err is not None:
            self._model_load_failed = True
            print(f"[acoustic] file setup failed: {err}", file=sys.stderr)
            self._safe_announce(
                "I couldn't fetch the acoustic-model files, sir. Check the logs."
            )
            return False

        # 2. Import + instantiate.
        t0 = time.monotonic()
        try:
            from panns_inference import AudioTagging, labels  # noqa: PLC0415
        except Exception as exc:
            self._model_load_failed = True
            print(f"[acoustic] panns_inference import failed: {exc}",
                  file=sys.stderr)
            self._safe_announce(
                "Acoustic awareness needs panns_inference, sir."
            )
            return False
        try:
            # device='cpu' — no GPU on this Ryzen 5 3400G; even if there
            # were one, this class is a small CPU-friendly CNN.
            self._model = AudioTagging(
                checkpoint_path=str(CHECKPOINT), device="cpu",
            )
        except Exception as exc:
            self._model_load_failed = True
            print(f"[acoustic] AudioTagging() failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            self._safe_announce(
                "I couldn't initialise the acoustic model, sir."
            )
            return False
        self._labels = list(labels)

        # 3. Resolve aliases → model output indices. Case-insensitive match
        # against the labels list. An alias that doesn't match is logged
        # but not fatal — a rule with at least one match still works.
        label_to_idx = {l.strip().lower(): i for i, l in enumerate(self._labels)}
        resolved: list[_ResolvedRule] = []
        for rule in self._rules:
            indices: list[int] = []
            missing: list[str] = []
            for alias in rule.aliases:
                idx = label_to_idx.get(alias.strip().lower())
                if idx is not None:
                    indices.append(idx)
                else:
                    missing.append(alias)
            if not indices:
                print(f"[acoustic] no AudioSet match for '{rule.name}' — "
                      f"skipping", file=sys.stderr)
                continue
            if missing:
                print(f"[acoustic] '{rule.name}' partial match — "
                      f"missing aliases: {missing}", file=sys.stderr)
            resolved.append(_ResolvedRule(
                rule=rule,
                indices=tuple(indices),
                tracker=_ClassTracker(rule.sustain, rule.cooldown_seconds),
            ))
        self._resolved = resolved
        print(f"[acoustic] model loaded in {time.monotonic() - t0:.1f}s — "
              f"{len(self._resolved)}/{len(self._rules)} rules resolved",
              file=sys.stderr)
        return True

    # ---------------------------------------------------------------------
    # Audio capture (sounddevice callback) — keep SHORT and never raise.
    # ---------------------------------------------------------------------

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ARG002
        if status:
            print(f"[acoustic] stream status: {status}", file=sys.stderr)
        chunk = indata[:, 0].copy()
        self._buf.append(chunk)
        self._buf_len += chunk.shape[0]
        if self._buf_len < WINDOW_SAMPLES:
            return
        window = np.concatenate(self._buf)[:WINDOW_SAMPLES]
        overshoot = self._buf_len - WINDOW_SAMPLES
        if overshoot > 0:
            tail = np.concatenate(self._buf)[WINDOW_SAMPLES:]
            self._buf = [tail]
            self._buf_len = overshoot
        else:
            self._buf = []
            self._buf_len = 0
        try:
            self._chunks.put_nowait(window)
        except queue.Full:
            # Worker is behind. Drop the OLDEST chunk and enqueue the newest
            # — recent audio matters more than ancient backlog. Logged at
            # most once per 30 dropped windows to avoid spam.
            try:
                self._chunks.get_nowait()
                self._chunks.put_nowait(window)
            except queue.Empty:
                pass
            self._dropped += 1
            if self._dropped % 30 == 1:
                print(f"[acoustic] inference falling behind "
                      f"(dropped {self._dropped} windows)", file=sys.stderr)

    # ---------------------------------------------------------------------
    # Inference worker
    # ---------------------------------------------------------------------

    def _infer_loop(self) -> None:
        print("[acoustic] inference loop started", file=sys.stderr)
        while not self._stop.is_set() and self._active.is_set():
            try:
                window = self._chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._classify_window(window)
            except Exception as exc:  # noqa: BLE001 — keep the loop alive
                print(f"[acoustic] classify failed: {exc}", file=sys.stderr)
        print("[acoustic] inference loop exited", file=sys.stderr)

    def _classify_window(self, window_int16: np.ndarray) -> None:
        # PANNs wants float32 in roughly [-1, 1], shape (batch, samples).
        audio = (window_int16.astype(np.float32)
                 / np.float32(32768.0))[None, :]
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        try:
            clipwise, _embedding = self._model.inference(audio)
        except Exception as exc:
            print(f"[acoustic] inference call failed: {exc}", file=sys.stderr)
            return
        # clipwise shape: (1, 527). Take the batch-of-one row.
        scores = clipwise[0]
        now = time.monotonic()
        for r in self._resolved:
            # Aggregate score across all of this rule's matched class indices
            # (a logical alert often corresponds to multiple AudioSet labels —
            # e.g. smoke_alarm = "Smoke detector, smoke alarm" + "Fire alarm").
            score = float(sum(scores[i] for i in r.indices))
            above = score >= r.rule.threshold
            # Volume guard for running_water. Even if the model says
            # "running water," refuse to count the window unless the room
            # is actually loud enough — keeps the pet fountain (a soft
            # continuous trickle, below the floor) from triggering an alert
            # the way a sink left on full would.
            if above and r.rule.needs_volume_guard and rms < _RUNNING_WATER_RMS_FLOOR:
                above = False
            if r.tracker.observe(above, now):
                self._fire(r.rule, score, rms)

    # ---------------------------------------------------------------------
    # Alert emission
    # ---------------------------------------------------------------------

    def _fire(self, rule: ClassRule, score: float, rms: float) -> None:
        print(f"[acoustic] FIRE {rule.name} (score={score:.2f} rms={rms:.3f})",
              file=sys.stderr)
        self._safe_announce(rule.speak)
        if self._discord:
            threading.Thread(
                target=self._push_discord, args=(rule.push,),
                name="AcousticNotify", daemon=True,
            ).start()

    def _push_discord(self, content: str) -> None:
        from src.notifications import send_discord_message  # noqa: PLC0415
        try:
            send_discord_message(self._discord, content)
        except Exception as exc:  # noqa: BLE001 — never break on a push
            print(f"[acoustic] discord push failed: {exc}", file=sys.stderr)

    def _safe_announce(self, text: str) -> None:
        try:
            self._announce(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[acoustic] announce failed: {exc}", file=sys.stderr)
