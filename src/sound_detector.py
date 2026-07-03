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
from collections import deque
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


_DISABLED = _env_set("JARVIS_ACOUSTIC_DISABLE")  # per-class kill switch
# Running-water volume floor. Below this RMS the class can't fire — keeps the
# always-on pet fountain (a soft trickle, well below room speech levels) from
# being treated as a sink left on. Default 0.025 is a guessed-conservative
# starting point; expected to need tuning to the actual fountain after the
# first live test (the M44.1 lesson — guess loose, calibrate from a real
# measurement). The other six classes ignore this floor.
_RUNNING_WATER_RMS_FLOOR = _env_float("JARVIS_ACOUSTIC_WATER_RMS_FLOOR", 0.025)
# Debug: set JARVIS_ACOUSTIC_DEBUG=1 to log the top AudioSet labels + scores for
# any non-quiet window. The data-driven way to tune a class (e.g. find what a
# table-knock actually classifies as — likely "Tap"/"Wood", not "Knock") instead
# of guessing thresholds. Off by default (would spam the log).
_DEBUG_TOPK = os.getenv("JARVIS_ACOUSTIC_DEBUG", "").strip() not in ("", "0", "false", "False")
_DEBUG_MIN_SCORE = 0.10  # only log a window whose strongest label clears this
# Loudness floor for the transient events (knock/doorbell/glass). The model
# occasionally hallucinates e.g. "Knock"=0.43 on a near-SILENT window
# (rms~0.002) — a false "someone's at the door" while away. Live data
# (2026-06-03) separated the classes: spurious fires peaked at rms 0.009, real
# events ran 0.033–0.32. The DOORBELL is the quietest real event (rms 0.033)
# because Windows Voice Focus (kept on "Automatic" for voice clarity) attenuates
# non-speech — so the floor must leave margin BELOW the real doorbell, not hug
# it. 0.02 sits in the gap (rejects ≤0.009 hallucinations, passes the 0.033
# doorbell with room to spare for further Voice-Focus attenuation). Leans toward
# detection on purpose: a missed knock/doorbell while away is worse than a rare
# false ping. Same idea as the running-water volume guard, generalised.
# Env-tunable per setup.
_EVENT_RMS_FLOOR = _env_float("JARVIS_ACOUSTIC_RMS_FLOOR", 0.02)

# M81 — armed intrusion-by-voice. While security is ARMED (away), human speech
# in the house is an anomaly worth a look: the voice_while_armed rule fires the
# same M72 visual alert (photo + Claude description → Discord) as the sound
# rules. This is the summed-score threshold for the "Speech"/"Conversation"/
# "Shout" AudioSet labels. Raised 0.45 → 0.60 (2026-06-15) after a live armed
# session false-fired repeatedly on adjacent unit/hallway voices bleeding through
# the walls — but note the SCORE is the weak lever here: that bleed IS genuine
# speech, so it scores high (0.65–1.15 in the log); raising this further barely
# helps. The real discriminator is loudness — see _VOICE_RMS_FLOOR below.
_VOICE_THRESHOLD = _env_float("JARVIS_ACOUSTIC_VOICE_THRESHOLD", 0.60)

# Loudness floor SPECIFIC to the voice rule — the through-the-wall discriminator.
# A voice in the room is physically louder than a adjacent unit's voice through a
# wall or down the hallway. Live data (2026-06-15, armed, space empty, no TV)
# showed adjacent unit/hallway bleed firing at rms 0.020–0.044 — right on top of the
# shared transient-event floor (_EVENT_RMS_FLOOR=0.02, tuned for the doorbell at
# 0.033). The voice rule needs its OWN, higher floor sitting ABOVE that bleed
# ceiling: 0.06 rejects the observed bleed with margin, while a person actually
# standing in the room talking near the omni boundary mic clears it easily. If a
# real in-room voice is ever missed while armed, lower this (calibrate with
# JARVIS_ACOUSTIC_DEBUG=1 — speak in the room while armed and read the real rms).
# A missed fire only costs a photo, not a deterrent, so we bias slightly toward
# fewer false pings. Generalises the running-water volume guard to the voice rule.
_VOICE_RMS_FLOOR = _env_float("JARVIS_ACOUSTIC_VOICE_RMS_FLOOR", 0.06)

# Cap torch's intra-op thread pool for PANNs inference. Same rationale as the
# armed watcher's JARVIS_YOLO_THREADS cap (2026-05-19 stutter post-mortem): an
# uncapped torch inference pins all cores and starves the real-time audio
# threads → `input overflow` → TTS stutter. set_num_threads is process-global;
# when security is ALSO armed, YOLO sets it too (same value) — but acoustic can
# run alone (the independent tray toggle / armed-while-home), so PANNs must cap
# it independently. The old "YOLO is the only torch consumer" assumption in
# security.py is FALSE since M58. <= 0 leaves torch's default (debug/parity).
_TORCH_THREADS = _env_int("JARVIS_ACOUSTIC_THREADS", 1)

# Recent-sounds ring buffer (M76 — the what_did_you_hear tool). Each salient
# window records its dominant AudioSet label so the user can ask "what did you
# just hear?" after an alert. Capacity bounds memory (tiny tuples); the tool
# filters by age, so this just needs to comfortably cover the longest report
# window (~15 min) of CONTINUOUS salient sound — far more than ever happens,
# since quiet/silent windows aren't recorded.
_RECENT_MAXLEN = 450
# Don't record a window unless its top label clears this — keeps "Silence" and
# near-quiet noise out of the soundscape (a touch above the 0.10 debug-log floor).
_RECENT_MIN_SCORE = 0.15
# AudioSet labels that are "nothing happened" — never worth recording as a
# heard sound even if they score high (a quiet room scores "Silence" near 1.0).
_RECENT_IGNORE_LABELS = {"silence"}


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
    rms_floor: float = 0.0                 # min window RMS to fire (0 = no floor);
                                           # rejects model hallucinations on silence
    armed_only: bool = False               # M81 — only count windows while
                                           # security is ARMED (away). For the
                                           # voice-intrusion rule: speech is
                                           # normal at home, anomalous away.


# The active allowlist — the THREE the user wants while armed (2026-06-03):
# knock, doorbell, glass breaking. Acoustic awareness is coupled to armed mode
# (armed = away = listen), and these are the away-relevant events: someone at
# the door (knock/doorbell — though a Ring doorbell usually flags a real
# visitor first) and the high-priority GLASS BREAK, which triggers the M72
# multimodal alert (photo + description to Discord) so the user can see what
# Jarvis sees and cross-check the Ring indoor camera. The other four events
# (smoke/phone/timer/water) are defined in _OTHER_RULES below but NOT active —
# add one back to _DEFAULT_RULES to re-enable.
#
# Thresholds + sustain are starting points; expect to tune (knock especially —
# a table-knock may classify as Tap/Wood, not "Knock"; see the knock-tuning
# follow-on). Cnn14 outputs a per-class sigmoid score in [0, 1]; multi-label,
# so scores DO NOT sum to 1 — independent per-class probabilities. A strong
# event typically scores 0.3–0.8 on its primary label.
_DEFAULT_RULES: tuple[ClassRule, ...] = (
    ClassRule(
        name="doorbell",
        aliases=("Doorbell", "Ding-dong"),
        threshold=0.30, sustain=1, cooldown_seconds=25.0,
        speak="Sir — that sounded like the doorbell.",
        push="🔔 Doorbell",
        rms_floor=_EVENT_RMS_FLOOR,
    ),
    ClassRule(
        name="knock",
        aliases=("Knock",),
        threshold=0.25, sustain=1, cooldown_seconds=30.0,
        speak="Sir — there's someone at the door.",
        push="🚪 Knock at the door",
        rms_floor=_EVENT_RMS_FLOOR,
    ),
    ClassRule(
        name="glass_break",
        # "Breaking" added 2026-06-03: a faint break scored Glass 0.19 +
        # Shatter 0.09 = 0.28 (just under threshold, missed) but had
        # Breaking 0.08 — aggregating it clears 0.30. Deliberately NOT adding
        # "Chink, clink" (fires on normal glasses touching → false positives).
        # Glass is the priority event (triggers the M72 photo so the user can
        # check the Ring indoor cam), so bias toward catching a faint break.
        aliases=("Glass", "Shatter", "Breaking"),
        threshold=0.30, sustain=1, cooldown_seconds=45.0,
        speak="Sir — I just heard glass breaking.",
        push="💥 Glass breaking",
        rms_floor=_EVENT_RMS_FLOOR,
    ),
    # M81 — armed intrusion-by-voice. ARMED_ONLY: it never counts a window at
    # home (where talking is normal), only while away — so a voice in the
    # supposedly-empty house fires the same M72 look-and-push as a glass break.
    # sustain=2 (~4 s of sustained speech) rejects a one-off blip (a single
    # notification read aloud, a passing car stereo); the RMS floor rejects
    # faint TV/adjacent unit bleed and the model's hallucinated speech on near-
    # silence. cooldown 120 s ⇒ at most one photo every 2 min during a
    # continuous intrusion. Jarvis's OWN announces/replies are already gated out
    # of inference by the speech gate (`_is_announcing`), so he can't self-fire
    # on his reply. experimental: thresholds want a live tune (the M58 lesson).
    ClassRule(
        name="voice_while_armed",
        aliases=("Speech", "Conversation", "Shout"),
        threshold=_VOICE_THRESHOLD, sustain=2, cooldown_seconds=120.0,
        speak="Sir — I'm hearing a voice, and the house is armed.",
        push="🗣 Voice heard while armed — check the snapshot.",
        # Its OWN loudness floor (above the adjacent unit-bleed ceiling), NOT the
        # shared transient-event floor — see _VOICE_RMS_FLOOR.
        rms_floor=_VOICE_RMS_FLOOR,
        armed_only=True,
        experimental=True,
    ),
)


# Defined but NOT in the active set (the user scoped acoustic to the three
# above on 2026-06-03). Kept here so re-enabling any is a one-line move into
# _DEFAULT_RULES, with the tuned aliases/thresholds preserved.
_OTHER_RULES: tuple[ClassRule, ...] = (
    ClassRule(
        name="smoke_alarm",
        aliases=("Smoke detector, smoke alarm", "Fire alarm"),
        threshold=0.30, sustain=2, cooldown_seconds=60.0,
        speak="Sir — I'm hearing a smoke alarm.",
        push="🚨 Smoke / fire alarm",
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


def _rule_window_passes(rule: ClassRule, score: float, rms: float,
                        armed: bool) -> bool:
    """Pure per-window gate for ONE rule: does this window count toward the
    rule's sustain streak? Applies, in order, the score threshold, the loudness
    floor (rejects the model's hallucinated high score on near-silence), the
    running-water volume guard, and (M81) the armed-only gate. No model, no
    state, no I/O — trivially unit-testable (like _ClassTracker)."""
    if score < rule.threshold:
        return False
    if rule.rms_floor > 0.0 and rms < rule.rms_floor:
        return False
    if rule.needs_volume_guard and rms < _RUNNING_WATER_RMS_FLOOR:
        return False
    if rule.armed_only and not armed:
        return False
    return True


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
    size = tmp.stat().st_size
    if size < expected_min_bytes:
        tmp.unlink()
        return (f"download too small ({size} < "
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


# 2026-07-02 QA: minimum seconds between stream-status log lines (see
# SoundDetector._on_audio) — the count since the last line rides along.
_STATUS_LOG_INTERVAL_SEC = 60.0


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
        speaking_event: "threading.Event | None" = None,
        on_visual_alert: "Callable[[str], None] | None" = None,
        is_armed: "Callable[[], bool] | None" = None,
    ) -> None:
        self._announce = announce
        self._discord = (discord_webhook_url or "").strip()
        # M81 — armed-state getter (SecurityWatcher.is_armed). An armed_only rule
        # (voice_while_armed) only counts windows when this returns True. None or
        # a raising getter ⇒ treated as NOT armed (fail-safe: never fire an
        # intrusion alert when we can't confirm the house is armed).
        self._is_armed = is_armed
        # M72 — multimodal alert hook: on a fired event, ALSO look through the
        # camera and push a photo + Claude description to Discord. main.py wires
        # this to a function that's a no-op unless armed (the camera is the
        # watcher's then). Optional + fail-soft; None ⇒ text-only (pre-M72).
        self._on_visual_alert = on_visual_alert
        # Honour the per-class kill switch at construction (reads env once).
        self._rules = tuple(r for r in rules if r.name not in _DISABLED)
        self._device = device              # sounddevice device index; None = default
        # Cooperative speech gate (the 2026-05-19 post-mortem, extended to
        # acoustic in this session). Set by the Announcer thread WHILE a
        # proactive announce is playing. The inference worker SKIPS the heavy
        # PANNs burst whenever this is set, so it never overlaps the Python-fed
        # TTS path and stutters the announce — the same fix SecurityWatcher
        # applies to its vision bursts. None → gate disabled (degrades to
        # pre-fix behaviour; never breaks the loop).
        self._speaking_event = speaking_event
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None                # sounddevice.InputStream
        # Bounded queue: bursts during a slow inference don't grow unbounded.
        # If the worker falls behind we drop the OLDEST and enqueue the
        # newest — recent audio matters more than ancient backlog.
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        # Audio callback accumulates into this until it has WINDOW_SAMPLES,
        # then dispatches and resets. No lock: while the stream runs ONLY the
        # sounddevice callback touches it; activate() clears it too, but always
        # BEFORE stream.start() (and re-activate early-returns), so the callback
        # can never be mid-append when the clear runs. Preserve that ordering.
        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._dropped = 0
        # 2026-07-02 QA: rate-limited stream-status (overflow) logging — see
        # _on_audio. Counter + last-emit timestamp, callback-thread only.
        self._status_count = 0
        self._status_last_log = float("-inf")
        # 2026-07-02 QA: serialize activate()/deactivate() — the idempotency
        # check and _active.set() are separated by seconds of model-load/
        # stream-open work, so two near-simultaneous activations (presence
        # auto-arm + a tray toggle) could BOTH open an InputStream and orphan
        # one forever.
        self._lifecycle_lock = threading.Lock()

        # Resolved at first activate (model load is heavy). Held across
        # activate/deactivate cycles so the second arm is instant.
        self._model = None
        self._labels: list[str] = []
        self._resolved: list[_ResolvedRule] = []
        self._model_load_failed = False

        # M76 — recent-sounds ring buffers for the what_did_you_hear tool.
        # Written by the inference thread (_classify_window / _fire), read by
        # the turn-worker thread (recent_sounds_summary) → guarded by a lock.
        # `_recent` = (monotonic_ts, label, score, rms) per salient window;
        # `_recent_fires` = (monotonic_ts, rule_name) per fired alert.
        self._recent_lock = threading.Lock()
        self._recent: deque = deque(maxlen=_RECENT_MAXLEN)
        self._recent_fires: deque = deque(maxlen=64)

    # ---------------------------------------------------------------------
    # Public state
    # ---------------------------------------------------------------------

    def is_active(self) -> bool:
        return self._active.is_set()

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def activate(self) -> None:
        with self._lifecycle_lock:
            self._activate_locked()

    def _activate_locked(self) -> None:
        if self._active.is_set():
            return
        # 2026-07-02 QA: a rapid deactivate→activate could observe the OLD
        # inference thread still mid-exit (it polls _stop every 0.5 s), skip
        # the spawn below, and end up active with NO inference loop. Join the
        # dying thread first so the is_alive() check below is truthful.
        if (self._thread is not None and self._stop.is_set()
                and self._thread.is_alive()):
            self._thread.join(timeout=2.0)
        if not self._ensure_model():
            return  # already auto-announced its own failure
        for r in self._resolved:
            r.tracker.reset()
        self._buf.clear()
        self._buf_len = 0
        # Fresh session — a "what did you hear?" right after re-arming should
        # not report sounds from a prior arm.
        with self._recent_lock:
            self._recent.clear()
            self._recent_fires.clear()
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
                # 2026-07-02 QA: nothing downstream is latency-sensitive (the
                # consumer assembles 2 s windows), but the sounddevice default
                # requests the device's LOW-latency ring — a few tens of ms —
                # which the armed-mode PANNs+YOLO bursts overflowed every ~3-4 s
                # for the entire armed window (dropped samples = gap-corrupted
                # detection windows). 'high' gives PortAudio a deep host buffer
                # that absorbs those bursts; detection semantics are unchanged.
                latency="high",
            )
            self._stream.start()
        except Exception as exc:
            print(f"[acoustic] mic open failed: {exc}", file=sys.stderr)
            # A constructed-but-unstarted stream must not leak (start() can
            # raise after the constructor succeeded).
            self._close_stream()
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
        with self._lifecycle_lock:
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

        # 2b. Cap torch's thread pool so a PANNs inference can't pin all cores
        # and starve the real-time audio threads (see _TORCH_THREADS above).
        # Process-global, but harmless if YOLO sets it again to the same value.
        if _TORCH_THREADS >= 1:
            try:
                import torch  # type: ignore  # noqa: PLC0415
                torch.set_num_threads(_TORCH_THREADS)
                print(f"[acoustic] torch capped to {_TORCH_THREADS} thread(s)",
                      file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 — never block on this
                print(f"[acoustic] could not cap torch threads: {exc}",
                      file=sys.stderr)

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
            # 2026-07-02 QA: rate-limited. While armed this fired every ~3-4 s
            # for the whole armed window (~1-2k log lines/day), and each print
            # is a timestamped FILE write inside the 80 ms-budget callback —
            # lengthening the very callback whose lateness caused the overflow.
            self._status_count += 1
            now = time.monotonic()
            if now - self._status_last_log >= _STATUS_LOG_INTERVAL_SEC:
                print(f"[acoustic] stream status: {status} "
                      f"(x{self._status_count} since last report)",
                      file=sys.stderr)
                self._status_count = 0
                self._status_last_log = now
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
            # Catch BOTH queue.Empty (queue drained between Full and get) AND
            # queue.Full (refilled before our put): a raise here escapes into
            # the sounddevice callback and would ABORT the input stream. Single-
            # producer makes the Full case unreachable today, but guarding it is
            # cheap insurance against a high-consequence, low-probability abort.
            try:
                self._chunks.get_nowait()
                self._chunks.put_nowait(window)
            except (queue.Empty, queue.Full):
                pass
            self._dropped += 1
            if self._dropped % 30 == 1:
                print(f"[acoustic] inference falling behind "
                      f"(dropped {self._dropped} windows)", file=sys.stderr)

    # ---------------------------------------------------------------------
    # Inference worker
    # ---------------------------------------------------------------------

    def _is_announcing(self) -> bool:
        """True while a proactive announce is on the wire. The cooperative
        speech gate: the heavy PANNs inference must not overlap this (it
        starves the Python-fed TTS path → audio stutter). None event → always
        False, i.e. gate disabled, never breaks the loop."""
        return self._speaking_event is not None and self._speaking_event.is_set()

    def _infer_loop(self) -> None:
        print("[acoustic] inference loop started", file=sys.stderr)
        while not self._stop.is_set() and self._active.is_set():
            try:
                window = self._chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            # Cooperative speech gate: while a proactive announce plays, DROP
            # this window instead of classifying it — a PANNs inference burst
            # overlapping the TTS path underruns the audio buffer and stutters
            # the announce. We're "deaf" for the ~1-3 s an announce lasts, the
            # same bounded tradeoff the security watcher accepts; the bounded
            # input queue means we resume on recent audio, not a stale backlog.
            if self._is_announcing():
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
        # Debug aid (JARVIS_ACOUSTIC_DEBUG=1): log what PANNs actually hears so
        # a class can be tuned from data, not guesswork. Top-6 labels for any
        # window above the floor — knock on the table while this is on and read
        # the labels your knock produces.
        if _DEBUG_TOPK and self._labels:
            try:
                top = np.argsort(scores)[::-1][:6]
                if float(scores[top[0]]) >= _DEBUG_MIN_SCORE:
                    pairs = ", ".join(
                        f"{self._labels[i]}={float(scores[i]):.2f}" for i in top
                    )
                    print(f"[acoustic] top: {pairs} (rms={rms:.3f})", file=sys.stderr)
            except Exception:  # noqa: BLE001 — debug log must never break the loop
                pass
        # M76 — record the dominant sound of this window for what_did_you_hear.
        # Cheap (one argmax); skips silence and near-quiet so the soundscape
        # reflects real events, not the floor. Never breaks the loop.
        self._record_observation(scores, rms, now)
        # M81 — armed state, sampled ONCE per window (not per rule). An
        # armed_only rule (voice_while_armed) only counts a window while armed;
        # a missing/raising getter ⇒ NOT armed (fail-safe — no intrusion alert
        # unless we can confirm the house is armed).
        armed = False
        if self._is_armed is not None:
            try:
                armed = bool(self._is_armed())
            except Exception as exc:  # noqa: BLE001 — never break the loop
                print(f"[acoustic] is_armed() raised: {exc}", file=sys.stderr)
        for r in self._resolved:
            # Aggregate score across all of this rule's matched class indices
            # (a logical alert often corresponds to multiple AudioSet labels —
            # e.g. smoke_alarm = "Smoke detector, smoke alarm" + "Fire alarm").
            score = float(sum(scores[i] for i in r.indices))
            # Threshold + loudness floor + running-water volume guard + the
            # M81 armed-only gate, all in the pure _rule_window_passes helper.
            above = _rule_window_passes(r.rule, score, rms, armed)
            if r.tracker.observe(above, now):
                self._fire(r.rule, score, rms)

    # ---------------------------------------------------------------------
    # Alert emission
    # ---------------------------------------------------------------------

    def _fire(self, rule: ClassRule, score: float, rms: float) -> None:
        print(f"[acoustic] FIRE {rule.name} (score={score:.2f} rms={rms:.3f})",
              file=sys.stderr)
        # M76 — record the fired alert so what_did_you_hear can report it with
        # certainty ("you heard the doorbell ~40s ago"), distinct from the
        # general soundscape.
        with self._recent_lock:
            self._recent_fires.append((time.monotonic(), rule.name))
        self._safe_announce(rule.speak)
        if self._discord:
            threading.Thread(
                target=self._push_discord, args=(rule.push,),
                name="AcousticNotify", daemon=True,
            ).start()
        # M72 — multimodal: look + describe + push a photo. On its OWN thread
        # (it does a ~2s vision call) so it never blocks the inference loop or
        # the text push. The hook itself is gated on armed (no-op at home).
        if self._on_visual_alert is not None:
            threading.Thread(
                target=self._safe_visual, args=(rule.name,),
                name="AcousticVisual", daemon=True,
            ).start()

    def _safe_visual(self, event_name: str) -> None:
        try:
            self._on_visual_alert(event_name)
        except Exception as exc:  # noqa: BLE001 — a proactive extra must never break
            print(f"[acoustic] visual alert failed: {exc}", file=sys.stderr)

    def _push_discord(self, content: str) -> None:
        from src.notifications import send_discord_message  # noqa: PLC0415
        try:
            send_discord_message(self._discord, content)
        except Exception as exc:  # noqa: BLE001 — never break on a push
            print(f"[acoustic] discord push failed: {exc}", file=sys.stderr)

    def _record_observation(self, scores: np.ndarray, rms: float, now: float) -> None:
        """Append this window's dominant sound to the recent ring buffer for
        the what_did_you_hear tool. Skips silence / near-quiet so the
        soundscape reflects real events. Never raises (proactive bookkeeping
        must not break the inference loop)."""
        if not self._labels:
            return
        try:
            top_idx = int(np.argmax(scores))
            top_score = float(scores[top_idx])
            if top_score < _RECENT_MIN_SCORE:
                return
            label = self._labels[top_idx]
            if label.strip().lower() in _RECENT_IGNORE_LABELS:
                return
            with self._recent_lock:
                self._recent.append((now, label, top_score, rms))
        except Exception as exc:  # noqa: BLE001 — bookkeeping must not break the loop
            print(f"[acoustic] record observation failed: {exc}", file=sys.stderr)

    def recent_sounds_summary(self, max_age_seconds: float = 180.0) -> str:
        """Voice-friendly summary of recently-heard sounds for the
        what_did_you_hear tool. Snapshots the buffers under the lock, then
        defers to the pure `_summarize_sounds` formatter (testable without
        threads)."""
        now = time.monotonic()
        active = self._active.is_set()
        with self._recent_lock:
            recent = list(self._recent)
            fires = list(self._recent_fires)
        return _summarize_sounds(active, recent, fires, now, max_age_seconds)

    def _safe_announce(self, text: str) -> None:
        try:
            self._announce(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[acoustic] announce failed: {exc}", file=sys.stderr)


# --- what_did_you_hear tool (M76) ------------------------------------------
# Completes the acoustic arc: M58 DETECTS sounds, M72 REACTS (look + describe +
# push while armed), and this lets the user RECALL on demand — "Jarvis, what
# did you just hear?" / "did you hear something?". The SoundDetector keeps a
# rolling buffer of recent dominant sounds + fired alerts; this tool reads it.
#
# Decoupled from main.py via a module-level singleton (the self_status.register
# / good_night.register_security_getter pattern): main.py registers the live
# detector at startup; the tool reads whatever's registered.

_ACTIVE_DETECTOR: "SoundDetector | None" = None


def register_detector(detector: "SoundDetector | None") -> None:
    """Register the live SoundDetector so the what_did_you_hear tool can read
    its recent-sounds buffer. Called once in main.py at construction."""
    global _ACTIVE_DETECTOR
    _ACTIVE_DETECTOR = detector


def _ago(seconds: float) -> str:
    """Human, voice-friendly recency: 'just now' / '~40s ago' / '~3m ago'."""
    if seconds < 10:
        return "just now"
    if seconds < 90:
        return f"~{int(round(seconds))}s ago"
    return f"~{int(round(seconds / 60.0))}m ago"


def _summarize_sounds(active, recent, fires, now, max_age_seconds, max_items=6):
    """Pure formatter for recent sounds — no threads, no I/O, trivially
    testable. `recent` = [(ts, label, score, rms)], `fires` = [(ts, name)],
    timestamps on the same monotonic clock as `now`.

    Three honest states: not-listening (acoustic off), listening-but-quiet
    (nothing recorded in the window), and a summary of fired alerts + the
    general soundscape."""
    if not active:
        return ("Acoustic awareness is off, sir — I'm not actively listening "
                "for sounds right now.")

    recent_in = [(ts, label, score) for (ts, label, score, _rms) in recent
                 if now - ts <= max_age_seconds]
    fires_in = [(ts, name) for (ts, name) in fires
                if now - ts <= max_age_seconds]

    if not recent_in and not fires_in:
        return "Nothing notable, sir — it's been quiet."

    lines: list[str] = []

    # Fired alerts first — the high-confidence monitored events, reported with
    # certainty and most-recent-first.
    if fires_in:
        fired_bits = [
            f"{name.replace('_', ' ')} ({_ago(now - ts)})"
            for ts, name in sorted(fires_in, key=lambda x: x[0], reverse=True)
        ]
        lines.append("Alerts that fired: " + ", ".join(fired_bits))

    # General soundscape — aggregate the dominant-label observations by label,
    # keeping peak score + most-recent time, then list most-recent-first.
    if recent_in:
        agg: dict[str, dict] = {}
        for ts, label, score in recent_in:
            e = agg.get(label)
            if e is None:
                agg[label] = {"peak": score, "recent": ts}
            else:
                e["peak"] = max(e["peak"], score)
                e["recent"] = max(e["recent"], ts)
        ranked = sorted(agg.items(), key=lambda kv: kv[1]["recent"], reverse=True)
        sound_bits = [
            f"{label} (peak {e['peak']:.2f}, {_ago(now - e['recent'])})"
            for label, e in ranked[:max_items]
        ]
        tail = (f" (+{len(ranked) - max_items} more)"
                if len(ranked) > max_items else "")
        lines.append("Sounds detected: " + ", ".join(sound_bits) + tail)

    mins = int(round(max_age_seconds / 60.0))
    header = f"In the last {mins} minute{'s' if mins != 1 else ''}, sir:"
    return header + "\n" + "\n".join(lines)


WHAT_DID_YOU_HEAR_TOOL = {
    "name": "what_did_you_hear",
    "description": (
        "Report the non-speech AMBIENT sounds Jarvis has recently heard "
        "through acoustic awareness (M58) — use for 'what did you just "
        "hear?', 'did you hear something?', 'what was that noise?', 'have you "
        "heard anything?'. Returns any monitored alerts that fired (doorbell, "
        "knock, glass breaking) plus the general soundscape detected over the "
        "last few minutes. If acoustic awareness is off, it says so. This is "
        "about sounds IN THE ROOM, NOT the user's own spoken words or a "
        "request to transcribe speech."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "minutes": {
                "type": "integer",
                "description": (
                    "How far back to look, in minutes (default 3, max 15). "
                    "Use a larger value for 'did you hear anything in the "
                    "last 10 minutes?'."
                ),
            },
        },
    },
}


def execute_what_did_you_hear(params: dict) -> str:
    """Run the tool. Always returns a string — never raises."""
    detector = _ACTIVE_DETECTOR
    if detector is None:
        return "Acoustic awareness isn't set up on this machine, sir."
    minutes = params.get("minutes")
    try:
        minutes = int(minutes) if minutes is not None else 3
    except (TypeError, ValueError):
        minutes = 3
    minutes = max(1, min(15, minutes))
    try:
        return detector.recent_sounds_summary(max_age_seconds=minutes * 60.0)
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"[acoustic] what_did_you_hear failed: {exc}", file=sys.stderr)
        return "I couldn't read the recent sounds just now, sir."
