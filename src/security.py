"""Security mode — proactive vision watcher (M34 foundation + M35 challenge).

The "armed mode" half of the proactive-vision design. While armed, a
background thread polls the Logi webcam at ~0.5 Hz, runs YOLOv8n person
detection on each frame, and fires a proactive announcement when a person
enters the scene. M35 layers the challenge-response protocol on top:
detection enters CHALLENGE state with a 15s passphrase timer, voice
authentication clears it, timeout fires the deterrent + saves evidence.

State machine (M35):

    DISARMED ──"activate security"──→ ARMED
       ▲                                 │
       │                                 │ (person detected, not in cooldown)
       │                                 ▼
       │                              CHALLENGE (15s passphrase window open)
       │                                 │
       │                          ┌──────┴──────┐
       │                          │             │
       │              passphrase match     15s timeout
       │              + cooldown 60s      ↓
       │                          │   DETERRENT_FIRED
       │                          │   (announce + save evidence)
       │                          │   + cooldown 60s
       │                          │             │
       │                          └──────┬──────┘
       │                                 ▼
       └────"stand down"──────────── ARMED (cooldown blocks re-fire)

If `security_passphrase` is empty (env var unset), the CHALLENGE step is
skipped — detection just announces movement (M34 announce-only behavior).
Graceful fallback so the system works without auth configured.

Threading:
- The watcher thread is daemon=True, so it dies with the process.
- `activate()`, `deactivate()`, `handle_transcript()` are thread-safe.
- The `announce` callback fires on the watcher thread; main.py marshals
  speech onto a dedicated Announcer thread (see main.py for why).

Defensive contract — camera errors, YOLO failure, network errors all
become log lines, never raise through the watcher's thread boundary.
"""

from __future__ import annotations

import gc
import os
import re
import sys
import threading
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable


# --- Tunables --------------------------------------------------------------
# Polling cadence — every 2s gives 0.5 Hz inference, which at ~150-300ms per
# frame on a Ryzen 5 3400G is ~10% of one core. Faster = more responsive but
# higher idle CPU. A person can't traverse a small space in <2s, so this
# is plenty for the use case.
_POLL_SECONDS = 2.0

# Grace period after arming before the watcher starts looking. Lets the user
# walk away from the PC, to the door, and out without triggering on
# themselves. 30s comfortably covers an room-sized walk; tighter (10s)
# left the user feeling rushed in live testing. A real intruder couldn't
# exploit this — they'd have to arm Jarvis from outside, which requires
# being at the mic, which means being inside.
_ARM_GRACE_SECONDS = 30.0

# How long a person must be absent from frame before a fresh detection
# counts as a "new" presence (and re-triggers the announcement). 30s avoids
# both the spammy "every frame is a new event" failure and the silent "user
# stepped out of frame for 2 seconds, came back, no re-alert" failure.
_PRESENCE_CLEAR_SECONDS = 30.0

# YOLO class 0 = "person" in the COCO dataset. Default v8n weights are
# trained on COCO so this index is stable.
_YOLO_PERSON_CLASS = 0

# Confidence threshold for "this is a person". 0.7 is the empirical floor
# for filtering character-shaped objects (action-figure backpacks, mannequin
# torsos, large posters of people) which score 0.5-0.7 — caught the user's
# wall of character backpacks during live testing. Real people score 0.85+
# even in partial view. Tighter (0.85) would miss partial-view edge cases;
# looser (0.5) admits the false-positive class of "person-shaped static
# objects in the room."
_YOLO_CONFIDENCE = 0.7

# Minimum person bounding-box height as a fraction of frame height. Defense
# in depth against YOLO misclassifying a pet as a person at low confidence:
# even if the class label is wrong, the SIZE never matches a human-scale
# detection. Sanity numbers (~8ft indoor camera distance):
#   - standing person:     50-70% of frame height
#   - sitting/crouched:    30-50%
#   - pet at same distance: 10-15%
#   - close-up pet (3ft):  20-25% (still below threshold)
#   - person at door (far): 30-40% (just above threshold — acceptable margin)
# 0.30 picks the lowest-pet-can-grow-to value with comfortable headroom
# for the smallest realistic person detection. If the user has children or
# the camera is unusually wide, tune downward; if false positives persist,
# tune upward.
_MIN_PERSON_HEIGHT_RATIO = 0.30

# Capture resolution — smaller than cameras.py's 1280×720 because YOLO
# resizes to 640×640 internally anyway, so a 640×480 source saves the
# downsample step + halves the JPEG-decode work.
#
# 2026-08-02 — MEASURED: the C920e IGNORES this request. cap.set() returns True
# for both width and height, but the driver keeps delivering 1920×1080 (the log
# has said "frame 1080px" since May; evidence JPEGs are ~330 KB). Setting
# CAP_PROP_FOURCC=MJPG first — the usual DSHOW workaround — does not change it
# either. Both were probed directly against the hardware.
#
# Do NOT "fix" this by downscaling after the read. Also measured, same session:
#   YOLO @1080p  89.9 ms   |  YOLO @640×480  105.5 ms  (+ 9.7 ms for the resize)
# ultralytics letterboxes to 640 regardless, so a smaller source buys nothing
# and the resize is pure added cost. Only the JPEG encode gets cheaper (11.6 →
# 2.0 ms), which happens once per challenge and is irrelevant to sustained load.
#
# The armed-mode audio starvation this was suspected of causing is NOT a
# resolution problem — it is GIL burstiness (one 90 ms YOLO call straddles an
# 80 ms audio callback slot). The fix is the deep capture/playback buffers in
# audio.py + text_to_speech.py, not fewer pixels. Left as-is deliberately.
_CAPTURE_WIDTH = 640
_CAPTURE_HEIGHT = 480

# Warmup frames per grab. cameras.py uses 6 for cold-open quality; we use 2
# because the watcher grabs every 2s and we'd rather keep CPU low than
# perfect-expose each frame. YOLO is robust to slightly underexposed input.
_WARMUP_FRAMES = 2

# M35 — challenge-response timing.
# 15s is enough to: hear the prompt (~2s) + say "Hey Jarvis" + passphrase
# (~3s) + STT process (~1-2s) = ~7s typical, ~10s worst case. 15s leaves
# margin for someone moving toward a mic or hesitating. Shorter (10s)
# feels tight; longer (20s) gives an intruder more time to either run or
# disable the system.
_CHALLENGE_TIMEOUT_SECONDS = 15.0

# Per-word fuzzy-match threshold. Word-level comparison (instead of full-
# string SequenceMatcher) avoids two real failure modes found in smoke
# testing:
#   1. "open sesame please" vs "open sesame" scores ~0.80 at the
#      character level (the "strongest " prefix dominates) — a synonym
#      attack false-positive. Word-level checking requires "defender" to
#      fuzzy-match "avenger" alone (ratio 0.53), correctly rejecting it.
#   2. "stronger ranger" with a permissive 0.75 word threshold also passes
#      ("ranger" vs "avenger" = 0.77) — a rhyme attack. Bumping to 0.80
#      drops it while still admitting realistic Whisper wobble.
#
# 0.80 is the empirical sweet spot from the matcher tests:
#   "strong" → "strongest"  = 0.80  ✓ accepted (legit Whisper short-form)
#   "stronges" → "strongest" = 0.94 ✓ accepted (legit Whisper typo)
#   "avengers" → "avenger"  = 0.93  ✓ accepted (legit Whisper plural)
#   "ranger" → "avenger"    = 0.77  ✗ rejected (rhyme attack)
#   "defender" → "avenger"  = 0.53  ✗ rejected (synonym attack)
#   "stranger" → "strongest" = 0.59 ✗ rejected (rhyme attack)
_PASSPHRASE_WORD_MATCH_THRESHOLD = 0.80

# After a CHALLENGE resolves (authenticated OR deterrent), suppress new
# challenges for this many seconds. Prevents the case where the legit
# user authenticates, walks past the camera again 5 seconds later, and
# gets re-challenged — which would be infuriating. 60s is comfortable.
_CHALLENGE_COOLDOWN_SECONDS = 60.0

# Evidence snapshots from triggered challenges land here. Created lazily on
# first save. Lives under %LOCALAPPDATA% so it follows the existing memory
# directory convention (jarvis.log, sessions/, summaries.jsonl).
_EVIDENCE_SUBDIR = "security/events"

# psutil memory watchdog — the structural backstop (M44). Post-mortem: an
# ultralytics-in-a-loop leak in the armed watcher consumed ~30 GB while the
# user was away, exhausting a ~10 GB machine and forcing a hard reboot
# (Windows Resource-Exhaustion-Detector named pythonw.exe). A self-monitoring
# guard means ANY in-process leak — this thread or another — degrades to a
# logged auto-disarm instead of taking the whole machine down. This is the
# CLAUDE.md "never crash the listening loop / degrade gracefully" rule
# enforced at the *process* level, where it originally failed.
#
# M44.2 post-mortem (2026-05-17) — THE METRIC WAS WRONG. The first real
# armed soak under an actual leak: commit hit 37 GB (Event 2004), the machine
# locked up, and the guard NEVER TRIPPED. Root cause: it watched RSS (the
# Windows *working set* — resident physical pages). The leak grows COMMIT /
# private bytes; under memory pressure Windows aggressively *trims the working
# set*, so RSS stayed below the limit while private bytes ran to 37 GB. RSS is
# the one number Windows actively suppresses during exactly this failure — an
# RSS guard is precisely backwards for it (it even false-tripped on the
# harmless 1501 MB cold-start working-set spike in M44.1 while staying silent
# on the lethal 37 GB commit growth). Fix: watch (a) this process's private
# bytes — the number that actually grew — AND (b) system-wide available
# memory. (b) is the robust, leak-source-agnostic backstop: it trips no matter
# where the memory went or which metric it shows up in, and is what actually
# prevents the lockup. With (b) as the real net, the per-process threshold no
# longer has to be calibrated precisely (defusing the M44.1 calibration trap).
#
# JARVIS_MEMORY_WATCHDOG_MB  — process private-bytes ceiling (default 3500;
#   armed baseline ≈1.5–2 GB private: torch CPU + YOLO + dlib/face_recognition
#   + openWakeWord ORT + Tk; STT offloaded off-box per M36). <= 0 disables
#   THIS check only.
# JARVIS_MEMORY_MIN_AVAIL_MB — system-available-memory floor (default 512).
#   IMPORTANT: the PROCESS check above is the load-bearing guard — the M44.2
#   leak was *this process's own* private bytes (→37 GB), so the 3500 MB
#   ceiling alone would have tripped within minutes, far before danger. This
#   system floor is a SECONDARY net for leaks the process check can't see
#   (e.g. an MCP child). It is deliberately conservative because this machine
#   is memory-tight: measured ~1.2 GB system-available while *unarmed*
#   (2026-05-17), and arming loads ~1.5 GB of ML stack — a 1 GB floor would
#   M44.1-false-trip on every arm. 512 MB is a guessed-low placeholder, NOT a
#   calibrated value: applying the M44.1 lesson (loose first, calibrate from a
#   real soak measurement, never trust a guessed fail-safe constant). On a
#   roomier box raise it. <= 0 disables THIS check only.
# Set BOTH <= 0 to fully disable the watchdog.
def _watchdog_limit_mb() -> int:
    raw = os.getenv("JARVIS_MEMORY_WATCHDOG_MB", "").strip()
    try:
        return int(raw) if raw else 3500
    except ValueError:
        return 3500


def _watchdog_min_avail_mb() -> int:
    raw = os.getenv("JARVIS_MEMORY_MIN_AVAIL_MB", "").strip()
    try:
        return int(raw) if raw else 512
    except ValueError:
        return 512


_MEMORY_WATCHDOG_MB = _watchdog_limit_mb()
_MEMORY_MIN_AVAIL_MB = _watchdog_min_avail_mb()


# JARVIS_YOLO_THREADS — cap on torch's intra-op thread pool for the armed
# watcher's YOLOv8n inference. Default 1.
#
# Why this exists: torch defaults its intra-op pool to ALL physical cores.
# On this 4-core box a single 0.5 Hz inference then pins every core for
# ~200-400 ms; when that burst overlaps a TTS phrase it starves the
# Python-fed audio threads, producing the *armed-only* speech stutter
# (diagnosed 2026-05-18 from the input-overflow log signature — output
# underrun is the same starvation, just uninstrumented). YOLOv8n on one
# frame every 2 s has an enormous latency budget: 1 thread is ~0.5-1 s/
# inference, still far inside the 2 s poll, and leaves 3 cores free so the
# real-time audio thread never starves. Inference speed is irrelevant at
# 0.5 Hz; free cores are not. Raise toward core-count only if inference
# latency ever becomes the actual constraint (it is not at this cadence).
# <= 0 → leave torch's default (uncapped) — for debugging/parity only.
def _yolo_threads() -> int:
    raw = os.getenv("JARVIS_YOLO_THREADS", "").strip()
    try:
        return int(raw) if raw else 1
    except ValueError:
        return 1


_YOLO_THREADS = _yolo_threads()


# --- Public API ------------------------------------------------------------
# AnnounceFn signature: (text, on_done=None) → None. Caller-supplied;
# main.py wraps speak_streaming so we get UI state coordination + the
# waveform pulse during proactive speech. The optional on_done callback
# fires AFTER playback completes (added in M35 follow-on so the watcher
# can defer its 15s challenge timer until the prompt has been heard).
AnnounceFn = Callable[..., None]


class SecurityWatcher:
    """Proactive person-detection watcher.

    Usage (from main.py):

        watcher = SecurityWatcher(announce_fn, set_armed_indicator_fn)
        # Voice intent parser:
        if "activate security" in transcript:
            watcher.activate()
        if "stand down" in transcript:
            watcher.deactivate()
    """

    def __init__(
        self,
        announce: AnnounceFn,
        on_armed_changed: Callable[[bool], None] | None = None,
        on_locked_changed: Callable[[bool], None] | None = None,
        camera_index: int = 0,
        passphrase: str = "",
        evidence_dir: Path | None = None,
        discord_webhook_url: str = "",
        smtp_host: str = "",
        smtp_port: int = 587,
        smtp_username: str = "",
        smtp_password: str = "",
        smtp_to: str = "",
        face_encoding_path: Path | None = None,
        face_match_threshold: float = 0.5,
        speaking_event: "threading.Event | None" = None,
    ) -> None:
        self._announce = announce
        self._on_armed_changed = on_armed_changed
        # Fires (True/False) on LOCKED state enter/exit. Optional; if None,
        # the UI just doesn't show a LOCKED indicator (functionality intact).
        self._on_locked_changed = on_locked_changed
        self._camera_index = camera_index
        # M44.3: persistent capture handle for the armed-watcher lifetime.
        # Opened lazily on the first grab, reused every poll, released when
        # the watcher loop exits (disarm / stop / watchdog trip). See
        # _grab_frame / _watch_loop's finally.
        # M71: a SECOND caller now exists — grab_frame_for_snapshot, invoked
        # off the LLM tool thread when a Discord camera_snapshot lands while
        # armed (the watcher owns the camera, so it serves the frame rather
        # than letting a contending handle grab black). Two threads reading
        # one cv2.VideoCapture is undefined, so an RLock now serialises all
        # capture access (_grab_frame / _release_cap / the snapshot grab).
        # RLock (not Lock) because _grab_frame calls _release_cap on a read
        # failure while already holding it.
        self._cap = None
        self._cap_lock = threading.RLock()
        # M35: empty passphrase = skip CHALLENGE state entirely. Detection
        # just announces movement (M34 behavior). Lets the user opt out of
        # the challenge step without removing security mode entirely.
        self._passphrase = passphrase.strip()
        # %LOCALAPPDATA%/Jarvis/security/events/ — created lazily on first
        # deterrent fire. Caller (main.py) computes the absolute path; we
        # just store it and stat/mkdir as needed.
        self._evidence_dir = evidence_dir
        # Discord webhook URL for deterrent-time push notifications.
        # Empty string = no notification path (bluff + local evidence only).
        self._discord_webhook_url = discord_webhook_url
        # SMTP email alert path. Fires in parallel with Discord when both are
        # configured. Any of the required fields blank → email path disabled.
        # Defensive check at fire-time mirrors the Discord pattern: missing
        # creds → silent skip, never a noisy exception in the watcher thread.
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_username = smtp_username
        self._smtp_password = smtp_password
        self._smtp_to = smtp_to
        # M39: face-recognition auth path. Encoding is loaded lazily on
        # activate() so re-enrollment between arm cycles takes effect
        # without a Jarvis restart. None = face auth disabled (no path
        # configured or no enrollment yet).
        self._face_encoding_path = face_encoding_path
        self._face_match_threshold = face_match_threshold
        self._enrolled_face = None  # numpy (128,) array, set on activate()
        # Cooperative speech gate (2026-05-19). Set by the Announcer thread
        # WHILE a proactive announce is playing. The watcher defers its
        # heavy vision bursts (camera grab, YOLO + per-tick gc.collect,
        # face encode, the cold-start warm) whenever this is set, so those
        # GIL/CPU bursts never overlap the Python-fed TTS path and stutter
        # the announce. None → gate disabled (degrades to pre-fix behaviour;
        # never breaks the watcher). Generalises the M44 "cooperative yield
        # while armed" follow-on; subsumes the A/B/C point-fixes.
        self._speaking_event = speaking_event

        # Coordination primitives. _armed is the public state; _stop fires
        # when we need the watcher thread to wind down (overlaps with
        # _armed.is_set() == False but distinct: deactivate sets _stop so an
        # in-progress poll cycle short-circuits, even before the thread
        # re-checks _armed at the top of its loop).
        self._armed = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._armed_at = 0.0

        # Presence-window state. _person_present is True between the
        # transition (empty → person) and the timeout firing. Tracked
        # entirely on the watcher thread so no lock needed.
        self._person_present = False
        self._person_last_seen = 0.0

        # M35: challenge-response state. Touched from BOTH the watcher
        # thread (timeout check, entering challenge) AND the listen_loop
        # thread (try_authenticate after STT). All mutations guarded by
        # _challenge_lock. Reads of the bool are fine without the lock
        # (single-word load is atomic on CPython), so is_in_challenge()
        # doesn't grab it.
        self._challenge_lock = threading.Lock()
        self._challenge_active = False
        self._challenge_started_at = 0.0
        self._cooldown_until = 0.0
        # M35-LOCKED: after a deterrent fires, instead of returning to
        # the normal ARMED+cooldown state, we enter LOCKED — _challenge_active
        # stays True (so listen_loop keeps diverting transcripts to the
        # passphrase comparator) but _challenge_started_at = 0 (no timer →
        # no second deterrent). _locked = True flags this for the UI and
        # for distinguishing log messages. Only a correct passphrase clears
        # it (voice). Tray disarm also works (physical-access escape hatch).
        # Critically: "stand down" by voice does NOT clear LOCKED, because
        # the listen_loop diversion catches it before the intent parser —
        # an intruder shouldn't be able to bypass with a guessed disarm.
        self._locked = False
        # JPEG bytes of the triggering frame, captured at challenge entry
        # so the deterrent path can save them without a re-grab (which
        # would capture the moment AFTER the 15s timeout, missing the
        # actual triggering view). Local YOLO path encodes from numpy
        # at entry; external camera integrations supply pre-encoded
        # bytes directly. Either way, one in-memory copy lives until
        # the challenge resolves.
        self._challenge_evidence_bytes: bytes = b""

        # Lazy-loaded YOLO. Held across activate/deactivate cycles so the
        # second arm is instant (no model reload). Becomes a sentinel
        # "failed to load" value if init blew up — see _ensure_model.
        self._model = None
        self._model_load_failed = False

    # ---------------------------------------------------------------------
    # Public state queries / mutators — thread-safe.
    # ---------------------------------------------------------------------

    def is_armed(self) -> bool:
        return self._armed.is_set()

    def set_on_armed_changed(
        self, callback: "Callable[[bool], None] | None",
    ) -> None:
        """Replace the on_armed_changed callback set at construction time.
        Used by main.py to *chain* late-constructed subsystems onto the
        armed-state edge — e.g. M58 acoustic awareness, which is built after
        SecurityWatcher and wants to auto-arm/disarm with security. Safe
        because activate() / deactivate() are only ever called post-startup
        (voice intent, tray, remote console), never during construction."""
        self._on_armed_changed = callback

    # ---------------------------------------------------------------------
    # Private utility: defensive call. Used by all the UI-callback and
    # announce sites so a misbehaving caller never breaks the watcher.
    # ---------------------------------------------------------------------

    def _safe_call(self, fn, *args, label: str) -> None:
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            print(f"[security] {label} raised: {exc}", file=sys.stderr)

    # M35: public API for listen_loop's challenge-transcript diversion.
    # is_in_challenge() is a hot read on every transcript, so it skips the
    # lock (single-word atomic load on CPython). try_authenticate() takes
    # the lock because it mutates challenge state.

    def is_in_challenge(self) -> bool:
        """True if a passphrase challenge is currently open. listen_loop
        checks this AFTER STT but BEFORE the intent parser / process_question
        — when True, the transcript routes to try_authenticate() and Claude
        isn't called for this turn.

        Includes both regular CHALLENGE (15s timer ticking) AND the LOCKED
        state (deterrent fired, no timer, awaiting passphrase). The check
        site doesn't need to distinguish: in both cases, transcripts go
        to try_authenticate() with the same logic."""
        return self._challenge_active

    def is_locked(self) -> bool:
        """True if we're in the post-deterrent LOCKED state (deterrent
        already fired, awaiting passphrase). Distinct from is_in_challenge()
        which is True for BOTH regular challenge AND locked."""
        return self._locked

    def handle_transcript(self, transcript: str) -> bool:
        """Single entry point for listen_loop to hand a freshly-transcribed
        utterance to the security subsystem. Returns True if consumed
        locally (don't pass to Claude). Encapsulates the ordering:
        challenge-auth check first, then activate/deactivate intent."""
        if not transcript:
            return False
        if self._challenge_active:
            self.try_authenticate(transcript)
            return True  # challenge state owns ALL transcripts (even non-matches)
        if _ACTIVATE_RE.search(transcript):
            self.activate()
            return True
        if _DEACTIVATE_RE.search(transcript):
            self.deactivate()
            return True
        return False

    def try_authenticate(self, transcript: str) -> bool:
        """Check a transcript against the configured passphrase. Returns
        True if it matches (challenge resolved as authenticated), False
        otherwise (challenge stays open until timeout).

        Word-level fuzzy match: every word in the passphrase must have a
        fuzzy match (SequenceMatcher ratio ≥ 0.75) somewhere in the
        transcript's words. Tolerates Whisper transcription wobble
        ("strong" → "strongest") while rejecting synonym attacks
        ("defender" → "avenger" scores 0.53, below threshold)."""
        if not self._passphrase:
            # No passphrase configured — challenge step skipped entirely
            # (M34 announce-only behavior). Defensive: if try_authenticate
            # is called anyway, do nothing.
            return False

        if not transcript or not transcript.strip():
            return False

        # Normalize both sides: lowercase, strip terminal punctuation, split
        # into words. Whisper sometimes appends a trailing period; we don't
        # want it to leak into the last word's similarity.
        clean_transcript = transcript.strip().lower().rstrip(".!?,")
        transcript_words = clean_transcript.split()
        passphrase_words = self._passphrase.lower().split()

        if not transcript_words or not passphrase_words:
            return False

        # Each passphrase word needs SOME word in the transcript that
        # fuzzy-matches at the per-word threshold. Short-circuit on first
        # word that has no match.
        def _has_fuzzy_match(target: str, candidates: list[str]) -> bool:
            return any(
                SequenceMatcher(None, target, c).ratio() >= _PASSPHRASE_WORD_MATCH_THRESHOLD
                for c in candidates
            )

        all_matched = all(_has_fuzzy_match(w, transcript_words) for w in passphrase_words)

        print(
            f"[security] challenge transcript={transcript!r} "
            f"words={transcript_words} all_passphrase_words_matched={all_matched}",
            file=sys.stderr,
        )

        if not all_matched:
            # At least one passphrase word missing. Challenge stays open
            # until the timer expires. Don't announce the failure — that
            # gives an intruder feedback to brute-force. Silent rejection.
            return False

        return self._mark_authenticated("voice")

    def _mark_authenticated(self, method: str) -> bool:
        """Resolve an active CHALLENGE (or LOCKED) state. Shared by the
        voice-passphrase path (try_authenticate) and the face-recognition
        path (_try_face_auth). `method` is "voice" or "face" for logging.

        Idempotent: if the challenge is no longer active (e.g. timer fired
        between the auth event and our taking the lock), returns True
        without announcing or re-firing UI callbacks.
        """
        # M35-LOCKED: handles BOTH the regular CHALLENGE clear and the
        # LOCKED unlock. Both transitions look the same from here: flip
        # _challenge_active to False, clear _locked, enter cooldown. The
        # was_locked snapshot lets us update the UI indicator + log a
        # different message for the lockout-cleared case.
        with self._challenge_lock:
            if not self._challenge_active:
                # Already resolved — idempotent, don't announce twice.
                return True
            was_locked = self._locked
            self._challenge_active = False
            self._locked = False
            self._cooldown_until = time.monotonic() + _CHALLENGE_COOLDOWN_SECONDS
            self._challenge_evidence_bytes = b""  # no longer needed

        if was_locked:
            print(f"[security] LOCKED state cleared by {method} auth — "
                  f"back to ARMED+cooldown", file=sys.stderr)
            # UI: clear the 🔒 LOCKED indicator. ARMED indicator stays on
            # since the watcher is still armed.
            self._safe_call(self._on_locked_changed, False,
                            label="on_locked_changed(False)")
        else:
            print(f"[security] CHALLENGE cleared by {method} auth",
                  file=sys.stderr)

        self._safe_call(self._announce, "Welcome back, sir.",
                        label=f"welcome-back ({method})")
        return True

    def _try_face_auth(self, frame) -> bool:
        """During an active CHALLENGE, check the current frame for the
        enrolled face. Returns True if a match cleared the challenge,
        False otherwise. Never raises (face_auth.verify_frame is
        defensive-on-import-failure).

        Frame is a BGR ndarray straight from the watcher's grab — same
        format _detect_person consumes, so no extra conversion."""
        from src import face_auth  # noqa: PLC0415 — defer until needed
        if face_auth.verify_frame(
            frame, self._enrolled_face, threshold=self._face_match_threshold,
        ):
            return self._mark_authenticated("face")
        return False

    def _is_announcing(self) -> bool:
        """True while a proactive announce is on the wire. The cooperative
        speech gate: heavy vision work must not overlap this (it starves
        the Python-fed TTS path → audio stutter). None event → always
        False, i.e. gate disabled, never breaks the watcher."""
        return self._speaking_event is not None and self._speaking_event.is_set()

    def _wait_for_quiet(self, timeout: float) -> bool:
        """Block until no proactive announce is playing, or `timeout`
        elapses, or _stop fires. Returns True if it should abort (stop
        requested). Bounded so a dropped/never-cleared announce can't
        deadlock the watcher — the gate degrades to 'proceed anyway',
        never to a hang."""
        deadline = time.monotonic() + timeout
        while self._is_announcing() and time.monotonic() < deadline:
            if self._stop.wait(0.2):
                return True
        return False

    def _warm_face_auth(self) -> None:
        """Pay the ENTIRE face-recognition/dlib cold-start cost once at
        watcher start, in silence — not lazily on the CHALLENGE path.

        Two costs, both GIL-holding bursts that stutter a Python-fed TTS
        announce if they overlap it (diagnosed 2026-05-19 from jarvis.log):
          1. `import face_recognition` — builds dlib's detector + 68-pt
             shape predictor + ResNet and loads their model blobs.
          2. the first real `face_encodings` — the encoder JIT, which a
             blank-frame warm never triggered (no face → no encode), so
             it used to fault in on the first real challenge face.
        face_auth.warm() now pays BOTH (it forces the encoder via an
        explicit synthetic box). And we run it only once the Announcer is
        quiet, so the ~2 s burst doesn't overlap the "Standing watch"
        confirmation either — the relocation bug the cooperative gate and
        this wait exist to prevent. There's ample time: a person must walk
        into frame before any challenge, and the 30 s arm-grace covers it.

        No-op when face auth isn't enrolled (nothing to warm, and the
        ~100 MB import would be pure waste). Defensive: any failure logs
        and falls back to lazy behaviour — never blocks the watcher."""
        if self._enrolled_face is None:
            return
        # Don't let the warm burst overlap the arm-confirmation announce.
        # Settle briefly first so the Announcer has actually picked the
        # queued item up and set the flag (it races our YOLO-load head
        # start), then wait for it to finish.
        if self._stop.wait(0.3):
            return
        if self._wait_for_quiet(timeout=20.0):
            return  # stop requested while waiting — abort cleanly
        t0 = time.monotonic()
        try:
            from src import face_auth  # noqa: PLC0415

            face_auth.warm()
        except Exception as exc:  # noqa: BLE001 — warm-up must never block arming
            print(
                f"[security] face auth warm-up skipped: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return
        print(
            f"[security] face auth warmed in {time.monotonic() - t0:.1f}s",
            file=sys.stderr,
        )

    def activate(self) -> None:
        """Idempotent: arming an already-armed system is a no-op (no second
        announcement, no second thread spawned)."""
        if self._armed.is_set():
            return
        self._armed.set()
        self._stop.clear()
        self._armed_at = time.monotonic()
        self._person_present = False
        # M39: refresh the enrolled face encoding from disk at each arm.
        # Cheap (~1ms numpy load) and picks up re-enrollment without a
        # Jarvis restart. None on any failure (no path configured, no
        # enrollment yet, corrupt file) — face auth path is just skipped.
        self._enrolled_face = None
        if self._face_encoding_path is not None:
            from src import face_auth  # noqa: PLC0415 — defer the ~100 MB import
            self._enrolled_face = face_auth.load_encoding(self._face_encoding_path)
            if self._enrolled_face is not None:
                print("[security] face auth enabled — enrolled encoding loaded",
                      file=sys.stderr)
            else:
                print("[security] face auth disabled — no enrolled encoding",
                      file=sys.stderr)
        self._safe_call(self._on_armed_changed, True, label="on_armed_changed(True)")

        # Announce BEFORE spawning the watcher so the user hears the
        # confirmation immediately. The 5s grace window starts now; by the
        # time the watcher's first inference fires, the user has had time
        # to step away from the camera.
        # Terse by design (2026-05-19): the project persona spec is "short,
        # understated J.A.R.V.I.S." — and a shorter announce is also a
        # smaller collision surface for the cooperative speech gate.
        self._safe_call(self._announce, "Standing watch, sir.",
                        label="announce on activate")

        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._watch_loop, name="SecurityWatcher", daemon=True
            )
            self._thread.start()

    def deactivate(self) -> None:
        """Idempotent: disarming an already-disarmed system is a no-op.

        M35-LOCKED: this is the only path other than voice-passphrase that
        clears the LOCKED state. Reachable from the tray menu (physical
        access = authority), not from voice "stand down" while locked
        (the listen_loop diversion catches that)."""
        if not self._armed.is_set():
            return
        self._armed.clear()
        self._stop.set()
        # M35: clear any pending challenge so disarming mid-challenge
        # doesn't leave stale state for the next arm. No deterrent fires
        # on this path — the user explicitly disarmed, so the trigger is
        # resolved by user authority. Same for LOCKED clear.
        with self._challenge_lock:
            self._challenge_active = False
            self._challenge_evidence_bytes = b""
            was_locked = self._locked
            self._locked = False
        self._safe_call(self._on_armed_changed, False, label="on_armed_changed(False)")
        if was_locked:
            self._safe_call(self._on_locked_changed, False,
                            label="on_locked_changed(False)")
        self._safe_call(self._announce, "Standing down, sir.",
                        label="announce on deactivate")

    def shutdown(self) -> None:
        """Called on app quit. Just clears state and signals the thread to
        exit; doesn't speak (the app is winding down)."""
        self._armed.clear()
        self._stop.set()
        # Don't join — daemon thread dies with the process. Joining could
        # block shutdown if the watcher is mid-inference (~300ms).

    # ---------------------------------------------------------------------
    # Watcher thread internals.
    # ---------------------------------------------------------------------

    def _watch_loop(self) -> None:
        """Daemon: poll camera, detect, enter challenge on person, fire
        deterrent on timeout. Exits when _stop fires."""
        # First iteration: lazy-load the model. Failure auto-disarms with
        # an announcement so the user isn't left thinking they're protected.
        if not self._ensure_model():
            return

        print("[security] watcher loop started", file=sys.stderr)

        # Option C (2026-05-19): warm the face-recognition/dlib stack now,
        # on this thread, during the harmless "standing watch" window —
        # NOT lazily on the first CHALLENGE, where the heavy import stutters
        # the time-critical "identify yourself" prompt. No-op when face auth
        # isn't enrolled. See _warm_face_auth for the full diagnosis.
        self._warm_face_auth()

        # M44 watchdog: resolve a psutil handle on this process once. Lazy +
        # defensive — psutil is a declared dep, but if it's somehow missing
        # the watcher must still run (just without the memory guard), per the
        # module's "never let a support feature break the watcher" contract.
        proc = None
        try:
            import psutil  # noqa: PLC0415 — light dep, lazy per convention

            proc = psutil.Process()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[security] psutil unavailable — memory watchdog disabled: {exc}",
                file=sys.stderr,
            )

        try:
            while not self._stop.is_set() and self._armed.is_set():
                # Grace period: don't fire detections in the first N seconds
                # after arming so the user has time to walk away from the
                # camera. _ARM_GRACE_SECONDS is the tunable.
                if time.monotonic() - self._armed_at < _ARM_GRACE_SECONDS:
                    if self._stop.wait(0.5):
                        return
                    continue

                # M35: check the 15s challenge timeout BEFORE the next frame
                # grab, so a deterrent fires promptly even if the camera is
                # briefly busy. Holds the lock briefly to read state.
                self._check_challenge_timeout()

                # M44: memory watchdog. Process-wide RSS guard — catches a
                # leak in THIS thread or any other while we're armed (the
                # exact unattended scenario that took the machine down). On
                # trip it logs, announces, and auto-disarms, then we exit.
                if self._check_memory_watchdog(proc):
                    return

                # Cooperative speech gate (2026-05-19). Never run a heavy
                # vision burst — camera grab, YOLO + per-tick gc.collect,
                # face encode — while a proactive announce is playing: the
                # watcher thread and the Python-fed TTS path share the GIL
                # + CPU, so an overlapping burst underruns the audio buffer
                # and stutters the announce (the armed-only stutter). The
                # cheap timeout + memory checks above stay live (safety);
                # only the heavy block below is deferred. Safe to skip
                # detection for the ~1-3 s an announce plays: the 30 s
                # arm-grace covers it, the challenge timer only starts
                # AFTER its prompt, and nobody authenticates in the few
                # seconds Jarvis is talking. This is the architectural fix
                # that subsumes the A/B/C point-fixes and the M44
                # "cooperative yield" follow-on.
                if self._is_announcing():
                    if self._stop.wait(0.3):
                        return
                    continue

                frame = self._grab_frame()
                if frame is None:
                    # Camera busy / closed / black frame — log once per 10
                    # consecutive failures to avoid log spam, then sleep.
                    # For v1 we just sleep and retry.
                    if self._stop.wait(_POLL_SECONDS):
                        return
                    continue

                # M39: during a CHALLENGE (or LOCKED) state, also try face
                # auth on the frame. Match → resolve like voice passphrase.
                # Cheap when no challenge is active (the bool short-circuits
                # before face_recognition runs). Skips YOLO this cycle on
                # match — cooldown handles the next cycle.
                if self._challenge_active and self._enrolled_face is not None:
                    if self._try_face_auth(frame):
                        if self._stop.wait(_POLL_SECONDS):
                            return
                        continue

                person_detected = self._detect_person(frame)
                now = time.monotonic()

                if person_detected:
                    if not self._person_present:
                        # Transition empty → person.
                        self._person_present = True
                        self._handle_person_first_seen(frame, now)
                    self._person_last_seen = now
                else:
                    # No person this cycle. If we were tracking a presence
                    # and it's been gone long enough, reset so the next
                    # detection counts as new.
                    if self._person_present and (
                        now - self._person_last_seen > _PRESENCE_CLEAR_SECONDS
                    ):
                        print(
                            f"[security] presence window closed (no person "
                            f"for {_PRESENCE_CLEAR_SECONDS:.0f}s)",
                            file=sys.stderr,
                        )
                        self._person_present = False

                # M44: break ultralytics' per-call reference cycle over the
                # Results objects (each holds .orig_img). _detect_person's
                # `del results` drops the acyclic refs; only the cyclic GC
                # reclaims the rest. Negligible at the 2 s poll cadence —
                # gc of a small young generation is sub-millisecond, and this
                # is what actually stopped the ~30 GB unattended growth.
                gc.collect()

                # Sleep till the next poll, but wake instantly on _stop so
                # disarming feels responsive.
                if self._stop.wait(_POLL_SECONDS):
                    return
        finally:
            # M44.3: free the persistent capture the instant the watcher
            # stops (disarm / stop / watchdog trip / model-load fail).
            # Scopes the held camera strictly to armed lifetime.
            self._release_cap()
            print("[security] watcher loop exited", file=sys.stderr)

    def _handle_person_first_seen(self, frame, now: float) -> None:
        """Local YOLO detected a person (EMPTY→PERSON transition). Encode
        the frame to JPEG bytes and feed into the shared challenge entry."""
        try:
            import cv2  # type: ignore
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg_bytes = bytes(buf) if ok else b""
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[security] failed to encode evidence frame: {exc}", file=sys.stderr)
            jpeg_bytes = b""
        self._enter_challenge(jpeg_bytes, source="local", now=now)

    def trigger_external_motion(
        self, source: str, jpeg_bytes: bytes | None = None,
    ) -> None:
        """Public entry for an external camera integration that detected
        motion. Feeds into the same challenge/deterrent state machine as
        local YOLO detection. Honors armed-state, cooldown, and the "one
        challenge at a time" rule — the watcher decides whether to act.

        Callers can pass jpeg_bytes=None and follow up with
        update_challenge_evidence() once a late-arriving snapshot
        finishes downloading — useful when the camera's snapshot API
        is slow but we want the challenge prompt to fire immediately."""
        if not self._armed.is_set():
            return
        self._enter_challenge(jpeg_bytes or b"", source=source, now=time.monotonic())

    def verify_person(self, jpeg_bytes: bytes) -> bool | None:
        """Decode JPEG bytes and run the same person-detection filter as
        the local Logi watcher (class==person + conf>=threshold + bbox
        height >=30% of frame). Returns True if a person is confirmed,
        False if the image is clear of people (pet / pet / nothing), or
        None if we couldn't make a determination (cv2 missing, decode
        failure, model unavailable, empty bytes).

        Callers should treat None as "unknown" and decide whether to
        fail open (fire alert) or fail closed (skip) based on context.
        Lazy-loads YOLO if not already in memory."""
        if not jpeg_bytes:
            return None
        try:
            import cv2  # type: ignore
            import numpy as np
        except ImportError:
            return None
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
        except Exception as exc:  # noqa: BLE001
            print(f"[security] verify_person decode failed: {exc}",
                  file=sys.stderr)
            return None
        if not self._ensure_model():
            return None
        try:
            return self._detect_person(frame)
        finally:
            # M44: _detect_person does `del results`, but the ultralytics
            # Results.orig_img reference CYCLE is only reclaimed by an explicit
            # gc.collect() — which _watch_loop runs per tick but this
            # external-camera path does not. Collect here too so EVERY YOLO
            # inference path uniformly breaks the cycle (defense-in-depth; this
            # path is rare, so the extra collect is free).
            gc.collect()

    def update_challenge_evidence(self, jpeg_bytes: bytes) -> None:
        """Attach late-arriving evidence bytes to an active challenge.
        For external camera integrations whose snapshot fetch completes
        AFTER the challenge prompt has already started playing. Silent
        no-op if no challenge is active, evidence is already set, or
        bytes are empty — first writer wins."""
        if not jpeg_bytes:
            return
        with self._challenge_lock:
            if self._challenge_active and not self._challenge_evidence_bytes:
                self._challenge_evidence_bytes = jpeg_bytes
                print(f"[security] late evidence attached: {len(jpeg_bytes)} bytes",
                      file=sys.stderr)

    def _enter_challenge(self, jpeg_bytes: bytes, source: str, now: float) -> None:
        """Shared challenge-entry path used by local YOLO and external
        cameras. Three exit paths:
            1. In cooldown — silently skip.
            2. No passphrase configured — announce-only (M34 fallback).
            3. Passphrase configured — enter CHALLENGE state with evidence
               cached as JPEG bytes; the 15s timer is deferred until the
               prompt finishes playing (see _start_challenge_timer below).
        """
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            print(
                f"[security] motion ({source}) but in cooldown "
                f"({remaining:.1f}s left) — skipping alert",
                file=sys.stderr,
            )
            return

        if not self._passphrase:
            print(f"[security] motion ({source}) — firing announcement (M34 mode)",
                  file=sys.stderr)
            self._safe_call(self._announce,
                            "Sir — I'm detecting movement in the monitored space.",
                            label="movement announce")
            return

        with self._challenge_lock:
            if self._challenge_active:
                # Already in a challenge — a second source firing on the
                # same person (e.g. an external camera + local YOLO both
                # seeing the same walk-through). Log so the suppression
                # is visible; otherwise the user only sees one source
                # name and wonders if the other one fired at all.
                print(f"[security] motion ({source}) suppressed — challenge "
                      f"already active from prior source", file=sys.stderr)
                return
            self._challenge_active = True
            self._challenge_started_at = 0.0  # SENTINEL: armed by on_done
            self._challenge_evidence_bytes = jpeg_bytes

        print(f"[security] motion ({source}) — entering CHALLENGE state "
              f"(15s timer starts after prompt)", file=sys.stderr)

        def _start_challenge_timer() -> None:
            with self._challenge_lock:
                if not self._challenge_active:
                    return  # auth resolved during playback — no-op
                self._challenge_started_at = time.monotonic()
            print("[security] challenge prompt finished — 15s timer armed",
                  file=sys.stderr)

        try:
            # Terse: drop the "within 15 seconds" recitation — the deterrent
            # timer still runs; reciting the number adds nothing the user or
            # an intruder needs, and the shorter prompt is a smaller speech-
            # gate collision surface. Persona spec: short, understated.
            self._announce(
                "Identify yourself, sir.",
                on_done=_start_challenge_timer,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[security] challenge announce raised: {exc}", file=sys.stderr)
            # Announce path is broken — arm the timer NOW so the user
            # can't be stranded in an un-timed challenge.
            with self._challenge_lock:
                if self._challenge_active:
                    self._challenge_started_at = time.monotonic()

    def _check_challenge_timeout(self) -> None:
        """Called once per watcher poll. If the 15s window has elapsed,
        fire the deterrent path: save the evidence snapshot, speak the
        bluff, enter cooldown. Idempotent — if no challenge is active or
        the timer hasn't expired, no-op.

        M35 follow-on: a `_challenge_started_at` of 0 means the prompt is
        still playing — timer hasn't been armed yet. Skip the check; the
        Announcer's on_done callback will arm it shortly."""
        with self._challenge_lock:
            if not self._challenge_active:
                return
            if self._challenge_started_at <= 0:
                # Prompt still playing (or announce path failed and the
                # defensive fallback hasn't fired yet). Wait.
                return
            elapsed = time.monotonic() - self._challenge_started_at
            if elapsed < _CHALLENGE_TIMEOUT_SECONDS:
                return
            # Timer expired. Flip state under the lock first to avoid a
            # race with try_authenticate (which might be running on the
            # listen_loop thread RIGHT NOW). Then release the lock before
            # the slow operations (file write + TTS).
            #
            # M35-LOCKED: instead of clearing _challenge_active + arming
            # cooldown, we ENTER the LOCKED state. _challenge_active stays
            # True (listen_loop keeps diverting transcripts to the passphrase
            # comparator), _challenge_started_at = 0 disables the timer
            # (no second deterrent fires), _locked = True flags it for the
            # UI. Only a correct passphrase or a tray disarm clears LOCKED.
            self._challenge_started_at = 0.0  # disable timer — no re-fire
            self._locked = True
            jpeg_bytes = self._challenge_evidence_bytes
            # Keep _challenge_evidence_bytes set — useful if we ever want
            # to re-save or re-send to Discord on user request.

        print(
            f"[security] CHALLENGE timeout ({_CHALLENGE_TIMEOUT_SECONDS:.0f}s) — "
            f"firing deterrent + entering LOCKED state",
            file=sys.stderr,
        )
        saved_path = self._save_evidence_bytes(jpeg_bytes)
        if saved_path is not None:
            print(f"[security] evidence saved: {saved_path}", file=sys.stderr)
        elif jpeg_bytes:
            print("[security] evidence save returned None despite non-empty "
                  "JPEG bytes — check evidence_dir permissions", file=sys.stderr)
        else:
            print("[security] no JPEG bytes to save (snapshot was empty)",
                  file=sys.stderr)

        # Discord notification on a daemon thread so a slow POST doesn't
        # delay the spoken deterrent or the LOCKED transition. ALWAYS
        # fire if the webhook is configured — send_discord_alert handles
        # empty image_bytes gracefully (sends a text-only message).
        # Skipping the push because the snapshot was empty would mean
        # losing the most valuable part of the alert system: the phone
        # ping. Better to ship a text-only alert than to skip.
        if self._discord_webhook_url:
            self._send_discord_alert_async(jpeg_bytes, saved_path)
        # SMTP email — same parallel fire-and-forget pattern. Both channels
        # are independent: Discord rate-limit doesn't suppress email and vice
        # versa. send_email_alert short-circuits if any required SMTP field
        # is blank, so we just check the username as the cheap gate.
        if self._smtp_username and self._smtp_to:
            self._send_email_alert_async(jpeg_bytes, saved_path)

        # 2026-07-02 QA: a successful try_authenticate can land in the gap
        # between the LOCKED transition above (the lock is released for the
        # slow evidence/push work) and this point — it clears _locked and
        # fires on_locked_changed(False). Re-check under the lock so we don't
        # pin a stale 🔒 indicator and speak a false "authorities notified"
        # right after "Welcome back, sir". (The evidence save + Discord/email
        # pushes above still fired — correct: someone WAS unauthenticated for
        # the whole challenge window; only the UI flip + deterrent are moot.)
        with self._challenge_lock:
            if not self._locked:
                print("[security] authenticated during deterrent prep — "
                      "suppressing LOCKED indicator + deterrent announce",
                      file=sys.stderr)
                return
        # Flip the LOCKED indicator BEFORE the announce so the UI updates
        # immediately, not after the ~5s deterrent playback finishes.
        self._safe_call(self._on_locked_changed, True, label="on_locked_changed(True)")
        self._safe_call(
            self._announce,
            "Identity not confirmed. Authorities have been notified. "
            "Images of the intruder have been transmitted to law enforcement.",
            label="deterrent announce",
        )

    def _send_discord_alert_async(
        self, jpeg_bytes: bytes, evidence_path: Path | None
    ) -> None:
        """Fire-and-forget Discord push on a daemon thread. Errors all
        swallowed inside send_discord_alert."""
        from src.notifications import send_discord_alert

        filename = evidence_path.name if evidence_path else "evidence.jpg"

        def _worker():
            send_discord_alert(
                self._discord_webhook_url, jpeg_bytes,
                image_filename=filename, when=datetime.now(),
            )

        threading.Thread(
            target=_worker, name="DiscordNotify", daemon=True
        ).start()

    def _send_email_alert_async(
        self, jpeg_bytes: bytes, evidence_path: Path | None
    ) -> None:
        """Fire-and-forget SMTP email push on a daemon thread. Errors all
        swallowed inside send_email_alert (same contract as Discord)."""
        from src.notifications import send_email_alert

        filename = evidence_path.name if evidence_path else "evidence.jpg"
        # Capture the trigger timestamp now, on the watcher thread, so the
        # email's body matches what the user just heard rather than the
        # moment SMTP eventually completes (which can lag a second or two).
        when = datetime.now()

        def _worker():
            send_email_alert(
                self._smtp_host,
                self._smtp_port,
                self._smtp_username,
                self._smtp_password,
                self._smtp_to,
                jpeg_bytes,
                image_filename=filename,
                when=when,
            )

        threading.Thread(
            target=_worker, name="EmailNotify", daemon=True
        ).start()

    def _save_evidence_bytes(self, jpeg_bytes: bytes) -> Path | None:
        """Write pre-encoded JPEG bytes to the evidence dir. Returns path
        on success, None on any failure. Defensive — never raises."""
        if not jpeg_bytes or self._evidence_dir is None:
            return None
        try:
            self._evidence_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = self._evidence_dir / f"{timestamp}.jpg"
            path.write_bytes(jpeg_bytes)
            return path
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[security] evidence save failed: {exc}", file=sys.stderr)
            return None

    def _ensure_model(self) -> bool:
        """Load YOLO once; cache across arm/disarm cycles. Returns True on
        success, False on permanent failure (and auto-disarms with an
        announcement so the user isn't silently unprotected)."""
        if self._model is not None:
            return True
        if self._model_load_failed:
            # Already tried this session and it didn't work — don't keep
            # retrying. User would need to fix the install + restart Jarvis.
            return False

        try:
            # Local import for the same reason cameras.py does it lazy:
            # heavy dep, broken install should degrade to a readable error
            # rather than crash the whole app at startup.
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            self._model_load_failed = True
            print(f"[security] ultralytics import failed: {exc}", file=sys.stderr)
            try:
                self._announce(
                    "Security model unavailable, sir. The vision library isn't installed."
                )
            finally:
                self._auto_disarm()
            return False

        t0 = time.monotonic()
        try:
            # First call downloads yolov8n.pt to ~/.cache/torch/hub (or
            # similar) — ~6 MB. Subsequent loads are instant from disk.
            # `verbose=False` is set per-inference, not per-construction.
            self._model = YOLO("yolov8n.pt")
        except Exception as exc:  # noqa: BLE001 — defensive
            self._model_load_failed = True
            print(
                f"[security] YOLO load failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            try:
                self._announce(
                    "Security model failed to load, sir. Check the logs."
                )
            finally:
                self._auto_disarm()
            return False

        # Cap torch's intra-op pool so a 0.5 Hz inference can't pin all
        # cores and starve the real-time audio threads (the armed-only TTS
        # stutter). Lazy import for the same reason ultralytics is: torch is
        # already in-process by here (YOLO pulled it), so this is cheap.
        # set_num_threads is process-global. NOTE: since M58, PANNs (acoustic
        # awareness) is ALSO a torch consumer — it caps itself the same way
        # (JARVIS_ACOUSTIC_THREADS in sound_detector.py), so when both are
        # armed they set the same value and the cap holds either way. (STT is
        # ctranslate2/GPU-offloaded, not torch.) <= 0 leaves torch's default.
        threads_note = "uncapped"
        if _YOLO_THREADS >= 1:
            try:
                import torch  # type: ignore  # noqa: PLC0415

                torch.set_num_threads(_YOLO_THREADS)
                threads_note = f"{_YOLO_THREADS} torch thread(s)"
            except Exception as exc:  # noqa: BLE001 — never block arming on this
                print(
                    f"[security] could not cap torch threads: {exc}",
                    file=sys.stderr,
                )

        print(
            f"[security] YOLO loaded in {time.monotonic() - t0:.1f}s "
            f"({threads_note})",
            file=sys.stderr,
        )
        return True

    def _release_cap(self) -> None:
        """Release + drop the persistent capture (M44.3). Idempotent.
        Called on a read failure (self-heal: next poll re-opens) and from
        _watch_loop's finally, so the camera is freed the instant the
        watcher stops — keeping the persistent capture strictly scoped to
        the armed-watcher lifetime (camera fully free whenever NOT armed).
        M71: under _cap_lock (the snapshot grab can race the watcher)."""
        with self._cap_lock:
            cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _grab_frame(self):
        """M44.3: persistent-capture grab. Opens self._cap ONCE (DSHOW —
        the backend documented to reliably open the C920e; MSMF's stale-
        comment claim aside, it still leaked ~3 MB/min per the repro), then
        reuses it every poll. The old open/release-every-tick was the M44.3
        native leak: ~160 MB per cycle / ~37 GB per soak, proven by
        scripts/leak_repro.py (persistent+DSHOW measured dead-flat over 400
        reads; per-tick DSHOW hit a 2.2 GB cap in 10). Released in
        _watch_loop's finally when the watcher exits, so the camera is fully
        free whenever security is NOT armed — camera_snapshot / face
        enrollment are unaffected (only contend if invoked WHILE armed,
        where existing code already degrades gracefully). Returns a numpy
        frame or None; any read failure drops the capture so the next poll
        transparently re-opens (self-healing)."""
        try:
            import cv2  # type: ignore
        except ImportError:
            return None

        # M71 — serialise capture access; grab_frame_for_snapshot can call in
        # from the LLM tool thread while the watcher loop is mid-grab. RLock,
        # so the _release_cap calls below (which re-acquire) don't deadlock.
        self._cap_lock.acquire()
        try:
            if self._cap is None:
                cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    return None
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAPTURE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)
                self._cap = cap
                # USB cams emit dark/stale frames until auto-exposure
                # settles — discard a few, but ONCE per open, not every
                # poll (per-poll warmup became wasted work + an extra
                # discarded frame the moment the capture went persistent).
                reads = _WARMUP_FRAMES
            else:
                reads = 1  # already warm — a single fresh read

            frame = None
            for _ in range(reads):
                ok, f = self._cap.read()
                if ok and f is not None:
                    frame = f
            if frame is None:
                # Read failed on an open capture — camera glitched /
                # unplugged. Drop it so the next poll re-opens cleanly.
                self._release_cap()
            return frame
        except Exception as exc:  # noqa: BLE001
            print(f"[security] frame grab failed: {exc}", file=sys.stderr)
            self._release_cap()
            return None
        finally:
            self._cap_lock.release()

    def grab_frame_for_snapshot(self):
        """M71 — serve a fresh frame from the armed watcher's persistent
        capture so a remote `camera_snapshot` (Discord, while away) doesn't
        contend for the webcam. Windows hands a UVC cam to ONE reader, so a
        second handle opened by cameras.py while armed gets a black/garbage
        frame (the 'camera covered' symptom). Returns a BGR ndarray when armed
        and holding the camera; None when not armed (the camera is free, so the
        caller opens its own handle — no contention). Never raises."""
        if not self.is_armed():
            return None
        try:
            with self._cap_lock:
                # 2026-07-02 QA: serve ONLY from an already-open persistent
                # capture — never open one here. A disarm can land between the
                # is_armed() check above and this point; if this path opened a
                # fresh capture after the watcher's finally released its own,
                # nothing would ever release ours (camera held for the whole
                # disarmed period — the M44 leak class). _cap is None ⇒ the
                # watcher isn't holding the camera ⇒ the caller can safely
                # open its own handle, so returning None is correct.
                if self._cap is None:
                    return None
                return self._grab_frame()
        except Exception as exc:  # noqa: BLE001 — a snapshot must never break
            print(f"[security] snapshot frame grab failed: {exc}", file=sys.stderr)
            return None

    def _detect_person(self, frame) -> bool:
        """Run YOLO inference, return True if any high-confidence person
        box is in the frame AND passes the size sanity check. Inference is
        ~150-300ms on CPU.

        Two-layer filter:
          1. Class == person (0) AND confidence ≥ 0.5 — catches the
             well-classified cases. Pets (class 15) and dogs (class 16) on
             COCO are correctly classified the vast majority of the time
             and rejected here.
          2. Bbox height ≥ 30% of frame height — defense against YOLO
             misclassifying a partially-visible pet as a "person" at low
             confidence. The SIZE never matches a human-scale detection
             regardless of label. Reasoning + thresholds documented at
             _MIN_PERSON_HEIGHT_RATIO above.
        """
        try:
            # verbose=False suppresses YOLO's per-call console banner that
            # would otherwise spam jarvis.log every 2 seconds.
            results = self._model(frame, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[security] inference failed: {exc}", file=sys.stderr)
            return False

        # frame.shape on a cv2 BGR frame is (H, W, 3).
        frame_h = float(frame.shape[0]) if hasattr(frame, "shape") else 0.0
        min_bbox_h = frame_h * _MIN_PERSON_HEIGHT_RATIO

        # try/finally so `del results` runs on EVERY exit (the early
        # short-circuit `return True` included). Each Results object retains
        # .orig_img — the full frame — so dropping our reference here, paired
        # with the loop-level gc.collect() in _watch_loop, is the fix for the
        # M44 leak: ultralytics' predictor holds a reference cycle over these,
        # and at 0.5 Hz unattended it grew ~tens of MB/tick until the machine
        # OOM'd. del alone reclaims the acyclic part; gc.collect() breaks the
        # cycle.
        try:
            for r in results:
                # r.boxes is the detected objects. Each box has .cls (class
                # index), .conf (confidence), and .xyxy ([x1,y1,x2,y2]).
                # Iterate looking for any qualifying person; short-circuit on
                # the first match.
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    try:
                        cls = int(box.cls)
                        conf = float(box.conf)
                    except (TypeError, ValueError, AttributeError):
                        continue
                    if cls != _YOLO_PERSON_CLASS or conf < _YOLO_CONFIDENCE:
                        continue
                    # Size filter: extract bbox height. xyxy is
                    # [x1, y1, x2, y2] in pixels. Defensive — log + skip if
                    # shape unexpected.
                    try:
                        coords = box.xyxy[0].tolist()
                        bbox_h = float(coords[3]) - float(coords[1])
                    except (AttributeError, IndexError, ValueError, TypeError):
                        continue
                    if bbox_h < min_bbox_h:
                        # Looks like a person to YOLO but is too small for a
                        # human-scale detection — almost certainly a pet or a
                        # picture-frame face. Log so we can audit
                        # false-rejects later if needed.
                        print(
                            f"[security] rejected: person@{conf:.2f} but bbox "
                            f"height {bbox_h:.0f}px < {min_bbox_h:.0f}px "
                            f"(< {_MIN_PERSON_HEIGHT_RATIO:.0%} of frame "
                            f"{frame_h:.0f}px) — likely a pet",
                            file=sys.stderr,
                        )
                        continue
                    return True
            return False
        finally:
            del results

    def _check_memory_watchdog(self, proc) -> bool:
        """Process + system memory guard (M44 / M44.2). Returns True — and
        fires a log line, a spoken notice, and an auto-disarm — if EITHER
        (a) this process's private/commit bytes cross
        JARVIS_MEMORY_WATCHDOG_MB, or (b) system-wide available memory falls
        below JARVIS_MEMORY_MIN_AVAIL_MB. Returns False otherwise.

        M44.2: was RSS-based; RSS is the working set, which Windows trims
        under pressure, so the real commit/private-bytes leak (37 GB) never
        tripped it. Now watches private bytes (the metric that actually grew)
        plus, as the leak-source-agnostic net, system available memory — that
        second check is what truly prevents the lockup, regardless of which
        metric or process the next leak shows up in.

        Defensive on every axis: a None proc (psutil missing), both limits
        disabled (<= 0), or a psutil read error all return False quietly. The
        guard must never be the thing that breaks the watcher — its whole job
        is to make a leak SAFER, not to add a new failure mode."""
        if proc is None:
            return False
        if _MEMORY_WATCHDOG_MB <= 0 and _MEMORY_MIN_AVAIL_MB <= 0:
            return False

        proc_mb = None
        avail_mb = None
        try:
            import psutil  # noqa: PLC0415 — light dep, lazy per convention

            if _MEMORY_WATCHDOG_MB > 0:
                mem = proc.memory_info()
                # .private == Windows Private Bytes (≈ per-process commit —
                # the M44 leak metric). Fall back to rss if absent (non-
                # Windows dev box) so this degrades instead of crashing.
                raw = getattr(mem, "private", None)
                if raw is None:
                    raw = mem.rss
                proc_mb = raw / (1024 * 1024)
            if _MEMORY_MIN_AVAIL_MB > 0:
                avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001 — psutil raises if proc gone
            print(f"[security] memory watchdog read failed: {exc}",
                  file=sys.stderr)
            return False

        proc_trip = proc_mb is not None and proc_mb >= _MEMORY_WATCHDOG_MB
        avail_trip = avail_mb is not None and avail_mb <= _MEMORY_MIN_AVAIL_MB
        if not proc_trip and not avail_trip:
            return False

        reason = (
            f"process private {proc_mb:.0f} MB >= {_MEMORY_WATCHDOG_MB} MB"
            if proc_trip else
            f"system available {avail_mb:.0f} MB <= "
            f"{_MEMORY_MIN_AVAIL_MB} MB floor"
        )
        print(
            f"[security] MEMORY WATCHDOG TRIPPED: {reason} — auto-disarming "
            f"security as a precaution (probable memory leak; see "
            f"docs/MILESTONES.md M44/M44.2)",
            file=sys.stderr,
        )
        # Announce before disarming so the user gets the spoken heads-up even
        # though the UI indicator flips immediately via _auto_disarm.
        self._safe_call(
            self._announce,
            "Sir, I'm consuming an unusual amount of memory. "
            "I'm standing down as a precaution.",
            label="memory watchdog announce",
        )
        self._auto_disarm()
        return True

    def _auto_disarm(self) -> None:
        """Internal: clear armed state + notify UI without an extra spoken
        message (caller already announced the reason for the failure).

        2026-07-02 QA: also clears any open CHALLENGE/LOCKED state, exactly
        like deactivate() — the watcher thread exits after this, so
        _check_challenge_timeout never runs again; leaving _challenge_active
        set would divert EVERY subsequent utterance into try_authenticate
        (including "stand down") with no path out, and a stale _locked would
        pin the 🔒 indicator forever."""
        self._armed.clear()
        self._stop.set()
        with self._challenge_lock:
            self._challenge_active = False
            self._challenge_evidence_bytes = b""
            was_locked = self._locked
            self._locked = False
        if self._on_armed_changed is not None:
            try:
                self._on_armed_changed(False)
            except Exception:
                pass
        if was_locked:
            self._safe_call(self._on_locked_changed, False,
                            label="on_locked_changed(False)")


# --- Voice-intent regexes --------------------------------------------------
# Loose patterns — Whisper sometimes hallucinates extra words ("Activate THE
# security please"). False-positive tolerance is fine here: arming/disarming
# is idempotent + low-impact, and the keywords don't occur in casual chat.

_ACTIVATE_RE = re.compile(
    r"\b(activate|engage|enable|arm|turn\s+on)\b.*\bsecurity\b",
    re.IGNORECASE,
)
_DEACTIVATE_RE = re.compile(
    r"\b(stand\s+down|disarm|deactivate|disable|security\s+off|turn\s+off\s+security)\b",
    re.IGNORECASE,
)
