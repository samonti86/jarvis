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
# walk away after saying "activate security" without immediately triggering
# the alert on themselves. Ten seconds is comfortable for the "say activate,
# walk to the door, step out" use case — bumped from M34's initial 5s after
# live testing showed 5s wasn't enough margin. A real intruder couldn't
# exploit this (they'd have to arm Jarvis from outside, which they can't —
# voice activation requires being at the mic).
_ARM_GRACE_SECONDS = 10.0

# How long a person must be absent from frame before a fresh detection
# counts as a "new" presence (and re-triggers the announcement). 30s avoids
# both the spammy "every frame is a new event" failure and the silent "user
# stepped out of frame for 2 seconds, came back, no re-alert" failure.
_PRESENCE_CLEAR_SECONDS = 30.0

# YOLO class 0 = "person" in the COCO dataset. Default v8n weights are
# trained on COCO so this index is stable.
_YOLO_PERSON_CLASS = 0

# Confidence threshold for "this is a person". YOLOv8n at 0.5 has very low
# false-positive rate on typical scenes. Lower would catch more (incl. pets
# misidentified as small persons); higher would miss partial views.
_YOLO_CONFIDENCE = 0.5

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
    ) -> None:
        self._announce = announce
        self._on_armed_changed = on_armed_changed
        # Fires (True/False) on LOCKED state enter/exit. Optional; if None,
        # the UI just doesn't show a LOCKED indicator (functionality intact).
        self._on_locked_changed = on_locked_changed
        self._camera_index = camera_index
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
        # at entry; external camera paths (Ring) supply pre-encoded bytes
        # directly. Either way, one in-memory copy lives until the
        # challenge resolves.
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

        # Match — resolve challenge as authenticated.
        # M35-LOCKED: this path now handles BOTH the regular CHALLENGE
        # clear and the LOCKED unlock. Both transitions look the same from
        # here: flip _challenge_active to False, clear _locked, enter
        # cooldown. The was_locked snapshot lets us update the UI
        # indicator + log a different message for the lockout-cleared case.
        with self._challenge_lock:
            if not self._challenge_active:
                # Already resolved (timer fired between transcript end and
                # our taking the lock — pre-LOCKED behavior). Idempotent;
                # don't announce twice.
                return True
            was_locked = self._locked
            self._challenge_active = False
            self._locked = False
            self._cooldown_until = time.monotonic() + _CHALLENGE_COOLDOWN_SECONDS
            self._challenge_evidence_frame = None  # don't need it anymore

        if was_locked:
            print("[security] LOCKED state cleared by passphrase — back to ARMED+cooldown",
                  file=sys.stderr)
            # UI: clear the 🔒 LOCKED indicator. ARMED indicator stays on
            # since the watcher is still armed.
            self._safe_call(self._on_locked_changed, False,
                            label="on_locked_changed(False)")

        self._safe_call(self._announce, "Welcome back, sir.", label="welcome-back announce")
        return True

    def activate(self) -> None:
        """Idempotent: arming an already-armed system is a no-op (no second
        announcement, no second thread spawned)."""
        if self._armed.is_set():
            return
        self._armed.set()
        self._stop.clear()
        self._armed_at = time.monotonic()
        self._person_present = False
        self._safe_call(self._on_armed_changed, True, label="on_armed_changed(True)")

        # Announce BEFORE spawning the watcher so the user hears the
        # confirmation immediately. The 5s grace window starts now; by the
        # time the watcher's first inference fires, the user has had time
        # to step away from the camera.
        self._safe_call(self._announce, "Security mode active, sir. I am standing watch.",
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
            self._challenge_evidence_frame = None
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

                frame = self._grab_frame()
                if frame is None:
                    # Camera busy / closed / black frame — log once per 10
                    # consecutive failures to avoid log spam, then sleep.
                    # For v1 we just sleep and retry.
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

                # Sleep till the next poll, but wake instantly on _stop so
                # disarming feels responsive.
                if self._stop.wait(_POLL_SECONDS):
                    return
        finally:
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
        """Public entry for an external camera (e.g. Ring) that detected
        motion. Feeds into the same challenge/deterrent state machine as
        local YOLO detection. Honors armed-state, cooldown, and the "one
        challenge at a time" rule — the watcher decides whether to act."""
        if not self._armed.is_set():
            return
        self._enter_challenge(jpeg_bytes or b"", source=source, now=time.monotonic())

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
                # Already in a challenge from a previous transition (e.g.
                # Logi flicker, or Ring + Logi co-firing). Let the existing
                # challenge run its course — don't restart timer or re-prompt.
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
            self._announce(
                "Please identify yourself within the next 15 seconds, sir.",
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

        # Discord notification on a daemon thread so a slow POST doesn't delay
        # the spoken deterrent or the LOCKED transition. Pass the in-memory
        # JPEG bytes directly rather than re-reading from disk.
        if self._discord_webhook_url and jpeg_bytes:
            self._send_discord_alert_async(jpeg_bytes, saved_path)

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

        print(
            f"[security] YOLO loaded in {time.monotonic() - t0:.1f}s",
            file=sys.stderr,
        )
        return True

    def _grab_frame(self):
        """Open camera → warmup → grab → release. Returns numpy frame or
        None on any failure. Mirrors cameras.py's pattern but with shorter
        warmup + smaller resolution since YOLO doesn't need full quality."""
        try:
            import cv2  # type: ignore
        except ImportError:
            return None

        cap = None
        try:
            cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)

            frame = None
            for _ in range(_WARMUP_FRAMES):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            return frame
        except Exception as exc:  # noqa: BLE001
            print(f"[security] frame grab failed: {exc}", file=sys.stderr)
            return None
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

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
                # Size filter: extract bbox height. xyxy is [x1, y1, x2, y2]
                # in pixels. Defensive — log + skip if shape unexpected.
                try:
                    coords = box.xyxy[0].tolist()
                    bbox_h = float(coords[3]) - float(coords[1])
                except (AttributeError, IndexError, ValueError, TypeError):
                    continue
                if bbox_h < min_bbox_h:
                    # Looks like a person to YOLO but is too small for a
                    # human-scale detection — almost certainly a pet or a
                    # picture-frame face. Log so we can audit false-rejects
                    # later if needed.
                    print(
                        f"[security] rejected: person@{conf:.2f} but bbox "
                        f"height {bbox_h:.0f}px < {min_bbox_h:.0f}px "
                        f"(< {_MIN_PERSON_HEIGHT_RATIO:.0%} of frame {frame_h:.0f}px) "
                        f"— likely a pet",
                        file=sys.stderr,
                    )
                    continue
                return True
        return False

    def _auto_disarm(self) -> None:
        """Internal: clear armed state + notify UI without an extra spoken
        message (caller already announced the reason for the failure)."""
        self._armed.clear()
        self._stop.set()
        if self._on_armed_changed is not None:
            try:
                self._on_armed_changed(False)
            except Exception:
                pass


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
