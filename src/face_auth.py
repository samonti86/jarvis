"""Face recognition auth path — parallel to the M35 voice passphrase.

Same goal as the voice passphrase: during a CHALLENGE state, prove the
person in frame is the authorized user. EITHER auth path can clear the
challenge — first match wins. Voice is still primary; face is a redundant
"I shouldn't have to talk to my own computer" UX.

Encoding lifecycle:
- Enrollment grabs N frames, runs face_recognition's HOG detector +
  encoder on each, averages the successful encodings into one (128,)
  vector, persists to %LOCALAPPDATA%/Jarvis/security/face_encoding.npy.
- Verification (during CHALLENGE) runs face_recognition on the current
  frame, computes Euclidean distance to the stored encoding, returns True
  if the closest face is within threshold.

Threshold notes:
- face_recognition.compare_faces defaults to 0.6 (a permissive setting
  optimized for "find me in a photo album"). We default to 0.5 because the
  security context wants fewer false-accepts even at the cost of needing
  better lighting / a more head-on pose.
- Configurable via JARVIS_FACE_MATCH_THRESHOLD in .env.

Defensive contract: import errors don't break this module. If
face_recognition isn't installed, AVAILABLE stays False and every public
function returns a sentinel value — never raises. Callers must check
AVAILABLE (or trust the False return).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


# face_recognition + its model package add ~100 MB and ~1s of import time.
# Lazy-load so Jarvis cold-start isn't penalized when face auth never runs.
# Module-level cache: _fr is the imported module (or None), AVAILABLE is
# the public sentinel callers check, _IMPORT_ERROR remembers a prior
# failure so we don't keep retrying a hopeless import.
_fr = None
AVAILABLE: bool = False
_IMPORT_ERROR: Exception | None = None

# A successful enrollment requires at least this many usable frames out of
# the N captured. 3-of-5 balances "tolerate one bad frame" against "don't
# enroll on a single lucky shot that might not generalize."
_MIN_USABLE_FRAMES_FOR_ENROLL = 3

# face_recognition.face_encodings returns 128-dim vectors. Tracked here as
# a sanity check on loaded encodings — a saved file with the wrong shape
# means version skew or corruption, both safe to ignore.
_ENCODING_DIM = 128


def _ensure_imported() -> bool:
    """First-call import. Subsequent calls are O(1) — module global caches.
    Returns True if face_recognition is usable, False on any import error."""
    global _fr, AVAILABLE, _IMPORT_ERROR
    if _fr is not None:
        return True
    if _IMPORT_ERROR is not None:
        return False
    try:
        import face_recognition  # noqa: PLC0415 — intentionally lazy
        _fr = face_recognition
        AVAILABLE = True
        return True
    except Exception as exc:  # noqa: BLE001 — any import failure disables the path
        _IMPORT_ERROR = exc
        print(
            f"[face_auth] face_recognition import failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False


def is_enrolled(path: Path) -> bool:
    """Cheap check: does an encoding file exist at `path`? Doesn't validate
    shape — use load_encoding() if you need a usable encoding."""
    return path.exists()


def load_encoding(path: Path) -> np.ndarray | None:
    """Returns the saved (128,) encoding from `path`, or None if absent /
    corrupt. Doesn't require face_recognition to be importable — pure
    numpy load."""
    if not path.exists():
        return None
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"[face_auth] couldn't load encoding {path}: {exc}",
              file=sys.stderr)
        return None
    if arr.shape != (_ENCODING_DIM,):
        print(
            f"[face_auth] saved encoding has unexpected shape {arr.shape}, "
            f"expected ({_ENCODING_DIM},); ignoring.",
            file=sys.stderr,
        )
        return None
    return arr


def delete_encoding(path: Path) -> bool:
    """Remove the enrolled encoding at `path`. Returns True if a file
    existed and was removed, False if there was nothing to remove. Never
    raises."""
    if not path.exists():
        return False
    try:
        path.unlink()
        print(f"[face_auth] deleted enrolled encoding at {path}", file=sys.stderr)
        return True
    except OSError as exc:
        print(f"[face_auth] couldn't delete encoding: {exc}", file=sys.stderr)
        return False


def enroll_from_frames(
    frames: list[np.ndarray],
    path: Path,
) -> tuple[bool, str]:
    """Process N BGR frames into one averaged encoding and persist it.

    `frames` are BGR ndarrays (cv2's default) of any size. Each is fed to
    face_recognition's HOG detector — fast, CPU-only, accurate enough for
    enrollment where the user is cooperating with the camera.

    Returns (success, message). The message is user-facing for both the
    tray popup and the voice flow ("Look at the camera, sir → here's how
    enrollment went, sir").

    Encoding strategy: per-frame encodings are MEAN-averaged into one
    (128,) vector. Averaging is the standard way to smooth pose/lighting
    variation across N captures — better match reliability than any single
    snapshot. We require at least _MIN_USABLE_FRAMES_FOR_ENROLL successful
    frames before saving; otherwise we reject and ask for a retry.
    """
    if not _ensure_imported():
        return False, "Face recognition isn't available, sir."
    if not frames:
        return False, "No frames were captured, sir."

    encodings: list[np.ndarray] = []
    rejected_multiface = 0
    rejected_noface = 0

    for frame in frames:
        # cv2 returns BGR; face_recognition (and dlib) want RGB. The
        # reverse-slice would be a zero-copy view, but dlib 20.0.1 segfaults
        # when handed a negative-stride array at 1080p (HOG iterates the
        # buffer assuming positive strides). ascontiguousarray forces the
        # copy — one memcpy per frame, well worth not crashing.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])

        # HOG model: CPU-only, fast (~50-150ms per frame), accurate for
        # head-on faces in good light. The CNN model would be more robust
        # to occlusion / angle but needs CUDA + much more time. Enrollment
        # is a controlled capture, so HOG is the right pick.
        face_locs = _fr.face_locations(rgb, model="hog")

        if not face_locs:
            rejected_noface += 1
            continue
        if len(face_locs) > 1:
            # Multiple faces in an enrollment frame is ambiguous — we can't
            # tell which one to enroll. Reject the frame; let the others
            # carry the average. (Reasonable failure mode: user has someone
            # standing behind them.)
            rejected_multiface += 1
            continue

        encs = _fr.face_encodings(rgb, known_face_locations=face_locs)
        if encs:
            encodings.append(encs[0])

    if not encodings:
        # No usable frames at all — surface the dominant failure reason
        # so the user knows what to fix.
        if rejected_multiface > rejected_noface:
            return False, "I saw more than one face in the frame, sir. Try again alone."
        return False, "I couldn't see your face clearly, sir. Try again with more light."

    if len(encodings) < _MIN_USABLE_FRAMES_FOR_ENROLL:
        return False, (
            f"I only got a clear look in {len(encodings)} of {len(frames)} frames, sir. "
            "Try again with better lighting or a steadier pose."
        )

    # Mean across the N (128,) vectors → one (128,) canonical encoding.
    # axis=0 averages frame-wise; the result is the "average appearance"
    # of the user across the capture window.
    mean_encoding = np.mean(np.stack(encodings, axis=0), axis=0)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mean_encoding)
    except OSError as exc:
        print(f"[face_auth] couldn't save encoding: {exc}", file=sys.stderr)
        return False, "I couldn't save the enrollment, sir. Check the security folder permissions."

    print(
        f"[face_auth] enrolled: {len(encodings)}/{len(frames)} frames used, "
        f"saved to {path}",
        file=sys.stderr,
    )
    return True, f"I'll recognize you now, sir. ({len(encodings)} of {len(frames)} frames used.)"


def verify_frame(
    frame: np.ndarray,
    enrolled_encoding: np.ndarray,
    threshold: float = 0.5,
) -> bool:
    """True if any face in the BGR `frame` matches `enrolled_encoding` within
    Euclidean distance `threshold`.

    `threshold` semantics — LOWER is stricter:
      0.4  ~ very strict, may reject valid matches in poor light
      0.5  ~ recommended for security
      0.6  ~ face_recognition's permissive default (photo-album use)

    Multi-face handling: if multiple faces are detected (user + guest, say),
    we take the MIN distance across all of them. That means the user being
    in frame with someone else doesn't block auth — the guest just needs
    to not be a closer match to the enrolled encoding, which is essentially
    never the case for two different people.

    Never raises — import failure / missing frame / no faces all return False.
    """
    if not _ensure_imported():
        return False
    if frame is None or frame.size == 0:
        return False

    # Contiguous copy, not a view — see enroll_from_frames for why dlib
    # crashes on the negative-stride view at production frame sizes.
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    face_locs = _fr.face_locations(rgb, model="hog")
    if not face_locs:
        return False

    encs = _fr.face_encodings(rgb, known_face_locations=face_locs)
    if not encs:
        return False

    # face_distance returns one Euclidean distance per detected face vs the
    # known encoding. Smaller = more similar. Take the min — the closest
    # face in frame wins.
    distances = _fr.face_distance(encs, enrolled_encoding)
    min_distance = float(np.min(distances))
    matched = min_distance <= threshold

    print(
        f"[face_auth] verify: faces={len(encs)} min_distance={min_distance:.3f} "
        f"threshold={threshold:.2f} match={matched}",
        file=sys.stderr,
    )
    return matched


def warm() -> None:
    """Pay the ENTIRE first-call cost of the face stack up front — import,
    HOG detector, AND the 68-point shape predictor + ResNet encoder — so
    none of it faults in lazily on the latency-critical CHALLENGE path.

    Why not just a blank verify_frame: on a blank frame `face_locations`
    finds nothing, so `face_encodings` (the encoder JIT — the heavy part)
    never runs. At a real challenge a real face IS present, so that cold
    encoder burst lands on the "identify yourself" prompt (diagnosed
    2026-05-19 from jarvis.log: input overflow at the prompt even after
    the import was warmed). We close that hole by handing `face_encodings`
    an EXPLICIT synthetic box: with a known location dlib runs the pose
    predictor + ResNet on that region regardless of content — exercising
    the exact code path verify_frame hits on a real face, no real face
    required.

    Never raises (mirrors verify_frame's defensive contract); a warm-up
    failure just means the cost falls back to lazy, never a crash."""
    if not _ensure_imported():
        return
    try:
        # Content is irrelevant — we only need the models to execute once.
        # np.zeros is C-contiguous, sidestepping the dlib negative-stride
        # crash (see verify_frame / enroll_from_frames notes).
        rgb = np.zeros((80, 80, 3), dtype=np.uint8)
        _fr.face_locations(rgb, model="hog")  # HOG detector first-call
        # Explicit (top, right, bottom, left) box → forces the pose
        # predictor + ResNet encoder JIT without needing a detected face.
        _fr.face_encodings(rgb, known_face_locations=[(8, 72, 72, 8)])
    except Exception as exc:  # noqa: BLE001 — warm-up must never raise
        print(
            f"[face_auth] warm-up incomplete: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


# --- Enrollment voice flow -------------------------------------------------
# Both the tray "Enroll my face" callback and the voice intent share the
# same orchestration. Voice-first design: a tk popup with a countdown adds
# code without adding clarity — Jarvis announces the prompt, captures, and
# announces the result. Both triggers use this same function.

# Regex matching: enroll / remember / register / memorize / learn / save +
# "face" (optionally prefixed with my/your). "remember me" alone is too
# broad — anchoring on the word "face" keeps false-positives away from
# casual conversation about memory.
_ENROLL_INTENT_RE = re.compile(
    r"\b(enroll|remember|register|memorize|learn|save)\b\s+(my\s+|your\s+)?face\b",
    re.IGNORECASE,
)


def matches_enroll_intent(transcript: str) -> bool:
    """Does the transcript express 'enroll my face' intent? Used by the
    listen_loop to consume the turn locally instead of routing to Claude."""
    return bool(_ENROLL_INTENT_RE.search(transcript or ""))


def run_voice_enrollment(
    announce_fn: Callable[..., None],
    capture_fn: Callable[[int, float], list | None],
    encoding_path: Path,
    num_frames: int = 5,
    interval_seconds: float = 0.5,
) -> None:
    """Speak the prompt → wait for prompt to finish → capture → enroll →
    speak the result. Non-blocking — the caller's thread returns as soon
    as the prompt is queued; the real work runs on the Announcer thread
    via the on_done callback.

    `announce_fn`: same signature as main.py's _announce — (text, on_done=None).
        Plays speech on Jarvis's dedicated Announcer thread (per the WASAPI
        constraint: proactive speech must go through this path, not direct
        speak_streaming calls).
    `capture_fn`: called as (n, interval) → list of BGR ndarrays or None.
        Decoupled for testability; in production this is cameras.capture_frames.
    `encoding_path`: where to save the resulting encoding (full path, not a
        base dir — encoder writes parent directories as needed).
    """
    # Upfront availability check — if face_recognition isn't installed, fail
    # cleanly BEFORE playing the prompt + capturing frames. Without this the
    # user would hear "hold still" → see the camera LED → wait → only then
    # get "face recognition isn't available," which feels broken.
    if not _ensure_imported():
        announce_fn("Face recognition isn't installed, sir.")
        return

    def _after_prompt() -> None:
        # Brief settle delay — gives the user a beat to stop reacting to
        # the voice and look at the camera. Without this, the first frame
        # often catches them mid-blink or mid-head-turn.
        time.sleep(0.5)
        try:
            frames = capture_fn(num_frames, interval_seconds)
        except Exception as exc:  # noqa: BLE001 — defensive against camera errors
            print(f"[face_auth] enrollment capture raised: {exc}",
                  file=sys.stderr)
            announce_fn("I couldn't access the camera, sir.")
            return
        if not frames:
            announce_fn("I couldn't access the camera, sir.")
            return
        _, msg = enroll_from_frames(frames, encoding_path)
        announce_fn(msg)

    announce_fn(
        "Hold still while I take a look, sir.",
        on_done=_after_prompt,
    )
