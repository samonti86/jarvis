"""camera_snapshot — give Jarvis eyes.

Grabs a single still frame from a USB/UVC webcam and returns it as a Messages-
API image content block, so Claude *sees* it and can answer in voice ("the pet
is on the couch and you left the kitchen light on, sir"). Unlocks the whole
class of "what do you see / is X there / did Y happen" queries.

Why a USB webcam (not an IP/Ring camera) for v1: the user's Logitech C920e is
already plugged in (it doubles as Jarvis's microphone — audio and video are
separate device interfaces on a UVC cam, so they coexist fine). No cloud, no
credentials, no RTSP setup. Sub-second-ish capture plus DirectShow's open cost
(~3-4s total) — comparable to the diagnostics collector, fine for an on-demand
"what do you see?" query.

Design notes:
- **Open-on-demand, never persistent.** We could keep cv2.VideoCapture open
  for instant snapshots, but that lights the camera's in-use LED the whole
  time Jarvis runs and blocks other apps (video calls) — wrong for a privacy-
  sensitive feature. Open, grab, release.
- **CAP_DSHOW backend.** On Windows the Media Foundation backend (CAP_MSMF,
  the cv2 default) fails to open the C920e here; DirectShow opens it cleanly.
- **Warm-up frames.** USB webcams return a dark/stale first frame while auto-
  exposure settles — read a few and keep the last.
- **Lazy cv2 import.** opencv-python-headless is a ~60 MB dep; importing it
  inside the executor (not at module top) means a broken/missing install
  degrades to a readable error string instead of breaking llm.py's import
  chain. Same pattern as system_control.py's lazy pycaw import.
- **Defensive contract:** never raises. Camera busy / not found / black frame
  / cv2 missing all become a readable string Claude relays apologetically.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from src.attachments import image_block


# --- Anthropic tool definition ---------------------------------------------
CAMERA_SNAPSHOT_TOOL = {
    "name": "camera_snapshot",
    "description": (
        "Take a still photo from the webcam and look at it — this is how you "
        "SEE the physical world around the user. Call it whenever the user "
        "asks what you can see, what's in the room, whether something is there "
        "or in a certain state (\"is the package on the porch?\", \"did the pet "
        "get on the couch?\", \"is the light on?\", \"how do I look?\"), or wants "
        "you to identify or read something they're holding up to the camera. "
        "Each call captures a fresh frame. After looking, describe what you see "
        "concisely and conversationally — like glancing over and remarking on "
        "what you notice, not narrating a photograph. Don't announce that "
        "you're taking a picture; just look and tell them."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


# Capture frame size — 1280x720 is a native C920e mode and well within
# Anthropic's image sweet spot (no benefit going past ~1568px on the long
# edge; it gets downscaled server-side). ~140 KB JPEG at quality 80.
_CAPTURE_WIDTH = 1280
_CAPTURE_HEIGHT = 720
_JPEG_QUALITY = 80

# Read this many frames after opening, keeping the last — discards the dark/
# stale frames a USB webcam emits before auto-exposure settles.
_WARMUP_FRAMES = 6

# Below this mean pixel value (0-255) the frame is effectively black — almost
# always a closed privacy shutter or a covered lens, not a real scene.
_BLACK_FRAME_MEAN = 8.0


def _camera_index() -> int:
    """Which DirectShow device to open. Default 0 (the first/only camera);
    override with CAMERA_INDEX in .env if you have several and want another.
    Read at call time — same pattern as games.py's RAWG_API_KEY."""
    raw = os.getenv("CAMERA_INDEX", "").strip()
    try:
        return int(raw) if raw else 0
    except ValueError:
        return 0


def execute_camera_snapshot(params: dict) -> str | list[dict]:
    """Capture a webcam frame. Returns either a list of content blocks (image
    + caption) on success, or a readable error string. Never raises."""
    try:
        import cv2  # type: ignore  # lazy: see module docstring
    except ImportError:
        return (
            "Camera support isn't installed (opencv-python-headless missing). "
            "Run: pip install opencv-python-headless"
        )

    index = _camera_index()
    cap = None
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return (
                f"Couldn't open the webcam (device {index}) — it may be in use "
                "by another app, disconnected, or the privacy shutter is closed."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)

        frame = None
        for _ in range(_WARMUP_FRAMES):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        if frame is None:
            return (
                "The webcam opened but didn't return a usable frame — try again "
                "in a moment."
            )

        if float(frame.mean()) < _BLACK_FRAME_MEAN:
            return (
                "The webcam image came back black — the privacy shutter is "
                "probably closed or something's over the lens."
            )

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            return "Couldn't encode the webcam frame."
        jpg = bytes(buf)
    except Exception as exc:  # noqa: BLE001 — defensive; this tool must never raise
        print(f"[cameras] snapshot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return f"Camera error: {exc}"
    finally:
        if cap is not None:
            cap.release()

    h, w = frame.shape[:2]
    print(f"[cameras] snapshot ok: {w}x{h}, {len(jpg)} bytes", file=sys.stderr)
    when = datetime.now().strftime("%I:%M %p").lstrip("0")  # "2:15 PM" — cross-platform
    return [
        image_block(jpg, "image/jpeg"),
        {"type": "text", "text": f"Live webcam snapshot, captured just now ({when})."},
    ]


def capture_frames(n: int, interval_seconds: float = 0.5):
    """Open the configured webcam once, capture `n` frames spaced by
    `interval_seconds`, release. Returns a list of BGR ndarrays on success
    or None on any error (camera busy, cv2 missing, black frames, etc.).

    Used by face_auth enrollment — capturing N frames in one camera-open
    cycle (instead of N independent open/close cycles) saves ~3-4s of
    DirectShow init per frame, and avoids the "in-use LED flickers N times"
    UX. Same warmup + black-frame guard as execute_camera_snapshot.

    Never raises — all failures become a None return with a log line.
    """
    try:
        import cv2  # type: ignore
        import time as _time
    except ImportError:
        print("[cameras] capture_frames: cv2 not installed", file=sys.stderr)
        return None

    index = _camera_index()
    cap = None
    frames: list = []
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[cameras] capture_frames: couldn't open device {index}",
                  file=sys.stderr)
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)

        # Same warmup as execute_camera_snapshot — discard the dark/stale
        # first frames the C920e emits before auto-exposure settles.
        for _ in range(_WARMUP_FRAMES):
            cap.read()

        for i in range(n):
            ok, f = cap.read()
            if not ok or f is None:
                print(f"[cameras] capture_frames: frame {i} read failed",
                      file=sys.stderr)
                continue
            if float(f.mean()) < _BLACK_FRAME_MEAN:
                # Almost certainly the privacy shutter closed mid-capture.
                # Skip this frame; let the others proceed.
                print(f"[cameras] capture_frames: frame {i} too dark, skipped",
                      file=sys.stderr)
                continue
            frames.append(f)
            # Sleep BETWEEN frames, not after the last — saves interval_seconds
            # on the total wall-clock time.
            if i < n - 1:
                _time.sleep(interval_seconds)
    except Exception as exc:  # noqa: BLE001 — defensive
        print(f"[cameras] capture_frames failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None
    finally:
        if cap is not None:
            cap.release()

    if not frames:
        return None
    print(f"[cameras] capture_frames: got {len(frames)}/{n} frames",
          file=sys.stderr)
    return frames
