"""End-to-end smoke test for the M39 face-auth path.

Runs the full enroll → verify flow without needing to launch Jarvis.
Saves the test encoding to a tmp path (not %LOCALAPPDATA%) so your real
enrollment, if any, isn't touched.

Run with: python smoketest_face.py
You'll need to be in front of the camera for ~10 seconds.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from src import cameras, face_auth


def _countdown(label: str, seconds: int) -> None:
    """Print a 1Hz countdown so the user knows when capture begins."""
    print(f"  {label} in", end="", flush=True)
    for s in range(seconds, 0, -1):
        print(f" {s}...", end="", flush=True)
        time.sleep(1)
    print(" capturing.")


def main() -> int:
    print("=== M39 face-auth smoke test ===")
    print()

    if not face_auth._ensure_imported():
        print("FAIL — face_recognition is not importable. Check the venv install.")
        return 1
    print("face_recognition import: OK")
    print()

    tmp_path = Path(tempfile.gettempdir()) / "jarvis_smoketest_face_encoding.npy"
    if tmp_path.exists():
        # Stale leftover from a prior run — remove it so we start fresh.
        tmp_path.unlink()

    # --- Phase 1: enrollment ----------------------------------------------
    print("Phase 1: ENROLLMENT")
    print("  Look straight at the camera. Capturing 5 frames over ~2.5s.")
    _countdown("Starting", 3)
    frames = cameras.capture_frames(n=5, interval_seconds=0.5)
    if not frames:
        print("FAIL — couldn't capture any frames. Camera busy? Privacy shutter?")
        return 1
    print(f"  Captured {len(frames)} frames.")

    success, msg = face_auth.enroll_from_frames(frames, tmp_path)
    print(f"  Enrollment: {msg}")
    if not success:
        print("FAIL — enrollment didn't produce a saved encoding.")
        return 1
    if not tmp_path.exists():
        print(f"FAIL — enrollment claimed success but {tmp_path} doesn't exist.")
        return 1
    print(f"  Encoding saved at {tmp_path}")
    print()

    encoding = face_auth.load_encoding(tmp_path)
    if encoding is None:
        print("FAIL — load_encoding returned None for a freshly-saved file.")
        return 1
    print(f"  Encoding shape: {encoding.shape} (expected (128,))")
    print()

    # --- Phase 2: verification --------------------------------------------
    print("Phase 2: VERIFICATION")
    print("  Stay in front of the camera. Grabbing 1 frame to verify.")
    _countdown("Verifying", 2)
    verify_frames = cameras.capture_frames(n=1, interval_seconds=0.5)
    if not verify_frames:
        print("FAIL — couldn't capture verification frame.")
        return 1

    matched = face_auth.verify_frame(verify_frames[0], encoding, threshold=0.5)
    print()
    if matched:
        print("=== PASS — face MATCH at threshold 0.5 ===")
        result = 0
    else:
        print("=== FAIL — no match. Bad lighting / pose / camera occluded? ===")
        result = 1

    # --- Cleanup ----------------------------------------------------------
    print()
    print(f"Cleaning up tmp encoding: {tmp_path}")
    if tmp_path.exists():
        tmp_path.unlink()

    return result


if __name__ == "__main__":
    raise SystemExit(main())
