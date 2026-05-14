"""One-time Ring auth + camera selection for M37.

Run this once: it prompts for your Ring email/password, handles the
SMS 2FA exchange, saves the OAuth refresh token to disk, lists the
cameras on your account, and lets you pick which one Jarvis should
watch. Subsequent Jarvis startups read the cached token + camera info
and proceed without prompting.

Usage:
    .\\venv\\Scripts\\python.exe setup_ring.py

Outputs written to %LOCALAPPDATA%\\Jarvis\\:
    ring_token.json   — OAuth tokens (access + refresh). Treat as secret.
    ring_config.json  — chosen camera id + name (not secret).

Re-run anytime to switch cameras or re-authenticate after Ring rotates
your token (happens once every few weeks/months — Jarvis will start
failing Ring polls with a 401 and the cached token will need refresh).
"""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import sys
from pathlib import Path

from ring_doorbell import Auth, Requires2FAError, Ring


_USER_AGENT = "Jarvis/1.0"
_BASE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
_TOKEN_PATH = _BASE / "ring_token.json"
_CONFIG_PATH = _BASE / "ring_config.json"


def _persist_token(token: dict) -> None:
    """Token-updater callback — called by ring_doorbell whenever the
    OAuth tokens are refreshed (initial fetch or background refresh).
    Writing to disk on every update means the next process always has
    the latest pair."""
    _BASE.mkdir(parents=True, exist_ok=True)
    _TOKEN_PATH.write_text(json.dumps(token))


async def main() -> int:
    print("=== Jarvis Ring setup (M37) ===")
    print()
    email = input("Ring account email: ").strip()
    if not email:
        print("No email provided. Aborting.", file=sys.stderr)
        return 1
    password = getpass.getpass("Ring account password: ")
    if not password:
        print("No password provided. Aborting.", file=sys.stderr)
        return 1

    auth = Auth(_USER_AGENT, None, _persist_token)
    try:
        try:
            await auth.async_fetch_token(email, password)
        except Requires2FAError:
            print()
            print("Ring sent a 2FA code via SMS. Enter it below.")
            code = input("2FA code: ").strip()
            if not code:
                print("No code entered. Aborting.", file=sys.stderr)
                return 1
            await auth.async_fetch_token(email, password, code)

        print(f"\nAuthenticated — token cached at {_TOKEN_PATH}")

        ring = Ring(auth)
        await ring.async_update_data()
        cameras = list(ring.video_devices())

        if not cameras:
            print("\nNo video devices found on this account. Make sure the "
                  "camera is registered + online in the Ring app.",
                  file=sys.stderr)
            return 1

        print(f"\nFound {len(cameras)} video device(s):")
        for i, cam in enumerate(cameras):
            print(f"  [{i}] {cam.name}  (model: {cam.model}, id: {cam.id})")

        if len(cameras) == 1:
            chosen = cameras[0]
            print(f"\nUsing the only camera: {chosen.name}")
        else:
            raw = input("\nWhich camera should Jarvis watch? [0]: ").strip()
            idx = int(raw) if raw else 0
            chosen = cameras[idx]

        config = {"device_id": chosen.id, "device_name": chosen.name}
        _CONFIG_PATH.write_text(json.dumps(config))
        print(f"\nSaved choice: {chosen.name} (id={chosen.id}) → {_CONFIG_PATH}")
        print("\nDone. Restart Jarvis to enable Ring-camera security events.")
        return 0
    finally:
        await auth.async_close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
