"""Ring camera watcher — second event source for security mode (M37).

Polls the user's Ring Indoor Cam via Ring's cloud API every few seconds.
When a motion event appears, fetches the snapshot and feeds it into
SecurityWatcher.trigger_external_motion() — same challenge / deterrent /
LOCKED flow as a local YOLO detection.

Why polling instead of push:
- `ring_doorbell` does have a Firebase Cloud Messaging listener for
  real-time events, but FCM setup is complex (separate auth, persistent
  websocket, project credentials). Polling at ~5s intervals is enough
  for a security alert — by the time the prompt finishes playing the
  user has the full 15s window anyway.

Why a dedicated asyncio loop in a thread:
- ring_doorbell 0.9.x is async-only (all methods are `async_*`). The
  rest of Jarvis is sync/threaded. Running asyncio.run() in a daemon
  thread gives Ring its own event loop without infecting the main
  process model. Same pattern would work for any future async-only
  integration.

Token lifecycle:
- Initial auth (with 2FA) happens once via `setup_ring.py`. That writes
  `ring_token.json` + `ring_config.json` to %LOCALAPPDATA%\\Jarvis\\.
- This watcher reads those files at start, builds Auth with the cached
  token + a token-updater callback, and Ring's lib auto-refreshes when
  needed. The updater callback writes the refreshed pair back to disk
  so the next process startup uses the latest.
- If Ring rotates the refresh token (rare; happens after suspicious
  activity), `async_update_data` returns 401 and the watcher logs +
  exits cleanly. User re-runs setup_ring.py to re-authenticate.

Defensive contract:
- Missing token / config files → watcher logs once and exits; Jarvis
  runs without Ring (same shape as Plex when PLEX_URL is unset).
- Network errors → caught, logged, polling continues.
- Snapshot fetch failure → log + still fire the motion event with empty
  bytes; the deterrent path tolerates an empty evidence frame.
- Any uncaught error in the poll loop → logged, watcher exits cleanly,
  doesn't affect the rest of Jarvis.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.security import SecurityWatcher


_USER_AGENT = "Jarvis/1.0"
_BASE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"
_TOKEN_PATH = _BASE / "ring_token.json"
_CONFIG_PATH = _BASE / "ring_config.json"

# Poll interval in seconds. 5s = 12 requests/minute, well under Ring's
# rate limits (anecdotally ~60/min). Worst-case detection latency for the
# user is poll-interval + Ring's own event-propagation latency (~1-3s
# from camera to cloud), so ~5-8s from the actual motion to the
# challenge prompt firing. Acceptable for security; not great for
# real-time but real-time needs FCM which is overkill here.
_POLL_INTERVAL_SECONDS = 5.0


def is_configured() -> bool:
    """True if setup_ring.py has been run successfully (token + camera
    selection both saved). Caller can use this to skip RingWatcher
    instantiation entirely when Ring isn't set up."""
    return _TOKEN_PATH.is_file() and _CONFIG_PATH.is_file()


class RingWatcher:
    """Polls Ring's cloud API for motion events and feeds them into
    SecurityWatcher. Lifecycle is bound to the security armed state —
    main.py calls start() on activate, stop() on deactivate."""

    def __init__(self, security_watcher: "SecurityWatcher") -> None:
        self._security = security_watcher
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Set the first time we successfully read history; subsequent polls
        # use it to filter for "events newer than this".
        self._last_event_id: int | None = None

    def start(self) -> None:
        """Spawn the poll-loop thread. Idempotent — re-starting an already-
        running watcher is a no-op."""
        if self._thread is not None and self._thread.is_alive():
            return
        if not is_configured():
            print("[ring] token or config missing — run setup_ring.py first",
                  file=sys.stderr)
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="RingWatcher", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the poll loop to exit. Idempotent. Daemon thread dies
        with the process either way; explicit stop just lets the current
        poll cycle wind down cleanly."""
        self._stop.set()

    # ---------------------------------------------------------------------
    # Internals — all run on the watcher thread or its asyncio loop.
    # ---------------------------------------------------------------------

    def _run(self) -> None:
        """Top of the watcher thread. Runs the async poll loop until
        _stop fires (or an unrecoverable error)."""
        try:
            asyncio.run(self._async_main())
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[ring] watcher crashed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        finally:
            print("[ring] watcher exited", file=sys.stderr)

    async def _async_main(self) -> None:
        """Authenticate, look up the configured camera, run the poll loop."""
        token = self._load_json(_TOKEN_PATH)
        config = self._load_json(_CONFIG_PATH)
        if token is None or config is None:
            return  # already logged by _load_json

        # Local import — ring_doorbell pulls firebase-messaging +
        # websockets transitively, ~20MB of imports. Don't pay this
        # for users who never configured Ring.
        from ring_doorbell import Auth, Ring

        auth = Auth(_USER_AGENT, token, self._persist_token)
        ring = Ring(auth)

        try:
            try:
                await ring.async_update_data()
            except Exception as exc:  # noqa: BLE001
                print(f"[ring] initial auth/data refresh failed: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                print("[ring] token may have expired — re-run setup_ring.py",
                      file=sys.stderr)
                return

            cam = self._find_camera(ring, config)
            if cam is None:
                return

            print(f"[ring] watching {cam.name} (id={cam.id}, model={cam.model})",
                  file=sys.stderr)

            # Establish baseline: the most recent event at startup is NOT
            # something we should fire on (it could be hours/days old).
            try:
                events = await cam.async_history(limit=1)
                if events:
                    self._last_event_id = events[0].id
            except Exception as exc:  # noqa: BLE001
                print(f"[ring] baseline history fetch failed: {exc}",
                      file=sys.stderr)

            while not self._stop.is_set():
                # asyncio sleep that wakes on stop — poll the event quickly
                # so disarming feels responsive.
                for _ in range(int(_POLL_INTERVAL_SECONDS * 10)):
                    if self._stop.is_set():
                        return
                    await asyncio.sleep(0.1)
                if self._stop.is_set():
                    return
                try:
                    await self._poll_once(ring, cam)
                except Exception as exc:  # noqa: BLE001
                    # Transient errors are expected (Wi-Fi blip, Ring API
                    # hiccup). Log and keep polling.
                    print(f"[ring] poll iteration failed: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            try:
                await auth.async_close()
            except Exception:
                pass

    async def _poll_once(self, ring, cam) -> None:
        """One polling cycle: refresh, check history, dispatch new motion."""
        await ring.async_update_data()
        events = await cam.async_history(limit=5)
        if not events:
            return

        # Events are returned newest-first. Collect everything newer than
        # our last seen id.
        new_events = []
        for event in events:
            if event.id == self._last_event_id:
                break
            new_events.append(event)

        if not new_events:
            return

        # Bump baseline to the newest event we just saw — important to do
        # this BEFORE firing alerts, so a slow snapshot fetch can't cause
        # us to re-process the same event on the next poll.
        self._last_event_id = events[0].id

        for event in new_events:
            # Ring event "kind" is "motion" for camera motion, plus other
            # kinds for doorbells (ding) and various device-specific events.
            # Filter to motion only — that's the security signal.
            kind = getattr(event, "kind", "")
            if kind != "motion":
                continue
            await self._fire_alert(cam, event)

    async def _fire_alert(self, cam, event) -> None:
        """Fetch snapshot for an event and call into SecurityWatcher.
        Snapshot is best-effort — empty bytes is acceptable, the
        deterrent path still fires with no image attached."""
        print(f"[ring] motion event id={event.id} at {event.created_at}",
              file=sys.stderr)
        snap_bytes = b""
        try:
            snap_bytes = await cam.async_get_snapshot()
        except Exception as exc:  # noqa: BLE001
            print(f"[ring] snapshot fetch failed: {exc} — proceeding without",
                  file=sys.stderr)
        # Hop back to sync land for the SecurityWatcher call. trigger_external_motion
        # is fully sync and thread-safe — it just enters the challenge state and
        # returns immediately, doesn't block on TTS or disk I/O.
        try:
            self._security.trigger_external_motion(
                source="ring", jpeg_bytes=snap_bytes or None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ring] trigger_external_motion raised: {exc}", file=sys.stderr)

    # ---------------------------------------------------------------------
    # Helpers.
    # ---------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            print(f"[ring] missing file: {path} — run setup_ring.py",
                  file=sys.stderr)
            return None
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[ring] couldn't read {path}: {exc}", file=sys.stderr)
            return None

    @staticmethod
    def _persist_token(token: dict) -> None:
        """Token-updater callback for Ring's Auth. ring_doorbell calls
        this when it refreshes the OAuth token in the background; we
        persist so the next Jarvis startup uses the fresh token."""
        try:
            _BASE.mkdir(parents=True, exist_ok=True)
            _TOKEN_PATH.write_text(json.dumps(token))
        except OSError as exc:
            print(f"[ring] failed to persist refreshed token: {exc}",
                  file=sys.stderr)

    @staticmethod
    def _find_camera(ring, config: dict):
        """Look up the camera the user picked during setup. Prefer device
        id (stable across rename); fall back to name (in case the device
        was re-paired and got a new id)."""
        device_id = config.get("device_id")
        device_name = config.get("device_name")
        for cam in ring.video_devices():
            if cam.id == device_id:
                return cam
        # ID didn't match — try name as fallback.
        for cam in ring.video_devices():
            if cam.name == device_name:
                print(f"[ring] device id changed; matched by name '{device_name}' "
                      f"(re-run setup_ring.py to refresh the id)",
                      file=sys.stderr)
                return cam
        print(f"[ring] camera id={device_id} name={device_name!r} not found "
              f"on account — run setup_ring.py to pick again", file=sys.stderr)
        return None
