"""M48.1 — LAN-only remote console (phone/web thin client to THIS PC's brain).

Jarvis is NOT installed on the phone. This is a thin client: a WebSocket
server in a daemon thread, additive to the unchanged assistant.

  - Conversational text rides the EXISTING text-queue seam (`on_text` →
    main.py's text_queue → process_question, untouched).
  - Control (arm/disarm/status) calls the SecurityWatcher directly via
    `on_control` — exactly the tray-toggle path, bypassing the LLM
    (instant, free, reliable: "I forgot to arm him on the way out").
  - Replies + state reach the phone because RemoteBridge is a THIRD sink
    on the JarvisUI fan-out facade (alongside console + tray). The brain
    never knows the phone exists.

Security (v1, LAN): the WS session must present JARVIS_REMOTE_TOKEN as its
first message (constant-time compare). Static PWA assets are inert and
served unauthenticated — the bearer of value is the live WS session, not
the HTML. The token + a LAN-scoped firewall rule are the controls; never
port-forward (off-LAN = Tailscale, M48.4). If the token is blank the
server is never constructed (main.py gate) — a surface that can DISARM
security must not exist unless deliberately armed with a secret
([[feedback-jarvis-least-privilege]]).

Defensive contract (same as every optional subsystem): nothing here may
break the assistant. Every exception is caught + logged; a dead or blocked
server thread just means "no remote console", never a crashed Jarvis.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import sys
import threading
from typing import Callable

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from src.remote_pwa import PWA_HTML, PWA_MANIFEST

_WS_PATH = "/ws"


def _log(msg: str) -> None:
    print(f"[remote] {msg}", file=sys.stderr)


class RemoteConsoleServer:
    """WebSocket + static-PWA server. One per process; started from main.py
    only when a token is configured.

    on_text(text)      — fed a user utterance; wire to the text_queue.
    on_control(action) — "arm" | "disarm" | "status"; returns a dict with
                          at least {"armed": bool}. Wire to SecurityWatcher.
    """

    def __init__(
        self,
        token: str,
        host: str,
        port: int,
        on_text: Callable[[str], None] | None = None,
        on_control: Callable[[str], dict] | None = None,
    ) -> None:
        self._token = token
        self._host = host
        self._port = port
        # Handlers are settable late: on_control is known at construction
        # (SecurityWatcher exists in main()), but on_text needs the
        # text_queue, which lives in listen_loop's scope and is wired once
        # that's up (mirrors main.py's on_enroll_face / on_knowledge_*
        # late-injection pattern). Safe stubs until then: an early phone
        # message is just dropped with a log, never an exception.
        self._on_text = on_text or (lambda t: _log(f"text dropped (not wired yet): {t!r}"))
        self._on_control = on_control or (lambda a: {"armed": self._snapshot.get("armed", False)})

        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[ServerConnection] = set()
        self._thread: threading.Thread | None = None
        # Last-known surface so a phone connecting mid-session immediately
        # sees the correct state instead of a blank console.
        self._snapshot: dict = {"state": "idle", "armed": False}

    def set_on_text(self, fn: Callable[[str], None]) -> None:
        self._on_text = fn

    def set_on_control(self, fn: Callable[[str], dict]) -> None:
        self._on_control = fn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the server on a daemon thread with its own asyncio loop
        (the app is threaded; an isolated loop is the clean integration)."""
        self._thread = threading.Thread(
            target=self._run, name="RemoteConsole", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # noqa: BLE001 — must never take down Jarvis
            _log(f"server thread crashed (remote console disabled): {exc!r}")

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            async with serve(
                self._handler,
                self._host,
                self._port,
                process_request=self._process_request,
            ):
                _log(
                    f"listening on {self._host}:{self._port} "
                    f"(LAN-only; token required). Open http://<this-pc-ip>:"
                    f"{self._port}/ on the phone."
                )
                await asyncio.Future()  # run forever
        except OSError as exc:
            # Port in use / bind denied — degrade, don't crash.
            _log(f"could not bind {self._host}:{self._port}: {exc}")

    # ------------------------------------------------------------------
    # HTTP: serve the one-page PWA; let the WS path upgrade.
    # ------------------------------------------------------------------

    def _process_request(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        path = request.path.split("?", 1)[0]
        if path == _WS_PATH:
            return None  # proceed to the WebSocket handshake
        if path in ("/", "/index.html"):
            return self._http(200, "text/html; charset=utf-8", PWA_HTML)
        if path == "/manifest.json":
            return self._http(200, "application/manifest+json", PWA_MANIFEST)
        if path == "/healthz":
            return self._http(200, "text/plain; charset=utf-8", "ok")
        return self._http(404, "text/plain; charset=utf-8", "not found", 404)

    @staticmethod
    def _http(
        _status: int, ctype: str, body: str, status: int = 200
    ) -> Response:
        data = body.encode("utf-8")
        headers = Headers(
            {
                "Content-Type": ctype,
                "Content-Length": str(len(data)),
                # PWA is a single inert page; no caching keeps iterate-fast.
                "Cache-Control": "no-store",
            }
        )
        return Response(status, "OK" if status == 200 else "ERROR", headers, data)

    # ------------------------------------------------------------------
    # WebSocket session
    # ------------------------------------------------------------------

    async def _handler(self, conn: ServerConnection) -> None:
        # 1) Auth handshake — first frame must be the token. Anything else,
        #    or a mismatch, is rejected and the socket closed. No token is
        #    ever logged.
        try:
            raw = await asyncio.wait_for(conn.recv(), timeout=10.0)
        except (TimeoutError, asyncio.TimeoutError):
            await self._safe_send(conn, {"type": "auth_fail", "reason": "timeout"})
            return
        except Exception:  # noqa: BLE001 — client vanished mid-handshake
            return

        if not self._authed(raw):
            await self._safe_send(conn, {"type": "auth_fail"})
            _log("rejected a connection (bad/missing token)")
            return

        # 2) Authenticated. Register, send ack + the current snapshot so the
        #    phone renders correct state immediately.
        self._clients.add(conn)
        await self._safe_send(conn, {"type": "auth_ok"})
        await self._safe_send(conn, {"type": "snapshot", **self._snapshot})
        _log(f"client connected ({len(self._clients)} now)")

        # 3) Message loop. Two classes only: converse + control.
        try:
            async for message in conn:
                await self._dispatch(conn, message)
        except Exception:  # noqa: BLE001 — normal disconnect throws; fine
            pass
        finally:
            self._clients.discard(conn)
            _log(f"client disconnected ({len(self._clients)} now)")

    def _authed(self, raw: object) -> bool:
        if not self._token:
            return False
        try:
            msg = json.loads(raw if isinstance(raw, str) else raw.decode())
        except Exception:  # noqa: BLE001
            return False
        if not isinstance(msg, dict) or msg.get("type") != "auth":
            return False
        supplied = str(msg.get("token", ""))
        # Constant-time compare — don't leak token length/prefix via timing.
        return hmac.compare_digest(supplied, self._token)

    async def _dispatch(self, conn: ServerConnection, message: object) -> None:
        try:
            msg = json.loads(message if isinstance(message, str) else message.decode())
        except Exception:  # noqa: BLE001
            return
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")

        if mtype == "text":
            content = str(msg.get("content", "")).strip()
            if content:
                # No optimistic echo: every user turn (voice OR typed, any
                # origin) appears exactly once via the JarvisUI.add_user_text
                # fan-out — the single source of truth. Echoing here too
                # would double phone-typed lines.
                try:
                    self._on_text(content)
                except Exception as exc:  # noqa: BLE001
                    _log(f"on_text raised: {exc}")
            return

        if mtype == "control":
            action = str(msg.get("action", ""))
            if action not in ("arm", "disarm", "status"):
                return
            try:
                result = self._on_control(action) or {}
            except Exception as exc:  # noqa: BLE001
                _log(f"on_control({action}) raised: {exc}")
                result = {}
            armed = bool(result.get("armed", self._snapshot.get("armed", False)))
            self._snapshot["armed"] = armed
            # Tell every client (not just the requester) so two phones stay
            # consistent and the requester gets definitive confirmation.
            self.broadcast({"type": "armed", "armed": armed, "via": action})
            return

        # Unknown message types are ignored by design (forward-compat).

    # ------------------------------------------------------------------
    # Outbound — thread-safe broadcast (called from JarvisUI's threads).
    # ------------------------------------------------------------------

    def broadcast(self, payload: dict) -> None:
        """Schedule `payload` to every connected client. Safe to call from
        ANY thread (the JarvisUI facade runs on the listen/announcer
        threads); marshals onto the server's loop. Pre-loop / no-client
        calls are dropped — a late client is caught up by the snapshot."""
        loop = self._loop
        if loop is None or not self._clients:
            return
        try:
            loop.call_soon_threadsafe(self._do_broadcast, payload)
        except RuntimeError:
            pass  # loop closed during shutdown — nothing to do

    def _do_broadcast(self, payload: dict) -> None:
        data = json.dumps(payload)
        for conn in list(self._clients):
            # Fire-and-forget; a slow/dead client must not block others or
            # the loop. websockets buffers + drops on its own close.
            asyncio.create_task(self._send_one(conn, data))

    async def _send_one(self, conn: ServerConnection, data: str) -> None:
        try:
            await conn.send(data)
        except Exception:  # noqa: BLE001 — client gone; handler cleans up
            self._clients.discard(conn)

    @staticmethod
    async def _safe_send(conn: ServerConnection, payload: dict) -> None:
        try:
            await conn.send(json.dumps(payload))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Snapshot updates — called by RemoteBridge (the JarvisUI sink).
    # ------------------------------------------------------------------

    def update_state(self, state: str) -> None:
        self._snapshot["state"] = state
        self.broadcast({"type": "state", "state": state})

    def update_armed(self, armed: bool) -> None:
        self._snapshot["armed"] = armed
        self.broadcast({"type": "armed", "armed": armed, "via": "system"})

    def push_line(self, role: str, text: str) -> None:
        """role: 'user' | 'jarvis' | 'system'. Mirrors a transcript line."""
        if text:
            self.broadcast({"type": role, "text": text})
