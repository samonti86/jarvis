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
import base64
import hmac
import io
import json
import ssl
import sys
import threading
import urllib.parse
from typing import Callable

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from src.remote_pwa import PWA_HTML, PWA_MANIFEST

_WS_PATH = "/ws"


def _log(msg: str) -> None:
    print(f"[remote] {msg}", file=sys.stderr)


def _build_silence_mp3() -> bytes | None:
    """M48.3 — ~200ms of silent MP3, generated once at server load via PyAV
    (already a `faster-whisper` transitive dep; no new requirement). Served
    from the new `/silence.mp3` route so the PWA can user-activate its
    `<audio>` element on the mic-start tap gesture, which is what unlocks
    iOS Safari TAB's stricter deferred-play rules (the installed home-screen
    PWA's lifecycle is more permissive, but the Safari tab path needs this).
    Returns None if MP3 encoding isn't available in this build of PyAV — the
    server still works fine, the PWA just falls back to the existing ▶︎
    button per the M48.2b accepted UX (no regression). The bytes are tiny
    (~1.2 KB) and the browser caches them after the first fetch."""
    try:
        import av  # noqa: PLC0415 — local import, never block startup
        import numpy as np  # noqa: PLC0415
        buf = io.BytesIO()
        container = av.open(buf, mode="w", format="mp3")
        try:
            stream = container.add_stream("libmp3lame", rate=22050)
            stream.layout = "mono"
            samples = np.zeros((1, int(0.2 * 22050)), dtype=np.int16)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.rate = 22050
            for pkt in stream.encode(frame):
                container.mux(pkt)
            for pkt in stream.encode(None):
                container.mux(pkt)
        finally:
            container.close()
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 — MP3 encoder absent on this build
        _log(f"silence.mp3 generation skipped ({type(exc).__name__}: {exc}) — "
             f"the PWA will fall back to the ▶︎ button on first voice turn")
        return None


_SILENCE_MP3: bytes | None = _build_silence_mp3()


class RemoteConsoleServer:
    """WebSocket + static-PWA server. One per process; started from main.py
    only when a token is configured.

    on_text(text)      — fed a user utterance; wire to the text_queue.
    on_control(action) — "arm" | "disarm" | "status"; returns a dict with
                          at least {"armed": bool}. Wire to SecurityWatcher.
    on_presence(event) — M70 geofenced auto-arm. "leave"/"arrive"/etc → a
                          result dict; wire to a PresenceController. Reached
                          via the token-gated HTTP /presence route, not the WS
                          session (an iOS Shortcuts automation fires it).
    """

    def __init__(
        self,
        token: str,
        host: str,
        port: int,
        on_text: Callable[[str], None] | None = None,
        on_control: Callable[[str], dict] | None = None,
        on_presence: Callable[[str], dict] | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._token = token
        self._host = host
        self._port = port
        # M48.3 prereq — when set, `serve(ssl=...)` makes the single listener
        # speak TLS for BOTH the PWA HTTP fetch and the WS upgrade (browser
        # sees https:// → upgrades to wss://, the PWA already detects this
        # via `location.protocol`, so no client-side change needed). None ⇒
        # plain HTTP/WS, today's LAN-mode behaviour, unchanged.
        self._ssl_context = ssl_context
        # Handlers are settable late: on_control is known at construction
        # (SecurityWatcher exists in main()), but on_text needs the
        # text_queue, which lives in listen_loop's scope and is wired once
        # that's up (mirrors main.py's on_enroll_face / on_knowledge_*
        # late-injection pattern). Safe stubs until then: an early phone
        # message is just dropped with a log, never an exception.
        # M48.2b: on_text now takes (text, reply_audio). reply_audio is a
        # thread-safe closure bound to THIS conn that unicasts a synthesized
        # mp3 back as {type:"speak"} — or None when the phone didn't ask for
        # audio (the "Speak replies" toggle off). Its presence/absence IS
        # the routing decision downstream ("audio → this phone, not the PC").
        self._on_text = on_text or (lambda t, ra: _log(f"text dropped (not wired yet): {t!r}"))
        # M48.3 — same late-injection pattern as on_text. on_audio takes
        # (blob_bytes, mime, reply_audio); the listen_loop wires it in once
        # text_queue exists. Until then a phone-voice frame just gets logged
        # and dropped — never an exception.
        self._on_audio: Callable[[bytes, str, Callable[[bytes], None] | None], None] = (
            lambda b, m, ra: _log(
                f"audio dropped (not wired yet): {len(b)}B {m!r}"
            )
        )
        self._on_control = on_control or (lambda a: {"armed": self._snapshot.get("armed", False)})
        # M70 — presence (geofenced auto-arm) hook. Settable late like
        # on_text, though in practice it's known at construction (the
        # PresenceController is built alongside SecurityWatcher). Safe stub
        # until wired: a presence ping is acknowledged but does nothing.
        self._on_presence: Callable[[str], dict] = on_presence or (
            lambda e: {"ok": False, "error": "presence not wired"}
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[ServerConnection] = set()
        self._thread: threading.Thread | None = None
        # Last-known surface so a phone connecting mid-session immediately
        # sees the correct state instead of a blank console.
        self._snapshot: dict = {"state": "idle", "armed": False}

    def set_on_text(
        self, fn: "Callable[[str, Callable[[bytes], None] | None], None]"
    ) -> None:
        self._on_text = fn

    def set_on_audio(
        self,
        fn: "Callable[[bytes, str, Callable[[bytes], None] | None], None]",
    ) -> None:
        """M48.3 — wire the phone push-to-talk callback. The listen_loop
        calls this once `text_queue` exists (same lifecycle as `set_on_text`)."""
        self._on_audio = fn

    def set_on_control(self, fn: Callable[[str], dict]) -> None:
        self._on_control = fn

    def set_on_presence(self, fn: Callable[[str], dict]) -> None:
        """M70 — wire the geofenced auto-arm handler (PresenceController)."""
        self._on_presence = fn

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

    @staticmethod
    def _swallow_proactor_resets(
        loop: asyncio.AbstractEventLoop, context: dict
    ) -> None:
        """M48.2c noise filter — silence the cosmetic asyncio ProactorEventLoop
        traceback that fires when a TLS peer drops mid-handshake. iOS Safari
        opens a couple of probe sockets while resolving HTTP/2 / OCSP / ALPN
        and abruptly closes the ones it doesn't use; the remote-side RST hits
        asyncio's `_call_connection_lost`, which calls
        `socket.shutdown(SHUT_RDWR)` on an already-gone socket and raises
        `ConnectionResetError [WinError 10054]`. The exception is handled
        inside asyncio (server keeps running, surviving connection completes
        the WS upgrade fine), but the default exception handler still logs
        the traceback — pure log bloat over time. Surgical filter: only swallow
        a `ConnectionResetError` whose callback is *specifically* the Proactor
        connection-lost cleanup; any other exception type or callback site
        still propagates to the default handler so we don't blind ourselves
        to real bugs."""
        exc = context.get("exception")
        msg = context.get("message", "")
        if (isinstance(exc, ConnectionResetError)
                and "_ProactorBasePipeTransport._call_connection_lost" in msg):
            return
        loop.default_exception_handler(context)

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._loop.set_exception_handler(self._swallow_proactor_resets)
        try:
            async with serve(
                self._handler,
                self._host,
                self._port,
                process_request=self._process_request,
                ssl=self._ssl_context,  # None ⇒ plain HTTP/WS; set ⇒ HTTPS/WSS
            ):
                scheme = "https" if self._ssl_context else "http"
                mode = (
                    "HTTPS/WSS (Tailscale secure-context — use the MagicDNS "
                    "name, not the IP, or the cert will mismatch)"
                    if self._ssl_context
                    else "HTTP/WS (LAN-mode; set JARVIS_TLS_CERT_FILE + "
                    "JARVIS_TLS_KEY_FILE to upgrade)"
                )
                _log(
                    f"listening on {self._host}:{self._port} ({mode}; "
                    f"token required). Open {scheme}://<this-host>:"
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
        if path == "/silence.mp3":
            # M48.3 — gesture-time `<audio>` element unlock. Returns the
            # pre-generated 200ms silent MP3 (built once at server load via
            # PyAV) or 404 if encoding wasn't available. The 404 case is
            # graceful: the PWA's play() rejects and falls back to ▶︎,
            # same as today's M48.2b accepted UX.
            if _SILENCE_MP3 is None:
                return self._http(
                    404, "text/plain; charset=utf-8",
                    "silence.mp3 not available (PyAV MP3 encoder absent)",
                    404,
                )
            return self._http_bytes(
                200, "audio/mpeg", _SILENCE_MP3,
                # Cacheable — the bytes never change for the server's lifetime,
                # and the PWA wants the browser to keep them.
                cache="public, max-age=86400, immutable",
            )
        if path == "/healthz":
            return self._http(200, "text/plain; charset=utf-8", "ok")
        if path == "/presence":
            return self._handle_presence(request)
        return self._http(404, "text/plain; charset=utf-8", "not found", 404)

    # ------------------------------------------------------------------
    # M70 — geofenced auto-arm endpoint (token-gated HTTP, not WS).
    # ------------------------------------------------------------------

    def _handle_presence(self, request: Request) -> Response:
        """An iOS Shortcuts personal automation (or any client) hits this on a
        geofence boundary:

            /presence?event=leave     → arm (deferred, flap-damped)
            /presence?event=arrive    → disarm + "welcome home, sir"

        Auth uses the SAME JARVIS_REMOTE_TOKEN as the WS console, supplied as
        an `Authorization: Bearer <token>` header (preferred), an
        `X-Jarvis-Token` header, or a `?token=` query param (least preferred —
        leaks into URLs/proxy logs). **GET only.** The `websockets` HTTP
        parser rejects any non-GET method at the request-line stage
        (`process_request` is never even invoked for a POST) — verified live,
        M70. That's fine: everything rides the query string + headers, and a
        body would be unread anyway (this is a WS server's HTTP shim, not a
        real HTTP framework). The iOS Shortcuts automation MUST use Method=GET
        (see .env.example). Returns a small JSON result. NEVER raises — a remote ping must not crash the loop."""
        parsed = urllib.parse.urlparse(request.path)
        params = urllib.parse.parse_qs(parsed.query)
        supplied = self._presence_token_from(
            request.headers.get("Authorization"),
            request.headers.get("X-Jarvis-Token"),
            (params.get("token") or [None])[0],
        )
        if not self._presence_authed(supplied):
            _log("presence: rejected (bad/missing token)")
            return self._http(
                401, "text/plain; charset=utf-8", "unauthorized", 401
            )
        event = ((params.get("event") or [""])[0] or "").strip()
        try:
            result = self._on_presence(event) or {}
        except Exception as exc:  # noqa: BLE001 — never let a remote ask crash
            _log(f"on_presence({event!r}) raised: {exc}")
            result = {"ok": False, "error": "internal"}
        _log(f"presence event={event!r} -> {result}")
        # State changes (the deferred arm firing, the disarm-on-arrive) reach
        # every client through SecurityWatcher's on_armed_changed →
        # ui.set_armed_indicator → update_armed broadcast — the same path the
        # tray/voice arming uses — so we deliberately do NOT broadcast here.
        return self._http(
            200, "application/json; charset=utf-8", json.dumps(result)
        )

    @staticmethod
    def _presence_token_from(
        auth_header: str | None,
        token_header: str | None,
        query_token: str | None,
    ) -> str:
        """Pull the bearer token from the request, preferring the most
        private channel: Authorization header → X-Jarvis-Token header →
        ?token= query param. Returns "" when none present."""
        if auth_header:
            stripped = auth_header.strip()
            # Accept "Bearer <tok>" (case-insensitive scheme) or a bare token.
            if stripped.lower().startswith("bearer "):
                return stripped[7:].strip()
            return stripped
        if token_header:
            return token_header.strip()
        if query_token:
            return query_token.strip()
        return ""

    def _presence_authed(self, supplied: str) -> bool:
        if not self._token or not supplied:
            return False
        # Constant-time compare — don't leak token length/prefix via timing.
        return hmac.compare_digest(supplied, self._token)

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

    @staticmethod
    def _http_bytes(
        status: int, ctype: str, data: bytes,
        cache: str = "no-store",
    ) -> Response:
        """M48.3 — binary response helper (vs `_http` which expects str body).
        Used for /silence.mp3; cacheable by default since the bytes never
        change for the lifetime of the server process."""
        headers = Headers(
            {
                "Content-Type": ctype,
                "Content-Length": str(len(data)),
                "Cache-Control": cache,
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
                #
                # M48.2b: the phone's "Speak replies" toggle rides per-message
                # ({"speak": true}) — stateless, no session-state desync. If
                # on, hand process_question a sink BOUND TO THIS conn so the
                # spoken reply is UNICAST back to the phone that asked (never
                # broadcast — only this device should hear it; the text
                # transcript keeps its fan-out). If off, pass None → today's
                # silent-phone-text behaviour, unchanged.
                want_audio = bool(msg.get("speak"))
                reply_audio = (
                    (lambda mp3: self._send_audio(conn, mp3))
                    if want_audio else None
                )
                try:
                    self._on_text(content, reply_audio)
                except Exception as exc:  # noqa: BLE001
                    _log(f"on_text raised: {exc}")
            return

        if mtype == "audio":
            # M48.3 — phone push-to-talk inbound. {mime, b64} where b64 is
            # the base64-encoded MediaRecorder Blob (audio/mp4 on Safari,
            # audio/webm on others — we don't trust the mime, PyAV auto-
            # detects from bytes; the field is logged only for diagnostics).
            #
            # Locked UX (set with the user during M48.3 scoping):
            # **voice-in always returns voice-out** — reply_audio is set
            # unconditionally for audio messages, ignoring the "Speak
            # replies" toggle (which continues to gate phone-TEXT only).
            # Simplest mental model: "if I pressed the mic, I want to hear
            # the answer." Less coupling, fewer surprises.
            mime = str(msg.get("mime", "audio/mp4"))
            b64 = str(msg.get("b64", ""))
            if not b64:
                return
            try:
                blob = base64.b64decode(b64, validate=False)
            except Exception:  # noqa: BLE001 — malformed input is dropped
                _log("audio decode failed (bad base64)")
                return
            if not blob:
                return
            reply_audio = lambda mp3: self._send_audio(conn, mp3)
            try:
                self._on_audio(blob, mime, reply_audio)
            except Exception as exc:  # noqa: BLE001
                _log(f"on_audio raised: {exc}")
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

    def _send_audio(self, conn: ServerConnection, mp3: bytes) -> None:
        """M48.2b — UNICAST a synthesized reply clip to one phone (the one
        that asked). Thread-safe: process_question runs on the listen_loop
        thread; marshal onto the server's loop exactly like broadcast().
        Not a broadcast — only the originating device should hear it (the
        text transcript fan-out is separate). Dropped if the loop is gone
        or the clip is empty (defensive — never break the turn)."""
        loop = self._loop
        if loop is None or not mp3:
            return
        payload = json.dumps({
            "type": "speak",
            "mime": "audio/mpeg",
            "b64": base64.b64encode(mp3).decode("ascii"),
        })
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._send_one(conn, payload))
            )
        except RuntimeError:
            pass  # loop closed during shutdown — nothing to do

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
