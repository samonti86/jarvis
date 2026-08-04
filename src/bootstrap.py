"""Subsystem wiring — the composition root's helpers.

Extracted from main.py (2026-07-28). main() had grown to ~590 lines of
`main()` plus ~490 lines of the builders it calls, which put the entry point
well past the god-file threshold this project sets for itself.

What lives here is everything that *assembles* a subsystem and hands back a
handle: the two optional Plex connections, the announcement fan-out, the
remote-console server, the status-report registry, and the ordered shutdown.
What does NOT live here is any per-turn logic — that is TurnRunner's job.

Everything here honours the project's fail-soft contract: an optional
subsystem that cannot start logs and returns None (or a no-op), and never
raises into the caller. main() is therefore a sequence of "try to build X"
steps that cannot individually kill startup.

These were `_`-prefixed while they were private to main.py. They are this
module's public API now, so the underscore is gone.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from typing import Callable, NamedTuple

from src import quiet_hours
from src.config import Config
from src.gates import CountedEvent
from src.plex_laptop import DEFAULT_LOG_PATH as DEFAULT_PLEX_LAPTOP_LOG, PlexLaptopClient
from src.plex_mcp import PlexMCPClient
from src.text_to_speech import speak_streaming
from src.tray import State
from src.ui import JarvisUI


def try_connect_plex(cfg: Config) -> PlexMCPClient | None:
    """M21: best-effort Plex MCP connection. Per the project contract,
    Plex is optional — any failure (no creds, server down, MCP server
    crashes, transitive dep import fail) logs and returns None. The rest
    of Jarvis runs unaffected."""
    if not cfg.plex_url or not cfg.plex_token:
        print("[plex-mcp] PLEX_URL/PLEX_TOKEN unset — Plex tools disabled", file=sys.stderr)
        return None
    try:
        client = PlexMCPClient(cfg.plex_url, cfg.plex_token)
    except Exception as exc:
        print(f"[plex-mcp] connect failed ({exc}); continuing without Plex tools", file=sys.stderr)
        return None
    print(f"[plex-mcp] connected — {len(client.tools)} tools available", file=sys.stderr)
    return client


def try_connect_plex_laptop(cfg: Config) -> PlexLaptopClient | None:
    """M24: best-effort SSH client for the Plex laptop diagnostic tools.

    Lazy connect — we don't actually open the TCP+SSH handshake here, just
    construct the client (which validates params). The first tool call opens
    the connection. That keeps Jarvis startup fast even when the laptop is
    asleep or off-network; voice queries to other tools work normally, and
    only an actual plex_logs_* / plex_laptop_health call would hit the
    "Plex laptop unreachable" string.

    Any failure (missing host/user, bad key path, paramiko import problem)
    logs and returns None. The rest of Jarvis runs unaffected.
    """
    if not cfg.plex_laptop_host or not cfg.plex_laptop_user:
        print(
            "[plex-laptop] PLEX_LAPTOP_HOST/USER unset — remote tools disabled",
            file=sys.stderr,
        )
        return None
    key_path = cfg.plex_laptop_key_path or os.path.expanduser("~/.ssh/id_ed25519")
    log_path = cfg.plex_laptop_log_path or DEFAULT_PLEX_LAPTOP_LOG
    try:
        client = PlexLaptopClient(
            host=cfg.plex_laptop_host,
            user=cfg.plex_laptop_user,
            key_path=key_path,
            log_path=log_path,
        )
    except Exception as exc:
        print(
            f"[plex-laptop] init failed ({exc}); continuing without remote tools",
            file=sys.stderr,
        )
        return None
    print(
        f"[plex-laptop] ready — will connect to {cfg.plex_laptop_user}@{cfg.plex_laptop_host} on first tool call",
        file=sys.stderr,
    )
    return client


class Announcer(NamedTuple):
    """Handle returned by build_announcer: the public announce fn, the
    cooperative speech-gate Event, and a shutdown callable."""
    announce: "Callable[..., None]"
    speaking_event: threading.Event
    shutdown: "Callable[[], None]"


def build_announcer(ui: JarvisUI, pc_speaking: threading.Event) -> Announcer:
    """Construct the dedicated proactive-speech subsystem (M34).

    `pc_speaking` (2026-05-29) is the "PC is speaking out loud" gate shared
    with TurnRunner — set here too while an announce plays, so the always-on
    voice-capture loop discards self-audio (the omni-mic echo fix). Distinct
    from `speaking_event` below (that gates security/sound CPU bursts; this
    gates mic capture).

    Speech is marshaled through a single dedicated Announcer thread because
    sd.play() on Windows WASAPI silently no-ops when called from worker
    threads that haven't previously played audio (the SecurityWatcher thread
    hits this exact case). Pinning all sd.play() calls to one thread sidesteps
    it — see the project_wasapi_thread_audio_owner memory.

    Returns an Announcer with `announce(text, on_done=None, label="🚨")` (the
    non-blocking entry every proactive caller uses), `speaking_event` (the
    cooperative speech gate the SecurityWatcher + SoundDetector defer their
    heavy CPU bursts against), and `shutdown()` for the wind-down sequence.

    ─────────────────────────────────────────────────────────────────────
    CONTRACT — `speaking_event` is the cooperative speech gate. It is SET
    while a proactive announce is on the wire and CLEARED when it finishes.

    Any background thread that does a sustained CPU/GIL burst (ML inference,
    a tight encode loop, a per-tick gc) MUST consume this Event and yield
    (defer or drop its work) while it is set — otherwise the burst starves
    the Python-fed TTS path, the audio buffer underruns, and the announce
    stutters. Treat it like a lock you must respect, not an optional hint.

    Current consumers: SecurityWatcher (defers grab/YOLO/encode — see
    `_is_announcing`/`_wait_for_quiet`) and SoundDetector (drops the PANNs
    inference window). A NEW continuous-CPU subsystem is not done until it is
    wired in here too — that omission is exactly the 2026-05-28 M67
    regression (M58's acoustic loop shipped outside the gate). See the
    stutter-gate post-mortem in docs/MILESTONES.md +
    project_security_audio_stutter_gate memory.
    ─────────────────────────────────────────────────────────────────────
    """
    # Queue items are (text, on_done, label) tuples; on_done fires after
    # playback so callers can defer state until the user has actually heard
    # the prompt (SecurityWatcher's 15s challenge timer uses this); label is
    # the console emoji tag (🚨 alert / ⏰ reminder). maxsize caps memory
    # growth if TTS wedges + announces pile up.
    announce_queue: queue.Queue[tuple[str, Callable[[], None] | None, str] | None] = (
        queue.Queue(maxsize=16)
    )
    announce_stop = threading.Event()
    # Set WHILE a proactive announce is actually playing. The SecurityWatcher
    # reads this to defer its heavy vision bursts (camera grab, YOLO + the
    # per-tick gc.collect, face encode) so they never overlap speech — the
    # cooperative speech gate that fixes the armed-only TTS stutter
    # (diagnosed 2026-05-19: any GIL/CPU burst overlapping the Python-fed
    # TTS path underruns the audio buffer). Owned by the Announcer thread.
    # 2026-07-02 QA: a CountedEvent, because turn replies ALSO raise this gate
    # (M68) and an announce can overlap a playing reply — with a plain Event,
    # whichever speaker finished FIRST dropped the gate while the other was
    # still talking (armed CPU loops resumed mid-speech). set/clear now nest.
    announce_speaking = CountedEvent()

    def _announcer_loop() -> None:
        """Dedicated proactive-speech thread (see docstring above for why).

        Each queued item is (text, on_done, label). on_done is fired in
        `finally` after playback so it runs even if TTS itself failed — a
        failed announce shouldn't strand the caller's state machine."""
        print("[announcer] worker thread started", file=sys.stderr)
        while not announce_stop.is_set():
            item = announce_queue.get()
            if item is None or announce_stop.is_set():
                break
            text, on_done, label = item
            # 2026-08-04: the ENTIRE per-item body is guarded. This thread is
            # the only path by which Jarvis speaks unprompted — reminders,
            # weather, homelab, and security alerts — so its death is silent
            # and total. It died exactly once, on 2026-07-30 19:17:59, when
            # ui.add_system_text (then unguarded, and outside the try below)
            # hit Tk before the mainloop was up; Jarvis ran mute for the next
            # 12+ hours overnight and nothing in the log said so. The UI facade
            # is now defensive too (ui._console_call), which fixes that
            # specific fault — this guard is the structural version: no callee
            # gets to kill the announcer, whether or not it was polite.
            try:
                print(f"[announce] {text}")
                # The console line is DECORATION; the speech is the product.
                # Guarded separately so a dead window costs a transcript line,
                # never the announcement itself — the regression test asserts
                # that a UI failure still lets this item be spoken.
                try:
                    ui.add_system_text(f"{label} {text}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[announce] transcript line skipped ({exc})",
                          file=sys.stderr)
                # Bracket actual playback so the watcher's cooperative gate
                # (security._is_announcing) defers heavy bursts for exactly the
                # window speech is on the wire. Cleared in finally so a TTS
                # failure can't leave the watcher permanently throttled.
                announce_speaking.set()
                pc_speaking.set()  # suppress voice-capture self-hearing
                try:
                    speak_streaming(
                        iter([text]),
                        "en",
                        on_first_audio=lambda: ui.set_state(State.SPEAKING),
                        on_amplitude=ui.set_amplitude,
                    )
                except Exception as exc:
                    print(f"[announce] TTS failed: {exc}", file=sys.stderr)
                finally:
                    announce_speaking.clear()
                    pc_speaking.clear()
                    ui.set_state(State.IDLE)
                    if on_done is not None:
                        try:
                            on_done()
                        except Exception as exc:
                            print(f"[announce] on_done callback raised: {exc}",
                                  file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 — the loop MUST survive
                print(f"[announce] item failed ({type(exc).__name__}: {exc}) — "
                      f"announcer continuing", file=sys.stderr)
        print("[announcer] worker thread exited", file=sys.stderr)

    threading.Thread(
        target=_announcer_loop, name="Announcer", daemon=True
    ).start()

    def _announce(
        text: str,
        on_done: Callable[[], None] | None = None,
        label: str = "🚨",
    ) -> None:
        """Public entry: enqueue a proactive announcement. Non-blocking —
        returns immediately and the Announcer thread plays it. Bypasses
        the mute check (security alerts override quiet mode by design).
        Multiple announcements queue FIFO so they don't overlap.

        on_done (optional) fires AFTER the playback completes (or fails) —
        used by SecurityWatcher's challenge path to defer the 15s timer
        start until the user has actually heard the prompt, otherwise
        prompt-playback time eats into the response budget.

        label (optional) tags the console line — 🚨 for the default
        (security) announce, ⏰ for an M53 reminder firing. It is ALSO the
        severity signal the M79 quiet-hours policy reads: during the configured
        DND window a routine status-monitoring label (🖥) is held back and
        recorded for the morning briefing's catch-up, while everything else
        pierces (the safe default). DND off ⇒ unchanged behaviour."""
        if not text:
            return
        if quiet_hours.decide(label) == "defer":
            # Held back by quiet hours — record it for the briefing catch-up,
            # don't speak, don't fire on_done (on_done is a timing callback used
            # only by the critical security path, which always pierces).
            quiet_hours.record_deferred(text, label)
            print(f"[announce] quiet hours — deferred ({label}): {text!r}",
                  file=sys.stderr)
            return
        try:
            announce_queue.put_nowait((text, on_done, label))
        except queue.Full:
            # Queue is wedged (TTS playback stuck?). Log + drop newest so
            # callers don't block. on_done won't fire — callers using it
            # for timing must handle a None return, but currently the only
            # caller is SecurityWatcher's challenge path which has its
            # own defensive immediate-arm fallback.
            print(
                f"[announce] queue full (TTS wedged?) — dropping: {text!r}",
                file=sys.stderr,
            )

    def _shutdown() -> None:
        # M34: stop the Announcer thread. Sentinel-None wakes the .get() in
        # _announcer_loop so it can check the stop flag and exit cleanly,
        # instead of being killed mid-playback as a daemon.
        announce_stop.set()
        # Non-blocking: announce_stop already guarantees the loop exits on its
        # next iteration; the sentinel only needs to wake a *blocked* get(), and
        # a full queue (maxsize 16) means get() ISN'T blocked. A plain blocking
        # put() here could hang shutdown forever if the queue is full and TTS is
        # wedged in speak_streaming — the case _announce already guards with
        # put_nowait.
        try:
            announce_queue.put_nowait(None)
        except queue.Full:
            pass

    return Announcer(_announce, announce_speaking, _shutdown)


def build_remote_console(
    cfg: Config, ui: JarvisUI, security_watcher, announce
) -> object | None:
    """M48.1 LAN remote console. Returns the started RemoteConsoleServer, or
    None when disabled (no token) or on any startup failure.

    Gated on the token — blank ⇒ the server is NEVER constructed (a surface
    that can disarm security must not exist unless deliberately configured
    with a secret; [[feedback-jarvis-least-privilege]]). on_control is wired
    here (SecurityWatcher is in scope); on_text is wired later in listen_loop
    once the text_queue exists. Optional + defensive: a failure to start logs
    and returns None — never blocks the assistant."""
    if not cfg.remote_token:
        print("[remote] JARVIS_REMOTE_TOKEN unset — remote console disabled",
              file=sys.stderr)
        return None

    def _remote_control(action: str) -> dict:
        try:
            if action == "arm":
                security_watcher.activate()
            elif action == "disarm":
                security_watcher.deactivate()
            # "status" is read-only — just report below.
        except Exception as exc:  # noqa: BLE001 — never break on a remote ask
            print(f"[remote] control {action!r} failed: {exc}", file=sys.stderr)
        return {"armed": bool(security_watcher.is_armed())}

    # M70 — geofenced auto-arm. The PresenceController owns the policy (deferred
    # arm + flap damping + greet-on-real-transition); main.py just binds it to
    # the live SecurityWatcher and the WASAPI-safe announce path (tagged 🏠).
    # Reached via the token-gated /presence HTTP route an iOS Shortcuts
    # automation fires — same secret as the WS console.
    from src.presence import PresenceController  # noqa: PLC0415
    _presence = PresenceController(
        arm=security_watcher.activate,
        disarm=security_watcher.deactivate,
        is_armed=security_watcher.is_armed,
        announce=lambda t: announce(t, label="🏠"),
        arm_delay=cfg.presence_arm_delay,
        greeting=cfg.presence_greeting,
    )

    try:
        # Import is INSIDE the try: a missing `websockets` (or any
        # import error in remote_console/remote_pwa) must degrade to
        # "no remote console", never an unhandled ImportError that
        # breaks startup — the project's optional-component contract.
        import ssl as _ssl  # noqa: PLC0415 — lazy: only needed when TLS configured
        from src.remote_console import RemoteConsoleServer  # noqa: PLC0415

        # M48.3 prereq — opportunistic TLS: build an SSLContext ONLY if
        # both .env paths are set AND both files exist on disk. Either
        # missing or unreadable ⇒ log loudly and fall back to plain
        # HTTP/WS (LAN-mode dev path). A misconfigured cert MUST NOT
        # crash Jarvis — same optional-component contract as the rest of
        # the remote console. Stays defensively quiet about secret
        # values: only the *paths* (not contents) ever reach the log.
        ssl_ctx = None
        cf, kf = cfg.tls_cert_file, cfg.tls_key_file
        if cf or kf:
            if not (cf and kf):
                print(
                    "[remote] TLS half-configured (only one of "
                    "JARVIS_TLS_CERT_FILE/JARVIS_TLS_KEY_FILE set) — "
                    "falling back to plain HTTP/WS",
                    file=sys.stderr,
                )
            elif not (os.path.isfile(cf) and os.path.isfile(kf)):
                print(
                    f"[remote] TLS files not found "
                    f"(cert={cf!r}, key={kf!r}) — falling back to "
                    f"plain HTTP/WS",
                    file=sys.stderr,
                )
            else:
                try:
                    ssl_ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
                    ssl_ctx.load_cert_chain(certfile=cf, keyfile=kf)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[remote] TLS load failed ({exc!r}) — "
                        f"falling back to plain HTTP/WS",
                        file=sys.stderr,
                    )
                    ssl_ctx = None

        remote_server = RemoteConsoleServer(
            token=cfg.remote_token,
            host=cfg.remote_bind,
            port=cfg.remote_port,
            on_control=_remote_control,
            on_presence=_presence.handle_event,
            ssl_context=ssl_ctx,
        )
        ui.set_remote(remote_server)
        remote_server.start()
        return remote_server
    except Exception as exc:  # noqa: BLE001 — optional; must not block startup
        print(f"[remote] failed to start (continuing without it): {exc}",
              file=sys.stderr)
        return None


def register_status(
    cfg: Config, security_watcher, sound_detector, homelab_monitor,
    plex_client, plex_laptop_client, remote_server, calendar_monitor,
    weather_monitor,
) -> None:
    """M60 — self-status registry. Each subsystem reports a one-line status
    via the `status_report` tool ("Jarvis, are you healthy?"). Order is
    display order in the report. Last-write-wins on a re-register. Cheap —
    the getters only run when status_report is called, not on every turn."""
    from src.self_status import (
        count_session_errors as _ss_errors,
        process_private_mb as _ss_mem,
        register as _ss_register,
    )

    def _status_security() -> str:
        if security_watcher.is_locked():
            return "Security: LOCKED (post-deterrent — awaiting passphrase)"
        if security_watcher.is_armed():
            return "Security: ARMED"
        return "Security: standing down"

    def _status_acoustic() -> str:
        return (f"Acoustic awareness: "
                f"{'active' if sound_detector.is_active() else 'off'}")

    def _status_homelab() -> str:
        return (f"Homelab monitor: "
                f"{'active' if homelab_monitor.is_active() else 'off'}")

    def _status_plex_mcp() -> str:
        if plex_client is None:
            return "Plex MCP: unavailable"
        try:
            n: object = len(plex_client.tool_names)
        except Exception:
            n = "?"
        return f"Plex MCP: connected ({n} tools)"

    def _status_plex_laptop() -> str:
        if plex_laptop_client is None:
            return "Plex laptop SSH: unconfigured"
        return f"Plex laptop SSH: configured ({cfg.plex_laptop_host})"

    def _status_remote() -> str:
        if remote_server is None:
            return "Remote console: off"
        tls = bool(cfg.tls_cert_file and cfg.tls_key_file)
        proto = "HTTPS/WSS" if tls else "HTTP/WS"
        return f"Remote console: listening on port {cfg.remote_port} ({proto})"

    def _status_stt() -> str:
        if cfg.stt_server_url:
            return (f"STT: GPU server ({cfg.stt_server_url}) — "
                    f"backend={cfg.stt_backend}")
        return f"STT: local CPU only — backend={cfg.stt_backend}"

    def _status_calendar() -> str:
        return (f"Calendar reminders: "
                f"{'active' if calendar_monitor.is_active() else 'off'}")

    def _status_weather() -> str:
        return (f"Weather alerts: "
                f"{'active' if weather_monitor.is_active() else 'off'}")

    def _status_reminders() -> str:
        from src.reminders import list_pending  # noqa: PLC0415 — lazy
        items = list_pending()
        n = len(items)
        # Any non-empty action is a scheduled composition (M59 briefing, M63
        # good_night, and whatever _COMPOSITION_ACTIONS grows to next) — count
        # them all, not just "briefing" (the old check drifted when M63 added
        # good_night as a second schedulable action).
        scheduled = sum(1 for r in items if r.get("action"))
        if scheduled:
            return f"Reminders: {n} pending ({scheduled} scheduled briefing/wrap-up(s))"
        return f"Reminders: {n} pending"

    def _status_memory() -> str:
        return f"Process memory: {_ss_mem():.0f} MB private bytes"

    def _status_errors() -> str:
        n, ctx = _ss_errors()
        return f"Log errors: {n} concerning lines ({ctx})"

    _ss_register("Security", _status_security)
    _ss_register("Acoustic awareness", _status_acoustic)
    _ss_register("Homelab monitor", _status_homelab)
    _ss_register("Plex MCP", _status_plex_mcp)
    _ss_register("Plex laptop", _status_plex_laptop)
    _ss_register("Remote console", _status_remote)
    _ss_register("STT backend", _status_stt)
    _ss_register("Reminders", _status_reminders)
    _ss_register("Calendar reminders", _status_calendar)
    _ss_register("Weather alerts", _status_weather)
    _ss_register("Process memory", _status_memory)
    _ss_register("Log errors", _status_errors)


def shutdown_subsystems(
    *, security_watcher, homelab_monitor, sound_detector, calendar_monitor,
    weather_monitor, anticipation_engine,
    announcer: Announcer, reminder_stop: threading.Event,
    worker: threading.Thread, plex_client, plex_laptop_client,
) -> None:
    """Wind down all background subsystems in order, join the listen loop (so
    it seals the active session to disk), then close the Plex connections.
    Does NOT handle relaunch — that's control flow (sys.exit) and stays in
    main(). Every daemon thread dies with the process regardless; the explicit
    shutdowns let in-flight work finish cleanly without log noise."""
    # M34: signal the security watcher to wind down.
    security_watcher.shutdown()
    # M56: stop the homelab monitor's poll loop.
    homelab_monitor.shutdown()
    # M58: stop acoustic awareness (closes the InputStream + inference loop).
    sound_detector.shutdown()
    # M62.2: stop the calendar reminder monitor.
    calendar_monitor.shutdown()
    # M83: stop the anticipation synthesis loop.
    anticipation_engine.shutdown()
    # M77: stop the severe-weather alerts monitor.
    weather_monitor.shutdown()
    # M34: stop the Announcer thread (sets stop + wakes its blocked get()).
    announcer.shutdown()
    # M53: stop the reminder scheduler.
    reminder_stop.set()

    # Give listen_loop time to see the shutdown event, exit its loop cleanly,
    # and run its try/finally — which seals the active session to disk. Worst
    # case: user quit mid-recording and we wait up to ~15s for max-recording
    # to time out, then a summarize call. M16 hardening: bumped from 20s → 30s
    # so the summarize round-trip (capped at 8s in memory.py, no SDK retries)
    # plus any tail TTS playback comfortably fits before we give up.
    worker.join(timeout=30.0)
    if worker.is_alive():
        print("[main] listen_loop didn't exit in time — session may not be sealed", file=sys.stderr)

    # M21: clean up MCP subprocess. Done after worker.join so any in-flight
    # tool call inside listen_loop has a chance to complete first.
    if plex_client is not None:
        try:
            plex_client.close()
        except Exception as exc:
            print(f"[plex-mcp] close failed: {exc}", file=sys.stderr)

    # M24: tear down the persistent SSH connection. Idempotent + non-blocking;
    # paramiko handles the channel cleanup internally.
    if plex_laptop_client is not None:
        try:
            plex_laptop_client.close()
        except Exception as exc:
            print(f"[plex-laptop] close failed: {exc}", file=sys.stderr)
