"""Jarvis — top-level entry point.

Threading layout:
- Main thread:    Tk's mainloop (JarvisConsole window). Tk strongly prefers
                  the main thread, so this is its home.
- Worker thread:  pystray icon loop (started by JarvisUI). Win32 lets pystray
                  run on any thread; we put it here so Tk can have main.
- Worker thread:  listen_loop (voice path). Wakes on "Hey Jarvis", transcribes,
                  then calls process_question.
- Worker thread:  text_input_loop (M15). Pops typed messages from a queue and
                  calls process_question. Both paths share state (history,
                  memory) and serialize via processing_lock so voice and text
                  can't overlap.

All UI surfaces (Tk console, pystray) are thread-safe — console via .after(),
pystray via attribute writes.

Conversation memory: history is alternating user/assistant messages. Trimmed
in pairs to MAX_PAIRS most recent exchanges. Reset paths:
(a) tray menu "Reset conversation" — fires reset_event, applied at the next
    question (voice or text)
(b) idle timeout — IDLE_RESET_SEC since last completed turn → forget
(c) app restart — history is in-process only, never persisted

Console hiding: when launched via pythonw.exe (jarvis.pyw), there's no console.
setup_logging() redirects stdout/stderr to %LOCALAPPDATA%\\Jarvis\\jarvis.log so
we don't lose debug output. Console mode (python main.py) is unchanged.

Echo handling: while TTS plays, the mic still captures it. session.drain() at
the top of each voice iteration discards that buffered echo so the wake-word
detector starts each turn on fresh, live audio.
"""

from __future__ import annotations

import os
import sys
import threading

from src import speaker_id
from src.audio import resolve_input_device
from src.bootstrap import (
    build_announcer,
    build_remote_console,
    register_status,
    shutdown_subsystems,
    try_connect_plex,
    try_connect_plex_laptop,
)
from src.config import load
from src.gates import CountedEvent
from src.listen_loop import listen_loop
from src.logging_setup import setup_logging
from src.memory import default_base_dir
from src.ui import JarvisUI


def main() -> None:
    log_path = setup_logging()

    cfg = load()
    if not cfg.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY missing. Add it to .env and try again.", file=sys.stderr)
        sys.exit(1)

    # M21: spin up Plex MCP before the UI. Synchronous; takes a few seconds
    # the first time as plex-mcp-server initializes its plexapi connection.
    # Failure is non-fatal — we simply run without Plex tools.
    plex_client = try_connect_plex(cfg)

    # M24: prepare the SSH client to the Plex laptop. This is lazy — no
    # network until the first tool call — so it adds zero startup latency
    # even when the laptop is off.
    plex_laptop_client = try_connect_plex_laptop(cfg)

    # M41: log elevation status to jarvis.log on every startup. Makes it
    # trivial to grep "was this session elevated?" without digging through
    # tool-call traces. The mutating verbs in system_control (M40 —
    # flush_dns, restart_service) only work when this is True.
    from src.autostart import is_admin as _is_admin  # noqa: PLC0415
    print(f"[main] running elevated: {_is_admin()}")

    print("Jarvis ready. Tray icon active — left-click to show window. Say 'Hey Jarvis' to begin.\n")

    reset_event = threading.Event()

    def _on_reset() -> None:
        # Fires on pystray's menu thread. Just signals — actual clear happens
        # in listen_loop after the next wake word + STT, so the click applies
        # to the very next question (not the one after).
        reset_event.set()
        print("[tray] reset queued — applies to your next 'Hey Jarvis'")

    # Memory dir = %LOCALAPPDATA%\Jarvis (parent of sessions/ and summaries.jsonl)
    # — exposed on the tray so the user can browse past transcripts in one click.
    ui = JarvisUI(log_path=log_path, memory_dir=default_base_dir())
    ui.set_on_reset(_on_reset)

    # Status bar bootstrap. Model + integration dots are set once at startup;
    # token counter accumulates as turns happen. Plex/laptop status reflects
    # whether the optional integrations were configured + connected — not
    # whether the underlying network is currently reachable (lazy connect on
    # the laptop side means we don't probe until first use).
    ui.set_model_name(cfg.claude_model)
    ui.set_integration("plex", plex_client is not None)
    ui.set_integration("laptop", plex_laptop_client is not None)

    # SecurityWatcher is constructed below; import it here (lazy — heavy
    # transitive deps via ultralytics/cv2).
    from src.security import SecurityWatcher

    # "PC is speaking out loud" gate (2026-05-29 omni-mic echo fix). SET while
    # ANY PC TTS plays — proactive announces (Announcer) AND turn replies
    # (TurnRunner) — so the always-on voice-capture loop discards self-audio
    # instead of transcribing Jarvis's own reply. Distinct from
    # `announce_speaking` (which gates security/sound CPU bursts): this gates
    # mic capture, and covers turn replies too, not just announces.
    # 2026-07-02 QA: CountedEvent so an announce overlapping a turn reply
    # can't drop the gate early — see src/gates.py.
    pc_speaking = CountedEvent()

    # Proactive-speech subsystem: a dedicated WASAPI-safe Announcer thread.
    # build_announcer returns the non-blocking announce() entry every
    # proactive caller uses, the cooperative speech-gate Event (the
    # SecurityWatcher + SoundDetector defer heavy CPU bursts against it), and
    # a shutdown() callable for the wind-down. See build_announcer.
    _ann = build_announcer(ui, pc_speaking)
    _announce = _ann.announce
    announce_speaking = _ann.speaking_event

    # M53: reminders & timers. The scheduler thread polls reminders.json and
    # fires due reminders through _announce — a reminder is a proactive
    # announce, so it rides the same WASAPI-safe Announcer path as a security
    # alert. Bound to the ⏰ label so reminders read as reminders in the
    # console, not 🚨 alerts. Daemon thread; reminder_stop ends it cleanly at
    # shutdown. The local import matches the SecurityWatcher pattern above.
    # A reminder also pushes to Discord on fire (2026-06-02) so it reaches the
    # user when away from the PC — the dual-channel pattern the security /
    # homelab alerts already use. Gated on BOTH a configured webhook AND the
    # opt-out flag; None when either is off, so the scheduler simply skips the
    # remote sink (no wasted POST per fire). No presence detection, so it
    # always pushes — redundant-but-harmless when home.
    from src.reminders import run_scheduler as _run_reminder_scheduler
    from src.notifications import send_discord_message as _send_discord_message
    _reminder_webhook = (
        cfg.discord_webhook_url if cfg.reminder_discord_enabled else ""
    )
    _reminder_notify = (
        (lambda t: _send_discord_message(_reminder_webhook, t))
        if _reminder_webhook else None
    )
    if _reminder_notify is not None:
        print("[reminders] Discord push on fire: enabled", file=sys.stderr)
    reminder_stop = threading.Event()
    threading.Thread(
        target=_run_reminder_scheduler,
        args=(lambda t: _announce(t, label="⏰"), reminder_stop),
        kwargs={"notify": _reminder_notify},
        name="ReminderScheduler",
        daemon=True,
    ).start()

    # M35: pass the configured passphrase (empty → CHALLENGE step skipped,
    # M34 announce-only behavior) and the evidence directory (where
    # deterrent-fired triggering frames get saved as JPEGs).
    # M38: Discord webhook URL + SMTP credentials. Both channels fire in
    # parallel on deterrent; either being blank simply disables that channel.
    # Reuse cameras.py's CAMERA_INDEX env reader so the watcher honors the
    # same override as camera_snapshot — otherwise users with CAMERA_INDEX
    # set get the right webcam for vision queries but the wrong one for
    # security mode.
    from src.cameras import _camera_index as _resolve_camera_index

    security_watcher = SecurityWatcher(
        announce=_announce,
        on_armed_changed=ui.set_armed_indicator,
        on_locked_changed=ui.set_locked_indicator,
        camera_index=_resolve_camera_index(),
        passphrase=cfg.security_passphrase,
        evidence_dir=default_base_dir() / "security" / "events",
        discord_webhook_url=cfg.discord_webhook_url,
        smtp_host=cfg.smtp_host,
        smtp_port=cfg.smtp_port,
        smtp_username=cfg.smtp_username,
        smtp_password=cfg.smtp_password,
        smtp_to=cfg.smtp_to,
        # M39: face-recognition auth path. Encoding (if enrolled) lives
        # alongside the deterrent evidence at %LOCALAPPDATA%/Jarvis/security/.
        # SecurityWatcher loads it lazily on each activate() so re-enrollment
        # mid-session takes effect without a restart.
        face_encoding_path=default_base_dir() / "security" / "face_encoding.npy",
        face_match_threshold=cfg.face_match_threshold,
        # Cooperative speech gate (2026-05-19): the watcher defers heavy
        # vision work for any tick this is set, so a YOLO/encode/warm burst
        # can't starve the Python-fed TTS path and stutter an announce.
        speaking_event=announce_speaking,
    )

    # M71: when armed, the watcher owns the webcam (persistent capture). Let a
    # camera_snapshot (Discord, while away) borrow a frame from it instead of
    # opening a contending handle that would only grab black. Disarmed ⇒ the
    # provider returns None ⇒ cameras.py opens its own (the camera is free).
    from src import cameras as _cameras  # noqa: PLC0415
    _cameras.set_armed_frame_provider(security_watcher.grab_frame_for_snapshot)

    # M39: face-enrollment trigger. Shared by the tray menu "Enroll my face"
    # AND the voice intent ("Jarvis, enroll my face"). Non-blocking — queues
    # the announce prompt and returns; capture + enrollment run later on the
    # Announcer thread via on_done. Both triggers reach the same flow.
    def _trigger_face_enrollment() -> None:
        from src import face_auth as _face_auth  # noqa: PLC0415 — lazy
        from src import cameras as _cameras  # noqa: PLC0415
        _face_auth.run_voice_enrollment(
            _announce,
            _cameras.capture_frames,
            default_base_dir() / "security" / "face_encoding.npy",
        )

    ui.set_on_enroll_face(_trigger_face_enrollment)

    # M69: voice-enrollment trigger (speaker ID). Mirrors face enrollment —
    # shared by the tray "Enroll my voice" and the voice intent "Jarvis,
    # enroll my voice". Records a few CONSECUTIVE clips from the pinned mic
    # (the user talks continuously) which enroll_from_audio averages into one
    # embedding. The capture opens its own short-lived recording stream — a
    # second stream on the mic device, the same coexistence M58 relies on —
    # so it doesn't race the listen loop's AudioSession.
    def _capture_voice_clips(num_clips: int, clip_seconds: float):
        import sounddevice as _sd  # noqa: PLC0415 — lazy
        clips = []
        try:
            for _ in range(num_clips):
                rec = _sd.rec(int(clip_seconds * cfg.sample_rate),
                              samplerate=cfg.sample_rate, channels=1,
                              dtype="int16", device=mic_device_index)
                _sd.wait()
                clips.append(rec[:, 0].copy())
        except Exception as exc:  # noqa: BLE001 — defensive against mic errors
            print(f"[speaker_id] enrollment capture failed: {exc}", file=sys.stderr)
            return None
        return clips

    def _trigger_voice_enrollment(name: str | None = None, lang: str = "en") -> None:
        # name=None ⇒ the primary user (the no-arg voice intent / tray item);
        # a name ⇒ a named household member (the typed "enroll Alice's voice").
        speaker_id.run_voice_enrollment(
            _announce,
            _capture_voice_clips,
            default_base_dir() / "speakers",
            name=name or cfg.user_name,
            lang=lang,
        )

    ui.set_on_enroll_voice(_trigger_voice_enrollment)

    # M69 Phase 4: the opt-in "voice lock" gate as a runtime-toggleable Event
    # (tray + JARVIS_SPEAKER_GATE). SET = only enrolled voices are answered; a
    # confidently-unrecognized VOICE turn is dropped (the text/console path is
    # never gated — physical access implies authorization). The tray toggle
    # flips it live, no restart.
    #
    # Persisted across restarts (2026-06-02): a user who locks the mic expects
    # it to STAY locked next launch. Initial state = the persisted ui_state
    # flag if present, else the env default (JARVIS_SPEAKER_GATE) for a fresh
    # install; the tray toggle writes the flag so the choice is sticky.
    from src import ui_state  # noqa: PLC0415 — local import, main.py convention
    speaker_gate = threading.Event()
    if ui_state.get_flag("speaker_gate", cfg.speaker_gate_enabled):
        speaker_gate.set()
        print("[speaker] voice-lock gate restored ON", file=sys.stderr)

    def _toggle_speaker_gate() -> None:
        if speaker_gate.is_set():
            speaker_gate.clear()
            ui_state.set_flag("speaker_gate", False)
            print("[speaker] voice-lock gate OFF", file=sys.stderr)
        else:
            speaker_gate.set()
            ui_state.set_flag("speaker_gate", True)
            print("[speaker] voice-lock gate ON", file=sys.stderr)

    ui.set_on_speaker_gate_toggle(_toggle_speaker_gate, speaker_gate.is_set)

    # M45: knowledge-base triggers. Shared by the tray ("Reindex knowledge")
    # AND the voice intents ("Jarvis, update your knowledge" / "remember this
    # permanently: ..."). Non-blocking: the reindex / file-write is pure
    # disk+sqlite I/O and runs on a throwaway daemon thread, but the spoken
    # result MUST go through _announce (the WASAPI constraint — proactive
    # speech only via the Announcer thread, never speak_streaming from a
    # fresh thread). See project_wasapi_thread_audio_owner memory.
    def _trigger_knowledge_reindex() -> None:
        from src import knowledge as _knowledge  # noqa: PLC0415 — lazy

        def _work() -> None:
            try:
                result = _knowledge.reindex()
                _announce(result.message)
            except Exception as exc:  # defensive — never strand the trigger
                print(f"[knowledge] reindex trigger failed: {exc}", file=sys.stderr)
                _announce("I couldn't update my knowledge, sir.")

        threading.Thread(target=_work, daemon=True, name="kb-reindex").start()

    def _trigger_knowledge_remember(fact: str) -> None:
        from src import knowledge as _knowledge  # noqa: PLC0415 — lazy

        def _work() -> None:
            try:
                _announce(_knowledge.remember_fact(fact))
            except Exception as exc:
                print(f"[knowledge] remember trigger failed: {exc}", file=sys.stderr)
                _announce("I couldn't save that, sir.")

        threading.Thread(target=_work, daemon=True, name="kb-remember").start()

    ui.set_on_reindex_knowledge(_trigger_knowledge_reindex)
    # Wire the tray's Security-mode toggle to the watcher. Tray was already
    # constructed inside ui.run()'s worker thread, but the toggle's `checked`
    # callback re-evaluates each menu open, so this late wiring is fine.
    ui.set_on_security_toggle(
        on_activate=security_watcher.activate,
        on_deactivate=security_watcher.deactivate,
        is_armed=security_watcher.is_armed,
    )

    # M56: proactive homelab monitoring. Constructed ALWAYS (the
    # homelab_status tool + the tray toggle need the instance) but the
    # background poll loop only runs once activate()d. Default off — opt-in
    # via JARVIS_HOMELAB_MONITOR or the tray toggle, the same safe-default,
    # least-privilege stance as security mode. Reuses the existing
    # plex_laptop_client (its run() is lock-serialised, so the monitor thread
    # safely shares the one SSH connection the tool path uses) and the M38
    # Discord webhook for phone push. Spoken alerts ride _announce (the
    # WASAPI-safe Announcer path), tagged 🖥 so they read distinctly from
    # 🚨 security alerts and ⏰ reminders in the console.
    from src.homelab_monitor import HomelabMonitor
    homelab_monitor = HomelabMonitor(
        announce=lambda t: _announce(t, label="🖥"),
        plex_host=cfg.plex_laptop_host,
        plex_laptop_client=plex_laptop_client,
        discord_webhook_url=cfg.discord_webhook_url,
    )
    if cfg.homelab_monitor_enabled:
        homelab_monitor.activate(announce=False)  # silent at boot (status pills suffice)
    ui.set_on_homelab_toggle(
        on_activate=homelab_monitor.activate,
        on_deactivate=homelab_monitor.deactivate,
        is_active=homelab_monitor.is_active,
    )

    # M58: acoustic awareness. Constructed ALWAYS (the tray toggle needs the
    # instance) but the audio capture + inference loop only runs once
    # activate()d. Default off — opt-in via JARVIS_ACOUSTIC_MONITOR or the
    # tray toggle, same safe-default + least-privilege stance as security
    # mode and the homelab monitor. Owns its own sd.InputStream at 32 kHz
    # (the AudioSession is single-reader-by-design — wake-word loop owns it
    # during listening, M52 barge-in during TTS — so M58 captures
    # independently). Reuses the M38 Discord webhook; tagged 🔔 on the
    # Announcer so alerts read distinctly from 🚨 / 🖥 / ⏰.
    # Mic device pin (JARVIS_MIC_DEVICE). Resolve ONCE here so the main
    # capture and acoustic awareness bind the same physical input and we log
    # the chosen device a single time. None ⇒ Windows default (legacy).
    mic_device_index = resolve_input_device(cfg.mic_device)

    from src.sound_detector import SoundDetector
    # M72 — multimodal acoustic alert: when Jarvis hears something WHILE ARMED,
    # he also looks through the camera and pushes a photo + a one-line Claude
    # description to Discord. Gated by the armed frame provider —
    # grab_frame_for_snapshot returns None unless armed, so this no-ops at home
    # (no camera light for every doorbell) and only fires for the away case it's
    # built for. Runs on the SoundDetector's own AcousticVisual thread; every
    # step is fail-soft so a missing key / network blip / no camera just skips
    # the visual and the text alert (already sent) stands.
    def _acoustic_visual_alert(event_name: str) -> None:
        if not cfg.discord_webhook_url:
            return
        frame = security_watcher.grab_frame_for_snapshot()
        if frame is None:
            return  # not armed → the watcher isn't holding the camera; skip
        jpeg = _cameras.encode_jpeg(frame)
        if not jpeg:
            return
        from src.vision_describe import describe_scene  # noqa: PLC0415
        from src.notifications import send_discord_photo  # noqa: PLC0415
        desc = describe_scene(
            jpeg, event_name,
            api_key=cfg.anthropic_api_key, model=cfg.claude_model,
        )
        pretty = event_name.replace("_", " ")
        caption = f"👁 {desc}" if desc else f"👁 I heard {pretty} — here's the room, sir."
        send_discord_photo(cfg.discord_webhook_url, caption, jpeg,
                           image_filename="acoustic.jpg")

    sound_detector = SoundDetector(
        announce=lambda t: _announce(t, label="🔔"),
        discord_webhook_url=cfg.discord_webhook_url,
        device=mic_device_index,
        on_visual_alert=_acoustic_visual_alert,
        # M81 — armed intrusion-by-voice: the voice_while_armed rule only counts
        # windows while away. Same getter good_night + the visual hook use.
        is_armed=security_watcher.is_armed,
        # Cooperative speech gate (same Event the SecurityWatcher uses): the
        # PANNs inference loop defers while a proactive announce plays so it
        # can't starve the TTS path. M58 coupled acoustic to armed mode but
        # left this loop ungated — the 2026-05-28 armed-stutter regression.
        speaking_event=announce_speaking,
    )
    # M76 — expose the detector's recent-sounds buffer to the what_did_you_hear
    # tool (the self_status.register decoupling pattern).
    from src.sound_detector import register_detector as _register_sound_detector
    _register_sound_detector(sound_detector)
    if cfg.acoustic_monitor_enabled:
        sound_detector.activate()
    ui.set_on_acoustic_toggle(
        on_activate=sound_detector.activate,
        on_deactivate=sound_detector.deactivate,
        is_active=sound_detector.is_active,
    )

    # M62.2 — pre-event proactive calendar reminders. Watches the Outlook
    # calendar in the background and announces an event a fixed lead time
    # (default 15 min) before it starts: "Sir — your 2 pm standup in 15
    # minutes." The proactive layer on top of the M62 read tool — same
    # reactive→proactive jump M56 made for the homelab. Constructed ALWAYS
    # (so the status registration always has a target); activate() is a
    # logged no-op when calendar isn't configured or the env kill switch is
    # set. Unlike M56/M58 (default off, opt-in) this is DEFAULT ON when
    # calendar is configured — the user already opted in by setting up
    # OUTLOOK_ICAL_URL, the proactive layer IS the
    # value-add. Kill via JARVIS_CALENDAR_REMINDERS=0. Spoken alerts ride
    # _announce (the WASAPI-safe Announcer path), tagged 📅 so they read
    # distinctly from 🚨 / 🖥 / 🔔 / ⏰ in the console.
    from src.calendar_monitor import CalendarMonitor
    calendar_monitor = CalendarMonitor(
        announce=lambda t: _announce(t, label="📅"),
        discord_webhook_url=cfg.discord_webhook_url,
    )
    calendar_monitor.activate()  # internally a no-op when not configured / disabled

    # M77 — severe-weather proactive alerts. Watches the NWS active-alerts feed
    # for the home location and warns on its own before a power-threatening
    # storm (tied to the no-UPS power-loss situation). Default ON when
    # JARVIS_HOME_LOCATION is set; no-op otherwise. Tagged ⛈.
    from src.weather_alerts import WeatherAlertMonitor
    weather_monitor = WeatherAlertMonitor(
        announce=lambda t: _announce(t, label="⛈"),
        discord_webhook_url=cfg.discord_webhook_url,
    )
    weather_monitor.activate()  # internally a no-op when not configured / disabled

    # M83 — anticipatory intelligence ("Sir, I've taken the liberty…"). The
    # SYNTHESIS layer over every single-signal monitor above: a background pass
    # fuses the live world-state (calendar + weather + alerts + reminders +
    # homelab + security) and asks Claude, as an extremely-selective chief-of-
    # staff, whether ONE cross-domain insight is worth surfacing. Tagged 🧠 so it
    # DEFERS in quiet hours (→ the morning catch-up). Default OFF — it makes a
    # recurring (small) LLM call, so it's opt-in via JARVIS_ANTICIPATION=1.
    from src.anticipation import AnticipationEngine, is_enabled as _anticip_enabled
    from src import briefing as _brief
    from src.weather_alerts import execute_weather_alerts_tool as _wx_alerts

    def _anticipation_snapshot() -> dict:
        """Compose the live world-state for the engine — each section fail-soft
        (a hiccup just drops that section). Reuses the briefing gatherers so the
        weather/reminders/calendar wording stays consistent across surfaces."""
        def _safe(fn):
            try:
                v = fn()
                return v.strip() if isinstance(v, str) else v
            except Exception as exc:  # noqa: BLE001 — a section must never break the tick
                print(f"[anticipation] snapshot section failed: {exc}", file=sys.stderr)
                return None
        snap: dict = {
            "Calendar": _safe(_brief._calendar_section),
            "Weather": _safe(_brief._weather_section),
            "Reminders": _safe(_brief._reminders_section),
        }
        alerts = _safe(lambda: _wx_alerts({}))
        if alerts and "no active" not in alerts.lower() and "isn't configured" not in alerts.lower():
            snap["Weather alerts"] = alerts
        if homelab_monitor.is_active():
            snap["Homelab"] = _safe(homelab_monitor.status_report)
        try:
            snap["Security"] = ("Armed — Saul is away from home"
                                if security_watcher.is_armed()
                                else "Disarmed — home")
        except Exception:  # noqa: BLE001
            pass
        return snap

    anticipation_engine = AnticipationEngine(
        announce=lambda t: _announce(t, label="🧠"),
        snapshot_fn=_anticipation_snapshot,
        api_key=cfg.anthropic_api_key,
        model=cfg.claude_model,
    )
    if _anticip_enabled():
        anticipation_engine.activate()
    from src.self_status import register as _ss_anticip_reg
    _ss_anticip_reg("Anticipation",
                    lambda: "active" if anticipation_engine.is_active() else "off")

    # M63 — "good night" wrap. Wire the security-state getter so the
    # composition tool can report current armed/standing-down state without
    # needing security_watcher threaded through stream_response. Same
    # decoupling pattern as homelab_monitor's _set_active_monitor — the tool
    # finds the live state through a module-level singleton.
    from src.good_night import register_security_getter as _register_gn_security
    _register_gn_security(security_watcher.is_armed)

    # M64 — self-update tool. Wire the restart trigger so a successful
    # `update_jarvis` (after the confirmation gate) can drive the standard
    # M32 restart path. `ui._handle_restart` sets relaunch_mode + signals
    # shutdown — the same thing the tray's "Restart Jarvis" click does.
    # Same decoupling pattern as the security-getter above.
    from src.self_update import register_restart_callback as _register_su_restart
    _register_su_restart(ui._handle_restart)

    # M58 follow-up — couple acoustic awareness to security mode. The user's
    # mental model is "armed = away from home, where Jarvis should also be
    # listening for non-speech events"; this chains sound_detector.activate
    # / deactivate onto the security-armed edge. The independent tray toggle
    # for acoustic still works for "I want it on while I'm home" cases — the
    # coupling is one-way (security state drives acoustic, never the
    # reverse). Activate/deactivate are idempotent, so a manual flip
    # followed by a coupled flip never double-fires.
    def _security_armed_changed(armed: bool) -> None:
        ui.set_armed_indicator(armed)
        try:
            if armed:
                sound_detector.activate()
            else:
                sound_detector.deactivate()
        except Exception as exc:
            print(f"[main] coupling acoustic to security failed: {exc}",
                  file=sys.stderr)
    security_watcher.set_on_armed_changed(_security_armed_changed)

    # M48.1: LAN remote console (token-gated, optional). Returns the started
    # server or None. on_text is wired later in listen_loop once the
    # text_queue exists (the late-injection pattern).
    remote_server = build_remote_console(cfg, ui, security_watcher, _announce)

    # M60 — self-status registry: each subsystem reports a one-line status via
    # the `status_report` tool. Registered with the live subsystem handles.
    # M91 — long-horizon background tasks. Constructed always so status_report
    # can say "off"; the poll thread only starts when JARVIS_BACKGROUND_AGENTS
    # is set. start() also RE-ATTACHES to any session still running from before
    # a restart — the watchdog respawning us or update_jarvis restarting us is
    # routine, and those sessions kept working server-side meanwhile.
    # Completions ride the same _announce path as every other proactive
    # subsystem; one that lands during quiet hours is held back and surfaces in
    # the morning briefing instead. Local import matches the pattern above.
    from src.background_tasks import BackgroundTaskManager  # noqa: PLC0415
    from src import self_status  # noqa: PLC0415

    background_tasks_mgr = BackgroundTaskManager(cfg.anthropic_api_key, _announce, ui=ui)
    background_tasks_mgr.start()
    self_status.register("background tasks", background_tasks_mgr.status_summary)

    register_status(
        cfg, security_watcher, sound_detector, homelab_monitor,
        plex_client, plex_laptop_client, remote_server, calendar_monitor,
        weather_monitor,
    )

    # M69 — if any voice is enrolled, warm the speaker-ID model in the
    # background now so the first identified turn doesn't eat the ~5-15s
    # first-call JIT (mirrors face_auth's cold-start warming). No enrolled
    # voice ⇒ skip entirely (no torch load, no cost).
    if speaker_id.load_registry(default_base_dir() / "speakers"):
        threading.Thread(target=speaker_id.warm, name="SpeakerWarm",
                         daemon=True).start()

    worker = threading.Thread(
        target=listen_loop,
        args=(cfg, ui, reset_event, plex_client, plex_laptop_client, security_watcher),
        kwargs={
            "on_enroll_face": _trigger_face_enrollment,
            "on_enroll_voice": _trigger_voice_enrollment,
            "on_knowledge_reindex": _trigger_knowledge_reindex,
            "on_knowledge_remember": _trigger_knowledge_remember,
            "remote_server": remote_server,
            "mic_device": mic_device_index,
            "pc_speaking": pc_speaking,
            "announce_speaking": announce_speaking,
            "speaker_gate": speaker_gate,
        },
        daemon=True,
    )
    worker.start()

    ui.run()  # blocks main thread on Tk's mainloop until Quit is clicked

    # Wind down all background subsystems, join the listen loop (so it seals
    # the active session to disk), and close the Plex connections. Relaunch is
    # handled below — it's control flow (sys.exit), not subsystem teardown.
    shutdown_subsystems(
        security_watcher=security_watcher,
        homelab_monitor=homelab_monitor,
        sound_detector=sound_detector,
        calendar_monitor=calendar_monitor,
        weather_monitor=weather_monitor,
        anticipation_engine=anticipation_engine,
        announcer=_ann,
        reminder_stop=reminder_stop,
        worker=worker,
        plex_client=plex_client,
        plex_laptop_client=plex_laptop_client,
    )
    # Stops polling only. Running sessions are left alone on purpose:
    # they live server-side and the next start() re-attaches.
    background_tasks_mgr.shutdown()

    # Restart-Jarvis tray click sets ui.relaunch_mode. We defer the actual
    # respawn until here so the mic + Plex MCP + SSH have all released
    # their handles — the new instance starts on a clean slate.
    #
    # M41: tristate dispatch. "normal" uses subprocess.Popen (silent,
    # always works). "elevated" uses ShellExecuteW(verb="runas") which
    # fires UAC — if the user clicks No, no new instance starts and
    # this process still exits (acceptable "occasional sudo" UX).
    mode = ui.relaunch_mode
    if mode is not None:
        # M65 — watchdog cooperation. When `jarvis_watchdog.pyw` is the
        # parent process, it sets `JARVIS_WATCHDOG=1` in our env; on a
        # NORMAL restart we exit with code 42 ("please respawn me") and
        # let the watchdog handle the spawn. This keeps the watchdog as
        # the sole spawner of main.py — if we ALSO spawned, two instances
        # would race for the audio device. Elevated restart goes through
        # ShellExecuteW("runas") regardless, since the watchdog can't
        # spawn an elevated child; the elevated instance becomes
        # un-watched (documented limitation, acceptable for "occasional
        # sudo" UX). See [[project-m65-crash-watchdog]].
        under_watchdog = os.environ.get("JARVIS_WATCHDOG", "").strip() == "1"
        try:
            from src import autostart
            if mode == "elevated":
                ok = autostart.relaunch_elevated()
                if ok:
                    print("[main] relaunched elevated Jarvis instance")
                else:
                    print(
                        "[main] elevated relaunch was cancelled or failed — "
                        "Jarvis will not restart",
                        file=sys.stderr,
                    )
                # Either way the watchdog should NOT respawn — the
                # elevated child takes over (success) or the user
                # cancelled (no relaunch wanted). Exit 0.
            elif under_watchdog:
                # The watchdog parent will see exit code 42 and respawn.
                print("[main] under watchdog — exiting 42 for respawn")
                sys.exit(42)
            else:  # "normal", no watchdog
                autostart.relaunch()
                print("[main] relaunched detached Jarvis instance")
        except SystemExit:
            raise  # the sys.exit(42) path above; don't swallow it
        except Exception as exc:
            print(f"[main] relaunch failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
