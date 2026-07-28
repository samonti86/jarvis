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
import queue
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable

from src.audio import AudioSession, resolve_input_device
from src.bootstrap import (
    build_announcer,
    build_remote_console,
    register_status,
    shutdown_subsystems,
    try_connect_plex,
    try_connect_plex_laptop,
)
from src.gates import CountedEvent
from src.config import Config, load
from src.logging_setup import setup_logging
from src.memory import default_base_dir
from src.plex_laptop import PlexLaptopClient
from src.plex_mcp import PlexMCPClient
from src import speaker_id
from src.speech_to_text import transcribe_after_wake
from src.tray import State
from src.turn_runner import TurnRunner
from src.ui import JarvisUI
from src.wake_word import wait_for_wake_word


# M51 — conversational follow-up window. After Jarvis answers, the listen
# loop stays open for this many seconds and accepts a follow-up WITHOUT a
# fresh "Hey Jarvis"; silence past the window falls back to wake-word mode.
# Long enough to gather a follow-up thought, short enough to bound the
# no-wake-word false-capture window (ambient speech / TV).
_FOLLOWUP_WINDOW_SEC = 12.0

# M87 — interpreter mode. While interpreting, the listen loop waits this long
# for the next person to start speaking before re-arming (no wake word ever).
# Longer than the follow-up window: two people taking turns through an
# interpreter pause naturally, and on elapse we simply re-listen (still in
# interpreter mode), so this is just the silence cadence, not a timeout.
_INTERPRETER_WINDOW_SEC = 30.0

# M88 — conversation mode (full-duplex, Phase 1). While in conversation mode the
# loop listens hands-free with this pre-speech window; on silence it re-arms
# (stays in the mode) rather than dropping to the wake word — UNTIL this many
# consecutive empty windows elapse, at which point it auto-exits to standby (so
# Jarvis isn't left listening to an empty room indefinitely). 25s × 3 ≈ 75s of
# silence before auto-exit.
_CONVERSATION_WINDOW_SEC = 25.0
_CONVERSATION_IDLE_EXITS = 3

# M51 — conversation sign-offs. When the user's turn ENDS with one of these,
# the follow-up window is NOT opened — an explicit "that's all" should close
# the conversation cleanly, not leave the mic listening for 12s. Matched by
# suffix on a normalized transcript: a genuine sign-off is the tail of what
# the user says, so "thanks Jarvis, that is all" matches but "that's all I
# need — what about the Jets?" does not.
_DISMISSAL_PHRASES = frozenset({
    "that is all", "thats all", "that is all for now", "thats all for now",
    "that will be all", "thatll be all", "that is it", "thats it",
    "that is everything", "thats everything", "nothing else", "nothing more",
    "no thank you", "no thanks", "im done", "im good", "im all set",
    "we are done", "were done", "all done", "thank you jarvis",
    "thanks jarvis", "goodbye",
    # 2026-07-02 QA: "good night"/"goodnight" REMOVED — as dismissals they
    # short-circuited before the LLM on any follow-up/conversation-mode turn,
    # shadowing the M63 get_good_night wrap (security state + tomorrow's
    # schedule/weather). "Good night" now reaches Claude and routes to the
    # wrap; the reply's follow-up window then just elapses to standby.
})

# M51 follow-on (2026-05-21): a trailing courtesy masks a sign-off. "That is
# all, thank you" ends with "thank you", not "that is all", so the bare suffix
# match missed it — the window stayed open and the user had to repeat "that is
# all". We strip ONE trailing courtesy and re-test, rather than registering
# bare "thank you" as a dismissal phrase: that would false-trip questions like
# "how do you say thank you". Longest-first so the whole phrase is removed.
_TRAILING_COURTESY = (
    "thank you very much", "thank you so much", "thanks so much",
    "thank you", "thanks", "please",
)


def _is_dismissal(text: str) -> bool:
    """True if the user's utterance is a conversation sign-off — M51 uses this
    to skip opening a follow-up window after it. Suffix match on a normalized
    transcript (lowercased, punctuation/apostrophes stripped), with one
    trailing courtesy ("...thank you") stripped before re-testing."""
    norm = " ".join(re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).split())
    if not norm:
        return False

    def _suffix_match(s: str) -> bool:
        return bool(s) and any(
            s == p or s.endswith(" " + p) for p in _DISMISSAL_PHRASES
        )

    if _suffix_match(norm):
        return True
    # Retry with a trailing courtesy removed — "that is all, thank you". A
    # bare "thank you" (norm == courtesy) strips to "" → _suffix_match False,
    # so plain politeness still does NOT close the conversation (M51's
    # deliberate choice: only "thank you JARVIS" — with the name — signs off).
    for courtesy in _TRAILING_COURTESY:
        if norm == courtesy or norm.endswith(" " + courtesy):
            return _suffix_match(norm[: -len(courtesy)].strip())
    return False


def listen_loop(
    cfg: Config,
    ui: JarvisUI,
    reset_event: threading.Event,
    plex_client: PlexMCPClient | None = None,
    plex_laptop_client: PlexLaptopClient | None = None,
    security_watcher: "SecurityWatcher | None" = None,
    on_enroll_face: "Callable[[], None] | None" = None,
    on_enroll_voice: "Callable[[], None] | None" = None,
    on_knowledge_reindex: "Callable[[], None] | None" = None,
    on_knowledge_remember: "Callable[[str], None] | None" = None,
    remote_server: object | None = None,
    mic_device: int | None = None,
    pc_speaking: threading.Event | None = None,
    announce_speaking: threading.Event | None = None,
    speaker_gate: threading.Event | None = None,
) -> None:
    """Daemon worker. Owns the input loops; delegates each turn to a shared
    TurnRunner (which owns the conversation state, persists turns, and seals
    sessions on every memory boundary — manual reset / idle / app quit).

    Two input paths share ONE TurnRunner:
      - voice path: this function's main loop (wake-word → STT → runner.process_question)
      - text path: text_input_loop spawned below (queue → runner.process_question)

    The runner's lock serializes the two, so a typed message that lands while
    Jarvis is answering a spoken one waits its turn — no overlapping responses.
    """
    # The conversation engine: owns history/summaries/session state + the
    # MemoryStore and runs each turn end-to-end. Voice + text share this ONE
    # instance (its lock keeps them from overlapping). See the TurnRunner class.
    runner = TurnRunner(cfg, ui, reset_event, plex_client, plex_laptop_client,
                        pc_speaking, announce_speaking, mic_device=mic_device)

    # M85 — tonal awareness. One analyzer per session; it holds a rolling
    # loudness baseline so "softer/louder than usual" is judged against the
    # user's own voice. Only the voice path feeds it (it needs the audio clip).
    from src.voice_tone import ToneAnalyzer, is_enabled as _tone_enabled
    tone_analyzer = ToneAnalyzer() if _tone_enabled() else None

    # Text-submission queue. Tk's submit handler puts (text, attachments)
    # tuples here; the text_input_loop worker pops them and calls
    # process_question. attachments is list[tuple[str, dict]] (filename + block).
    # M48.2/M48.2b/M48.3: item is
    #   (text, attachments, origin, reply_audio, lang, reply_text, reply_image).
    # origin ∈ {"console","phone_text","phone_voice","discord"} — derives
    # PC-TTS gating (phone_*/discord don't speak on the PC) AND restricted tool
    # surface (phone + discord origins lose system/shell/file/etc.).
    # reply_audio: None (console — PC behaviour) or a conn-bound sink (phone)
    # — its presence routes this reply's audio to THAT phone instead of PC.
    # lang: M48.3 — whisper-detected ISO-639-1 for phone_voice (so Spanish-
    # spoken into the phone gets a Spanish reply + a Spanish voice); "en"
    # for typed inputs where we have no detection. Voice path on the PC
    # mic calls process_question directly (not via this queue), unaffected.
    # reply_text (2026-06-02): None, or a PER-TURN text sink (Discord) that
    # posts the reply back to the originating channel — NOT a broadcast, so a
    # PC/phone turn never leaks into the shared channel.
    # reply_image (M71): None, or a PER-TURN image sink (Discord only) that
    # buffers a webcam frame captured mid-turn so it's posted into the same
    # thread as the reply. Set ONLY for discord turns (camera_snapshot is
    # clawed back for origin="discord"); None everywhere else.
    text_queue: queue.Queue[
        tuple[
            str,
            list[tuple[str, dict]],
            str,
            "Callable[[bytes], None] | None",
            str,
            "Callable[[str], None] | None",
            "Callable[[bytes, str], None] | None",
        ]
    ] = queue.Queue()

    def text_input_loop() -> None:
        """Consume typed questions from the queue. Short timeout so we notice
        shutdown promptly without wasting cycles."""
        while not ui.shutdown.is_set():
            try:
                item = text_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if ui.shutdown.is_set():
                break
            text, attachments, origin, reply_audio, lang, reply_text, reply_image = item
            blocks = [block for _, block in attachments] if attachments else []
            print(
                f"[text-input] received: {text} (attachments={len(blocks)})",
                file=sys.stderr,
            )

            # Discord is a conversational-only remote surface: it must NOT fire
            # the local privileged shortcuts below (face/voice enrollment
            # captures the PC's OWN camera/mic — nonsensical and unsafe to
            # trigger from a chat message; the security + knowledge shortcuts
            # are likewise reserved for the physically-present / trusted
            # personal paths). A Discord turn goes straight to the LLM with the
            # restricted tool surface. Easy to relax later (e.g. deliberate
            # remote arming), but v1 keeps it strictly conversational.
            local_shortcuts = origin != "discord"

            # Same security-intent dispatch as the voice listen_loop:
            # "activate security", "stand down", and challenge-passphrase
            # responses are handled locally without calling Claude. Without
            # this, typed "activate security" gets routed to Claude, which
            # has no tool to arm the watcher and ends up calling
            # system_control to lock the workstation instead.
            if local_shortcuts and security_watcher is not None and not attachments:
                try:
                    if security_watcher.handle_transcript(text):
                        ui.add_user_text(text, lang)
                        ui.set_state(State.IDLE)
                        continue
                except Exception as exc:
                    print(f"[text-input] security.handle_transcript raised: {exc}",
                          file=sys.stderr)

            # M39: typed "enroll my face" — same dispatch as the voice path.
            # Uses the on_enroll_face kwarg (wired by main()) rather than a
            # bare name — _trigger_face_enrollment lives in main()'s scope,
            # not listen_loop's.
            if local_shortcuts and not attachments and on_enroll_face is not None:
                from src.face_auth import matches_enroll_intent  # noqa: PLC0415
                if matches_enroll_intent(text):
                    ui.add_user_text(text, lang)
                    on_enroll_face()
                    ui.set_state(State.IDLE)
                    continue

            # M69: typed voice-enrollment — the RELIABLE named path (typed
            # names beat Whisper's guesses). Named ("enroll Alice's voice",
            # "enroll voice Bob in Spanish") checked before the primary
            # ("enroll my voice"). This text path is never voice-gated.
            if local_shortcuts and not attachments and on_enroll_voice is not None:
                _named = speaker_id.parse_named_enroll_intent(text)
                if _named is not None:
                    ui.add_user_text(text, lang)
                    on_enroll_voice(_named[0], _named[1])
                    ui.set_state(State.IDLE)
                    continue
                if speaker_id.matches_enroll_intent(text):
                    ui.add_user_text(text, lang)
                    on_enroll_voice()
                    ui.set_state(State.IDLE)
                    continue

            # M45: typed knowledge-base intents — same dispatch as the voice
            # path (remember-permanently before reindex; both before Claude).
            if local_shortcuts and not attachments and (
                on_knowledge_remember is not None or on_knowledge_reindex is not None
            ):
                from src import knowledge as _kb  # noqa: PLC0415
                _fact = (_kb.extract_remember_fact(text)
                         if on_knowledge_remember is not None else None)
                if _fact is not None:
                    ui.add_user_text(text, lang)
                    on_knowledge_remember(_fact)
                    ui.set_state(State.IDLE)
                    continue
                if (on_knowledge_reindex is not None
                        and _kb.matches_reindex_intent(text)):
                    ui.add_user_text(text, lang)
                    on_knowledge_reindex()
                    ui.set_state(State.IDLE)
                    continue

            ui.set_state(State.THINKING)
            try:
                # M48.2/M48.2a: origin passed straight through — process_question
                # derives BOTH the text-only gate (phone_text doesn't speak on
                # the PC) AND the restricted tool surface (phone origins lose
                # system/shell/file/etc.) from it.
                # M48.3: language is "en" for typed input (no detection
                # available) but the whisper-detected ISO code for
                # phone_voice — so "¿qué hora es?" spoken into the phone
                # gets a Spanish reply with a Spanish voice. Claude still
                # follows the input's language via the system prompt; the
                # `language` arg drives TTS voice selection downstream.
                runner.process_question(
                    text, lang, attachments=blocks, origin=origin,
                    reply_audio=reply_audio, reply_text=reply_text,
                    reply_image=reply_image,
                )
            finally:
                ui.set_state(State.IDLE)

    # Wire the Tk submit handler. Putting on the queue is non-blocking, so
    # this returns immediately and Tk's mainloop stays responsive even while
    # an LLM stream + TTS playback is in progress. Lambda repackages the
    # (text, attachments) args into a single queue item.
    ui.set_on_text_submit(
        lambda text, attachments: text_queue.put(
            (text, attachments, "console", None, "en", None, None)  # console → no phone audio sink, no remote text/image sink; lang "en"
        )
    )

    # M48.1: now that text_queue exists, wire the remote console's converse
    # path to the SAME seam (the brain can't tell phone text from console
    # text — exactly the design intent). on_control was already wired in
    # main() where SecurityWatcher is in scope.
    if remote_server is not None:
        # M48.2b: the server passes a per-turn reply_audio sink bound to the
        # originating phone conn (or None if its "Speak replies" toggle is
        # off). reply_audio's presence IS the routing decision downstream:
        # non-None ⇒ this reply's audio goes to THAT phone, not the PC.
        remote_server.set_on_text(
            lambda t, reply_audio: text_queue.put(
                (t, [], "phone_text", reply_audio, "en", None, None)  # phone typed → "en"; text reply rides the broadcast sink; no image sink
            )
        )

        # M48.3 — phone push-to-talk. The phone records a blob (mp4 on
        # Safari, webm on others) and ships it base64-over-WS. The server
        # decodes the b64 and hands us (blob, mime, reply_audio); we must
        # transcribe off the WS thread (whisper can take seconds, must
        # never block the server loop) and then push the resulting text
        # onto the SAME text_queue the brain already drains — origin
        # "phone_voice" so the routing matrix flags it restricted (tool
        # boundary) AND silent on PC (reply_audio sinks the audio to the
        # phone instead). reply_audio is ALWAYS set for audio messages
        # (voice-in always returns voice-out — the locked M48.3 UX
        # decision; the "Speak replies" toggle continues to gate phone-
        # text only).
        def _on_phone_audio(blob: bytes, mime: str, reply_audio) -> None:
            def _worker() -> None:
                try:
                    from src.speech_to_text import transcribe_blob  # noqa: PLC0415
                    t = transcribe_blob(
                        blob, mime,
                        cfg.whisper_model, cfg.stt_server_url, cfg.stt_backend,
                    )
                    text = (t.text or "").strip()
                    if not text:
                        ui.add_system_text(
                            "(no speech detected in phone audio)"
                        )
                        return
                    # Whisper-detected language flows through to TTS voice
                    # selection — phone Spanish gets a Spanish voice reply.
                    lang = (t.language or "en").strip() or "en"
                    text_queue.put(
                        (text, [], "phone_voice", reply_audio, lang, None, None)
                    )
                except Exception as exc:  # noqa: BLE001 — never crash the WS loop's caller
                    print(
                        f"[phone-voice] transcription failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    ui.add_system_text(
                        f"phone audio error: {type(exc).__name__}"
                    )
            threading.Thread(
                target=_worker, name="PhoneAudioSTT", daemon=True
            ).start()
        remote_server.set_on_audio(_on_phone_audio)

    # Discord bot bridge (2026-06-02) — a private channel as a two-way Jarvis
    # client, riding the SAME text_queue as the phone. Messages from
    # allowlisted users in the configured channel enqueue with origin="discord"
    # (→ restricted tools + text-only); the per-turn `reply` sink posts the
    # answer back to THAT channel (never a broadcast — no leak of PC/phone
    # turns into the shared channel). Fail-closed + graceful: needs ALL THREE
    # of token/channel/allowlist, else it doesn't start. Outbound gateway
    # connection, so no inbound port / Tailscale needed — works off-network.
    if (cfg.discord_bot_token and cfg.discord_channel_id
            and cfg.discord_allowed_user_ids):
        try:
            from src.discord_bot import DiscordBot  # noqa: PLC0415
            _discord_bot = DiscordBot(
                cfg.discord_bot_token, cfg.discord_channel_id,
                cfg.discord_allowed_user_ids,
            )
            # M71: the bot now hands us TWO per-turn sinks — reply_text (post
            # the answer) and reply_image (buffer a webcam frame so it's
            # attached to that same threaded reply). reply_image is Discord-only
            # because camera_snapshot is clawed back for origin="discord" alone.
            _discord_bot.set_on_text(
                lambda t, reply_text, reply_image: text_queue.put(
                    (t, [], "discord", None, "en", reply_text, reply_image)
                )
            )
            _discord_bot.start()
            print("[discord] bot bridge enabled", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — a bad bot must never stop Jarvis
            print(f"[discord] bot failed to start: {exc}", file=sys.stderr)
    elif cfg.discord_bot_token:
        print("[discord] token set but channel/allowlist incomplete — bot "
              "disabled", file=sys.stderr)

    text_thread = threading.Thread(target=text_input_loop, daemon=True)
    text_thread.start()

    # 2026-07-02 QA: the voice loop's mic session lifetime, extracted so the
    # supervisor below can re-open it after a failure. Live incident: a
    # PortAudioError -9985 ("Device unavailable") on this open killed the
    # thread unhandled and Jarvis ran DEAF all day — every other subsystem
    # stayed up, and the watchdog only sees process exit, not thread death.
    # The acoustic stream opened the SAME device 35 minutes later, so a retry
    # recovers. Mode state (follow-up window, interpreter/conversation Events)
    # deliberately re-initializes on re-open — standby is the only honest
    # state after the mic has been away.
    def _voice_session_loop() -> None:
        nonlocal mic_failures
        with AudioSession(sample_rate=cfg.sample_rate, device=mic_device) as session:
            if mic_failures:
                print("[audio] microphone recovered — voice loop resuming",
                      file=sys.stderr)
                ui.add_system_text("🎙 Microphone recovered — listening again.")
                mic_failures = 0
            # M51: True ⇒ the previous turn just ended; listen for a follow-up
            # WITHOUT requiring "Hey Jarvis" (a longer pre-speech window;
            # silence past it falls back to wake-word mode). Starts False —
            # the first turn after startup always needs the wake word.
            followup = False
            # M52: True for the single iteration right after a barge-in. The
            # user typically says "Hey Jarvis" AND the new question in one
            # breath, so the buffered mic audio holds the start of that
            # question. We must NOT drain it (draining is correct after a
            # normal wake word, where the user pauses) — see below.
            barged = False
            # M87: interpreter mode. When set, the loop skips the wake word and
            # runs a continuous capture→translate→speak cycle (handled in the
            # dedicated branch at the top of the loop) until "stop interpreting".
            # A local Event — activation/exit both happen inside this loop, so
            # nothing outside needs a handle (a tray toggle is a clean follow-on).
            interpreter_mode = threading.Event()
            # M88: conversation mode — persistent hands-free Q&A (no wake word
            # between turns). Local Event like interpreter_mode; idle_empties
            # counts consecutive silent windows toward the auto-exit-to-standby.
            conversation_mode = threading.Event()
            idle_empties = 0
            # 2026-07-02 QA: consecutive interpreter STT failures — the
            # interpreter analog of conversation mode's idle_empties escape.
            interp_failures = 0
            while not ui.shutdown.is_set():
                # M52: skip the drain on the post-barge-in iteration so the
                # already-spoken follow-up survives into capture; otherwise
                # the user's words between "Hey Jarvis" and here are discarded
                # (the M52 "missed the start of my question" bug). The barge-in
                # monitor consumed the queue up to ~end of "Hey Jarvis", so
                # what remains is the question itself (plus a brief TTS-echo
                # tail Whisper tolerates). Every other iteration drains as
                # before — clearing the just-finished reply's echo.
                if barged:
                    barged = False
                else:
                    session.drain()

                # M51: a queued reset cancels any open follow-up window — drop
                # to the wake-word path below, which handles the reset. M88: a
                # reset also exits conversation mode (a clean-slate request means
                # back to standby).
                if (followup or conversation_mode.is_set()) and reset_event.is_set():
                    followup = False
                    if conversation_mode.is_set():
                        conversation_mode.clear()
                        print("[conversation] mode OFF (reset)", file=sys.stderr)

                # M87 — interpreter mode: a self-contained continuous loop that
                # supersedes the normal wake-word/follow-up path entirely. It
                # listens with NO wake word, translates whatever it hears, speaks
                # it in the other language's voice, and re-listens — until the
                # user says "stop interpreting". Kept as a separate branch (a
                # full capture cycle that `continue`s) so the normal hot path
                # below is untouched. Self-speech is already gated out of capture
                # by `pc_speaking` (set by interpret()), so the mic won't
                # transcribe Jarvis's own translation.
                if interpreter_mode.is_set():
                    followup = False  # interpreter supersedes the follow-up window
                    # 2026-07-02 QA: honour a tray "Reset conversation" while
                    # interpreting — this branch never consulted reset_event, so
                    # the click was silently inert until the mode was exited BY
                    # VOICE (impossible when STT itself is failing). Exit the
                    # mode and fall through; the wake-word path seals the reset.
                    if reset_event.is_set():
                        interpreter_mode.clear()
                        print("[interpreter] mode OFF (reset)", file=sys.stderr)
                        continue
                    ui.set_state(State.LISTENING)
                    try:
                        transcript = transcribe_after_wake(
                            session,
                            model_name=cfg.whisper_model,
                            on_speech_ended=lambda: ui.set_state(State.THINKING),
                            on_amplitude=ui.set_amplitude,
                            server_url=cfg.stt_server_url,
                            backend=cfg.stt_backend,
                            max_pre_speech_sec=_INTERPRETER_WINDOW_SEC,
                            suppress_event=pc_speaking,
                        )
                    except Exception as exc:
                        print(f"[main] interpreter STT failed: {exc}",
                              file=sys.stderr)
                        # 2026-07-02 QA: a PERSISTENT STT failure (mic yanked,
                        # backend down) must not spin this wake-word-less branch
                        # as a hot loop with no exit — the only voice escape
                        # ("stop interpreting") needs a SUCCESSFUL transcription.
                        # Same escape shape as conversation mode's idle counter.
                        interp_failures += 1
                        if interp_failures >= _CONVERSATION_IDLE_EXITS:
                            interpreter_mode.clear()
                            interp_failures = 0
                            print("[interpreter] repeated STT failure — mode OFF, "
                                  "back to wake word", file=sys.stderr)
                            ui.set_state(State.IDLE)
                        continue
                    interp_failures = 0
                    if ui.shutdown.is_set():
                        break
                    if not transcript.text:
                        # Window elapsed with no speech — stay in interpreter
                        # mode and keep listening (this is the silence cadence,
                        # not an exit).
                        ui.set_state(State.IDLE)
                        continue
                    from src import interpreter as _interp  # noqa: PLC0415
                    if _interp.is_stop_intent(transcript.text):
                        ui.add_user_text(transcript.text, transcript.language)
                        interpreter_mode.clear()
                        print("[interpreter] mode OFF (voice)", file=sys.stderr)
                        runner.speak_line(_interp.STOP_CONFIRM_EN, "en")
                        if "es" in _interp.LANG_PAIR:
                            runner.speak_line(_interp.STOP_CONFIRM_ES, "es")
                        ui.set_state(State.IDLE)
                        continue
                    runner.interpret(transcript.text, transcript.language)
                    ui.set_state(State.IDLE)
                    continue

                if followup or conversation_mode.is_set():
                    # Follow-up window (M51) OR conversation mode (M88): skip the
                    # wake word, capture (below) directly with a longer pre-speech
                    # timeout. The blue LISTENING pill is the visual cue; if the
                    # user says nothing, a follow-up window elapses to wake-word
                    # mode while conversation mode re-arms (until the idle cap).
                    ui.set_state(State.LISTENING)
                else:
                    ui.set_state(State.IDLE)
                    wait_for_wake_word(
                        session,
                        threshold=cfg.wake_word_threshold,
                        shutdown_event=ui.shutdown,
                        reset_event=reset_event,
                    )
                    if ui.shutdown.is_set():
                        break

                    # Reset clicked while we were idle-listening. Seal+clear
                    # now, without forcing another question first.
                    if reset_event.is_set():
                        if runner.has_active_conversation():
                            print("[main] conversation reset (manual; sealing active session)")
                            ui.add_system_text("conversation reset.")
                        else:
                            print("[main] reset clicked, but no active conversation to seal")
                        runner.seal_and_refresh()
                        reset_event.clear()
                        continue  # back to top — listen for next wake

                    ui.set_state(State.LISTENING)

                try:
                    transcript = transcribe_after_wake(
                        session,
                        model_name=cfg.whisper_model,
                        # Flip state the moment audio capture ends — don't let the
                        # LISTENING pill linger through Whisper's ~1-2s transcription.
                        on_speech_ended=lambda: ui.set_state(State.THINKING),
                        # M18: drive the waveform visualizer with mic amplitude
                        # while LISTENING. SPEAKING and LISTENING never overlap
                        # so the same callback safely serves both phases.
                        on_amplitude=ui.set_amplitude,
                        # M36: GPU offload (or local CPU if unset / forced).
                        # Auto-fallback by default.
                        server_url=cfg.stt_server_url,
                        backend=cfg.stt_backend,
                        # M51/M88: a follow-up turn waits longer for the user to
                        # start; conversation mode waits longer still (it re-arms
                        # on silence rather than exiting); the normal post-wake
                        # path uses the default.
                        max_pre_speech_sec=(
                            _CONVERSATION_WINDOW_SEC if conversation_mode.is_set()
                            else _FOLLOWUP_WINDOW_SEC if followup else None
                        ),
                        # 2026-05-29 omni-mic echo fix: abort + discard the
                        # capture if the PC starts speaking (a console-turn
                        # reply or a proactive announce) mid-window — so the
                        # no-wake-word follow-up window can't transcribe
                        # Jarvis's own voice as a question.
                        suppress_event=pc_speaking,
                    )
                except Exception as exc:
                    print(f"[main] STT failed: {exc}")
                    # M88: a PERSISTENT STT failure (GPU server down, mic device
                    # yanked, decode error) must not spin conversation mode
                    # forever — unlike the wake-word path it has no blocking
                    # fallback, so an exception that re-arms immediately is a hot
                    # loop. Count a failure like an empty window so the same idle
                    # auto-exit drops us back to standby; a transient blip just
                    # adds one (reset to 0 on the next real utterance).
                    if conversation_mode.is_set():
                        idle_empties += 1
                        if idle_empties >= _CONVERSATION_IDLE_EXITS:
                            print("[conversation] repeated STT failure — mode OFF, "
                                  "back to wake word\n", file=sys.stderr)
                            conversation_mode.clear()
                            idle_empties = 0
                            ui.set_state(State.IDLE)
                    followup = False
                    continue

                if not transcript.text:
                    # M88: in conversation mode, silence re-arms (stay hands-free)
                    # until enough consecutive empty windows accumulate, then
                    # auto-exit to standby so we're not listening to an empty
                    # room forever. Silent exit (no spoken note — the user has
                    # likely stepped away).
                    # 2026-07-02 QA: an empty transcript caused by the
                    # suppress_event abort (a proactive announce started mid-
                    # window) is NOT user idleness — don't let a chatty reminder
                    # schedule silently exit conversation mode while the user is
                    # present. The announce is still playing when we get here
                    # (the abort fires the instant it starts), so pc_speaking
                    # reliably tags the suppressed case.
                    if (conversation_mode.is_set() and pc_speaking is not None
                            and pc_speaking.is_set()):
                        continue
                    if conversation_mode.is_set():
                        idle_empties += 1
                        if idle_empties >= _CONVERSATION_IDLE_EXITS:
                            print("[conversation] idle — mode OFF, back to wake "
                                  "word\n", file=sys.stderr)
                            conversation_mode.clear()
                            followup = False
                            idle_empties = 0
                            ui.set_state(State.IDLE)
                        continue
                    # M51: no speech. In a follow-up window that just means
                    # the window elapsed — fall back to wake-word mode.
                    if followup:
                        print("[main] follow-up window elapsed — back to wake word\n",
                              file=sys.stderr)
                        followup = False
                    else:
                        print("[main] (no speech captured)\n")
                    continue

                # M69: identify the speaker on the SAME audio STT just captured.
                # Active only when at least one voice is enrolled. The per-turn
                # registry reload is sub-ms at personal scale, so a fresh
                # enrollment takes effect on the very next turn with no plumbing.
                # Phase 4: if the "voice lock" gate is ON and this is a
                # CONFIDENTLY unrecognized voice, drop the turn — don't route
                # background media / strangers to Claude. Fail-open by design:
                # the threshold sits below the user's rough-voice floor, so a
                # degraded real-user clip still 'recognizes' and never reaches
                # the drop. (The typed/console path is never gated — placed
                # BEFORE the security + enroll intents so a locked-out voice
                # can't drive them either.)
                # M80: the recognized speaker for THIS turn, threaded into the
                # LLM call so Jarvis addresses them by name + knows their usual
                # language. Resolved here (cfg is in scope): the primary enrolls
                # under the "you" sentinel, so map it to cfg.user_name for a
                # natural display name without re-enrolling the voice. None when
                # unrecognized or no registry → no speaker block downstream.
                speaker_name: "str | None" = None
                speaker_lang: "str | None" = None
                # M85: a short vocal-delivery cue (how he sounded — volume/pace/
                # pauses) from the SAME clip, threaded into the LLM call so Claude
                # can calibrate its tone. None on most turns (only notable
                # delivery produces a cue) and whenever tone awareness is off.
                vocal_cue: "str | None" = None
                if tone_analyzer is not None and transcript.audio is not None:
                    vocal_cue = tone_analyzer.analyze(transcript.audio, transcript.text)
                    if vocal_cue:
                        print(f"[tone] {vocal_cue}", file=sys.stderr)
                if transcript.audio is not None:
                    _speakers = speaker_id.load_registry(default_base_dir() / "speakers")
                    if _speakers:
                        _who = speaker_id.identify(
                            transcript.audio, _speakers, cfg.speaker_threshold)
                        if _who.recognized:
                            speaker_name = (cfg.user_name if _who.name == "you"
                                            else _who.name)
                            speaker_lang = _who.lang
                            print(f"[speaker] recognized {speaker_name} "
                                  f"(score={_who.score:.2f})", file=sys.stderr)
                        else:
                            print(f"[speaker] unrecognized voice "
                                  f"(best={_who.score:.2f})", file=sys.stderr)
                            if speaker_gate is not None and speaker_gate.is_set():
                                print("[speaker] voice-lock active → ignoring "
                                      "unrecognized turn", file=sys.stderr)
                                # 2026-07-02 QA: a gate-dropped turn is NOT the
                                # user — count it toward the conversation-mode
                                # idle exit, else background media (TV/YouTube)
                                # keeps the hands-free window alive forever.
                                if conversation_mode.is_set():
                                    idle_empties += 1
                                    if idle_empties >= _CONVERSATION_IDLE_EXITS:
                                        print("[conversation] only unrecognized "
                                              "voices heard — mode OFF, back to "
                                              "wake word\n", file=sys.stderr)
                                        conversation_mode.clear()
                                        idle_empties = 0
                                ui.set_state(State.IDLE)
                                followup = False
                                continue

                # M88: heard a real (non-gate-dropped) utterance — reset the
                # conversation-mode idle counter. (2026-07-02 QA: moved BELOW
                # the voice-lock gate — a dropped media turn used to reset the
                # counter it should have been incrementing.)
                idle_empties = 0

                # Hand the transcript to the security subsystem first. It
                # consumes the turn (returns True) for both challenge
                # authentication AND activate/disarm intents — claude
                # isn't called for either. Falls through to process_question
                # on a non-match. M51: an intent turn is handled but is NOT a
                # conversational Q&A, so it ends any follow-up chain.
                if security_watcher is not None:
                    try:
                        if security_watcher.handle_transcript(transcript.text):
                            ui.add_user_text(transcript.text, transcript.language)
                            ui.set_state(State.IDLE)
                            followup = False
                            continue
                    except Exception as exc:
                        print(f"[main] security.handle_transcript raised: {exc}",
                              file=sys.stderr)

                # M39: "enroll my face" / "remember my face" intent. Checked
                # AFTER security so a challenge-time utterance can't be
                # hijacked into an enrollment; BEFORE Claude so the LLM
                # doesn't try to fulfill the intent with the wrong tool.
                if on_enroll_face is not None:
                    from src.face_auth import matches_enroll_intent  # noqa: PLC0415
                    if matches_enroll_intent(transcript.text):
                        ui.add_user_text(transcript.text, transcript.language)
                        on_enroll_face()
                        ui.set_state(State.IDLE)
                        followup = False
                        continue

                # M69: voice-enrollment intents — same placement as face-enroll
                # (after security, before Claude). NAMED ("enroll Alice's voice"
                # → a household member) is checked BEFORE the primary ("enroll
                # my voice"), because "enroll voice <name>" also matches the
                # primary pattern's bare "voice". Spoken names are error-prone
                # (Whisper) — the typed console path is the reliable one.
                if on_enroll_voice is not None:
                    _named = speaker_id.parse_named_enroll_intent(transcript.text)
                    if _named is not None:
                        ui.add_user_text(transcript.text, transcript.language)
                        on_enroll_voice(_named[0], _named[1])
                        ui.set_state(State.IDLE)
                        followup = False
                        continue
                    if speaker_id.matches_enroll_intent(transcript.text):
                        ui.add_user_text(transcript.text, transcript.language)
                        on_enroll_voice()
                        ui.set_state(State.IDLE)
                        followup = False
                        continue

                # M45: knowledge-base voice intents. AFTER face-enroll (a
                # challenge/enroll utterance must win first), BEFORE Claude
                # (so the LLM writes/refreshes the corpus instead of
                # answering from training). "remember ... permanently" is
                # tested before "update ... knowledge": it's the more
                # specific intent (requires an explicit permanence word).
                if on_knowledge_remember is not None or on_knowledge_reindex is not None:
                    from src import knowledge as _kb  # noqa: PLC0415
                    _fact = (_kb.extract_remember_fact(transcript.text)
                             if on_knowledge_remember is not None else None)
                    if _fact is not None:
                        ui.add_user_text(transcript.text, transcript.language)
                        on_knowledge_remember(_fact)
                        ui.set_state(State.IDLE)
                        followup = False
                        continue
                    if (on_knowledge_reindex is not None
                            and _kb.matches_reindex_intent(transcript.text)):
                        ui.add_user_text(transcript.text, transcript.language)
                        on_knowledge_reindex()
                        ui.set_state(State.IDLE)
                        followup = False
                        continue

                # M87 — interpreter-mode activation ("Jarvis, be my
                # interpreter"). Checked among the intents (after security /
                # enroll / knowledge so those win first) and BEFORE the LLM turn
                # so the request isn't answered conversationally. Engages the
                # continuous loop handled at the top of this loop; the start
                # confirmation is spoken in both languages so the other party
                # hears it too.
                from src import interpreter as _interp_start  # noqa: PLC0415
                if _interp_start.is_start_intent(transcript.text):
                    ui.add_user_text(transcript.text, transcript.language)
                    interpreter_mode.set()
                    # Modes are mutually exclusive: entering interpreter from
                    # conversation mode must drop conversation mode, else "stop
                    # interpreting" later would silently fall back INTO it
                    # instead of returning to the wake-word baseline.
                    conversation_mode.clear()
                    print("[interpreter] mode ON (voice)", file=sys.stderr)
                    runner.speak_line(_interp_start.START_CONFIRM_EN, "en")
                    if "es" in _interp_start.LANG_PAIR:
                        runner.speak_line(_interp_start.START_CONFIRM_ES, "es")
                    ui.set_state(State.IDLE)
                    followup = False
                    continue

                # M88 — conversation-mode toggle. If active, an explicit "exit
                # conversation" leaves the mode (natural sign-offs are handled by
                # the dismissal block below, which also exits it). If inactive,
                # "let's talk" enters it. Checked here among the intents (after
                # security/enroll/knowledge) and BEFORE the LLM turn so the
                # request isn't answered conversationally.
                from src import conversation_mode as _convo  # noqa: PLC0415
                if conversation_mode.is_set():
                    if _convo.is_stop_intent(transcript.text):
                        ui.add_user_text(transcript.text, transcript.language)
                        conversation_mode.clear()
                        followup = False
                        print("[conversation] mode OFF (voice)", file=sys.stderr)
                        runner.speak_line(
                            _convo.stop_confirmation(transcript.language),
                            transcript.language)
                        ui.set_state(State.IDLE)
                        continue
                elif _convo.is_start_intent(transcript.text):
                    ui.add_user_text(transcript.text, transcript.language)
                    conversation_mode.set()
                    interpreter_mode.clear()  # mutually exclusive (see above)
                    followup = True
                    idle_empties = 0
                    print("[conversation] mode ON (voice)", file=sys.stderr)
                    runner.speak_line(
                        _convo.start_confirmation(transcript.language),
                        transcript.language)
                    ui.set_state(State.IDLE)
                    continue

                # 2026-06-02: a FOLLOW-UP utterance that is a pure sign-off
                # ("thank you, that is all") must not run a full LLM turn. With
                # a freshly-set reminder still in context, Claude re-issued
                # set_reminder on exactly such a turn and created a DUPLICATE
                # (the "fired twice" report). The sign-off was already detected
                # AFTER the turn (to close the window); short-circuiting it
                # BEFORE the turn means a dismissal takes NO action at all.
                # Scoped to follow-up turns only — a wake-word-invoked "that's
                # all" still reaches Claude, since the user deliberately
                # summoned him. Closes cleanly and silently: _announce isn't on
                # this thread (so no gate-safe spoken reply here), and a quiet
                # close after "that's all" is the expected sign-off behaviour.
                # M88: in conversation mode every turn is hands-free, so a
                # natural sign-off here ("that's all", "goodbye") both closes the
                # turn AND exits the mode — with a brief spoken acknowledgement
                # (unlike the silent follow-up close, since the user explicitly
                # ended a back-and-forth).
                is_signoff = _is_dismissal(transcript.text)
                if (followup or conversation_mode.is_set()) and is_signoff:
                    ui.add_user_text(transcript.text, transcript.language)
                    print("[main] sign-off — closing without an LLM turn\n",
                          file=sys.stderr)
                    if conversation_mode.is_set():
                        conversation_mode.clear()
                        from src import conversation_mode as _convo_exit  # noqa: PLC0415
                        print("[conversation] mode OFF (sign-off)", file=sys.stderr)
                        runner.speak_line(
                            _convo_exit.stop_confirmation(transcript.language),
                            transcript.language)
                    ui.set_state(State.IDLE)
                    followup = False
                    continue

                # M52: pass the AudioSession so process_question can run the
                # barge-in monitor — only the voice path has the mic, so only
                # the voice path can be interrupted. Returns True iff the user
                # cut Jarvis off mid-reply with "Hey Jarvis".
                interrupted = runner.process_question(
                    transcript.text, transcript.language, session=session,
                    speaker_name=speaker_name, speaker_lang=speaker_lang,
                    vocal_cue=vocal_cue,
                )
                # M51/M52: open the follow-up window so the next utterance
                # needs no wake word. A barge-in ALWAYS opens it — the user
                # interrupted precisely to say something now, and the wake
                # word that triggered the barge-in carries no transcript to
                # sign off with, so the dismissal check is skipped on that
                # path. Otherwise (a normal completed turn) honour an explicit
                # sign-off ("that's all", "no thank you", ...) and close
                # cleanly instead of leaving the mic open for 12s.
                if interrupted:
                    print("[main] barge-in — straight into listening\n",
                          file=sys.stderr)
                    followup = True
                    barged = True  # M52: next iteration skips session.drain()
                elif is_signoff:
                    print("[main] user signed off — no follow-up window\n",
                          file=sys.stderr)
                    followup = False
                else:
                    followup = True

    # Mic-session supervisor: the listening loop degrades and retries, never
    # dies. Any escape from the body (mic-open failure, a mid-session device
    # error out of wait_for_wake_word, a stalled-stream RuntimeError from
    # AudioSession.read) lands here; we surface it once per outage — console
    # line + spoken alert (the speakers usually survive a mic loss) — then
    # re-open on a capped backoff. The typed/remote paths keep running
    # throughout (text_input_loop is its own thread).
    mic_failures = 0
    try:
        while not ui.shutdown.is_set():
            try:
                _voice_session_loop()
            except Exception as exc:  # noqa: BLE001 — the voice loop must never die
                mic_failures += 1
                delay = min(60.0, 10.0 * mic_failures)
                if mic_failures == 1:
                    traceback.print_exc(file=sys.stderr)
                    ui.add_system_text(
                        "⚠ Microphone unavailable — voice commands are down; retrying…")
                    runner.speak_line(
                        "I seem to have lost the microphone, sir. I'll keep trying.")
                print(f"[audio] voice loop error ({type(exc).__name__}: {exc}) — "
                      f"retry {mic_failures} in {delay:.0f}s", file=sys.stderr)
                ui.shutdown.wait(delay)
                continue
            # A clean return means shutdown was requested — exit the supervisor.
            break
    finally:
        # Quit / shutdown: seal whatever's still in memory so we don't lose it.
        if runner.has_active_conversation():
            print("[memory] sealing in-progress session on shutdown...", file=sys.stderr)
            runner.seal_on_shutdown()


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
