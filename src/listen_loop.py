"""listen_loop — the voice path: wake word to dispatched turn.

Extracted from main.py (2026-07-28), the last of the four cuts that took the
entry point from 2,805 lines to ~660. What remains in main.py is main()
itself: load config, build subsystems, start threads, wait.

This module owns the CONVERSATION-level state machine that sits above a single
turn — everything about *when* Jarvis should be listening and to whom:

  - wake-word arming, and barge-in while a reply is playing
  - the M51 follow-up window (answer a follow-up with no fresh "Hey Jarvis")
  - M88 conversation mode (hands-free until silence or an explicit sign-off)
  - M87 interpreter mode (relay between two languages, no wake word at all)
  - the M69 speaker gate (optionally answer only enrolled voices)
  - dismissal detection, which closes a window early on "that's all"

A single turn — history, streaming, the tool loop, TTS — belongs to
TurnRunner, which this loop constructs and calls. Subsystem assembly belongs
to src/bootstrap.py. Keeping those three apart is what makes any of them
testable: process_question was once a closure in here, closing over ~15
locals, and could not be tested at all.

The four window constants below are the tuning surface for how long Jarvis
stays listening in each mode; each carries the reasoning for its value.
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import traceback
from typing import Callable

from src import speaker_id
from src.audio import AudioSession
from src.config import Config
from src.memory import default_base_dir
from src.plex_laptop import PlexLaptopClient
from src.plex_mcp import PlexMCPClient
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
