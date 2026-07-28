"""TurnRunner — one conversational turn, from text in to speech out.

Extracted from main.py (2026-07-28). `process_question` was originally a
closure inside `listen_loop`, closing over ~15 locals with no seam, and was
therefore untestable; promoting it to a class gave it an injectable surface
and its first automated coverage. Moving it out of the 2,800-line entry point
is the next step of the same arc.

A turn owns: appending to history, streaming the model response, running the
agentic tool loop, feeding completed sentences to TTS while the model is still
generating, and finalising (memory write, telemetry). The conversation-level
concerns around it — wake word, follow-up windows, conversation mode — stay in
listen_loop; the subsystem assembly stays in bootstrap.

TESTING NOTE — read before changing the imports below.
`scripts/turn_runner_test.py` swaps out this module's collaborators by
rebinding them AT MODULE LEVEL (`turn_runner.stream_response = ...`,
`turn_runner.speak_streaming`, `turn_runner.speak`, `turn_runner.MemoryStore`,
`turn_runner._seal_session`). That works only because the code below resolves
those names from module globals at CALL time. If you rewrite an import to
`from src import llm` + `llm.stream_response(...)`, or bind a collaborator to
`self` in __init__, the test will keep passing while exercising the REAL
implementation instead of the fake — a green test that tests nothing. Change
the test in the same commit if you change how these names are resolved.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime

from src import aec_barge
from src.audio import AudioSession
from src.config import Config
from src.gates import CountedEvent
from src.llm import TelemetryRecord, stream_response, stream_translation
from src.memory import MemoryStore, SummaryRecord, summarize_session
from src.plex_laptop import PlexLaptopClient
from src.plex_mcp import PlexMCPClient
from src.text_to_speech import speak, speak_streaming
from src.tray import State
from src.ui import JarvisUI
from src.wake_word import monitor_for_wake_word


MAX_PAIRS = 10            # cap conversation at 10 exchanges (20 messages)
IDLE_RESET_SEC = 600.0    # 10 min of silence → forget conversation


def _barge_in_enabled() -> bool:
    """M52 kill switch. JARVIS_BARGE_IN ∈ {0,false,no,off} disables barge-in
    entirely — the escape hatch if the soak ever shows the concurrent
    wake-word monitor stutters the Python-fed TTS path (the documented
    stutter-gate risk: see project_security_audio_stutter_gate). Default on.
    Read once at import; it never changes mid-session."""
    raw = os.getenv("JARVIS_BARGE_IN", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


_BARGE_IN_ENABLED = _barge_in_enabled()


def _trim_history(history: list[dict]) -> None:
    """Drop oldest pairs in place, keeping the most recent MAX_PAIRS exchanges.
    Always called when history is in clean pair state (ends on 'assistant')."""
    while len(history) > MAX_PAIRS * 2:
        del history[:2]  # drop oldest user + assistant


def _seal_session(
    memory: MemoryStore,
    history: list[dict],
    language: str,
    started_at: str,
    cfg: Config,
) -> None:
    """Best-effort: ask Haiku for a summary of the just-ended conversation
    and append it to summaries.jsonl. Idempotent — caller clears history
    after this returns. Never raises; logs and moves on if summarization fails."""
    if not history:
        return
    summary_text = summarize_session(
        history,
        language=language,
        api_key=cfg.anthropic_api_key,
        model=cfg.summary_model,
    )
    if not summary_text:
        return  # already logged by summarize_session
    memory.append_summary(SummaryRecord(
        started_at=started_at,
        ended_at=datetime.now().isoformat(timespec="seconds"),
        language=language,
        summary=summary_text,
    ))
    print(f"[memory] sealed session: {summary_text}", file=sys.stderr)


class TurnRunner:
    """Runs one conversation turn end-to-end and owns the session state.

    Owns the conversation `history`, the recalled `summaries`, the active
    session's language / start-time / last-turn-time, the `MemoryStore`, and a
    processing lock. The PC-voice path and the text/phone path share ONE
    TurnRunner; both call `process_question`, which holds the lock for the
    whole LLM-stream + TTS duration — so a typed message that lands while
    Jarvis is answering a spoken one waits its turn (no overlapping responses).

    This is the unit that used to be the `process_question` closure inside
    `listen_loop`; pulled out so it has a testable surface
    (scripts/turn_runner_test.py) and listen_loop stays a readable loop.
    """

    def __init__(
        self,
        cfg: Config,
        ui: JarvisUI,
        reset_event: threading.Event,
        plex_client: PlexMCPClient | None = None,
        plex_laptop_client: PlexLaptopClient | None = None,
        pc_speaking: "threading.Event | None" = None,
        announce_speaking: "threading.Event | None" = None,
        mic_device: "int | None" = None,
    ) -> None:
        self._cfg = cfg
        self._ui = ui
        self._reset_event = reset_event
        self._plex_client = plex_client
        self._plex_laptop_client = plex_laptop_client
        # M88 Phase 2: the resolved mic device index for the hands-free barge-in
        # duplex stream (None = default input). Only used when
        # JARVIS_HANDS_FREE_BARGE is on; otherwise inert.
        self._mic_device = mic_device
        # "PC is speaking out loud" gate — SET while this turn plays audio so
        # the always-on voice-capture loop discards self-audio (omni-mic echo
        # fix). Shared with the Announcer (announces set it too); a fresh local
        # Event if unwired (tests / standalone) so it's never None.
        self._pc_speaking = pc_speaking if pc_speaking is not None else CountedEvent()
        # Cooperative SPEECH gate (the 2026-05-19 stutter post-mortem) — SET
        # while this turn plays audio so the armed CPU loops (SoundDetector's
        # PANNs inference, SecurityWatcher's YOLO/grab/gc bursts) DEFER and
        # don't starve the Python-fed TTS path. Same Event the Announcer sets
        # for proactive announces; extending it to TURN replies here closes the
        # 2026-06-01 gap where a reply spoken while armed stuttered because only
        # announces were gated (see build_announcer's CONTRACT). Distinct from
        # `pc_speaking` (mic-capture echo) — different failure, same lifetime.
        self._announce_speaking = (
            announce_speaking if announce_speaking is not None else CountedEvent()
        )

        self._history: list[dict] = []
        self._last_turn_time = 0.0
        self._memory = MemoryStore()
        self._memory.prune(retain_raw_days=cfg.retain_raw_days)
        self._summaries = self._memory.recent_summaries(cfg.memory_recall_count)
        print(f"[memory] loaded {len(self._summaries)} prior session summaries",
              file=sys.stderr)
        # Language + start-time of the *currently active* session (the one
        # being built up in `history`). Reset on every seal.
        self._session_language = "en"
        self._session_started_at = datetime.now().isoformat(timespec="seconds")
        # Serializes process_question so voice and text paths never run together.
        self._lock = threading.Lock()

    def has_active_conversation(self) -> bool:
        """True if there's an unsealed conversation in memory — the reset
        handler + shutdown seal use this to decide whether there's anything to
        seal/log."""
        return bool(self._history)

    def seal_and_refresh(self) -> None:
        """Seal the active session (if any) and reload summaries for the next one."""
        _seal_session(self._memory, self._history, self._session_language,
                      self._session_started_at, self._cfg)
        self._history.clear()
        self._summaries = self._memory.recent_summaries(
            self._cfg.memory_recall_count)
        self._session_started_at = datetime.now().isoformat(timespec="seconds")

    def seal_on_shutdown(self) -> None:
        """Final seal at app quit — persist whatever's still in memory. No
        reload (we're shutting down)."""
        _seal_session(self._memory, self._history, self._session_language,
                      self._session_started_at, self._cfg)

    def _emit_remote_reply(self, reply_text: "Callable[[str], None] | None",
                           answer: str) -> None:
        """Post the final reply to a per-turn TEXT sink (Discord), if one was
        supplied. ONLY the originating surface — deliberately NOT the
        add_jarvis_text broadcast, so a PC/voice/phone turn never leaks into a
        shared Discord channel. Covers the success, apology, and empty-reply
        paths so a remote user always gets *something* back. Fail-soft — a post
        hiccup can't break the turn."""
        if reply_text is None:
            return
        try:
            reply_text(answer)
        except Exception as exc:  # noqa: BLE001
            print(f"[main] reply_text sink failed: {exc}", file=sys.stderr)

    def _begin_turn(self, text: str, language: str,
                    attachments: list[dict] | None) -> None:
        """Open one turn: apply a pending reset/idle session boundary, capture
        the session language on the first turn, surface the user message to the
        console/UI, and append it to history (as a content-block list when there
        are attachments, else a plain string). Runs under the caller's lock —
        does NOT acquire self._lock itself."""
        # Apply any pending reset/idle boundary BEFORE this turn — so the
        # user's question starts a fresh session rather than tacking onto
        # a stale one.
        if self._reset_event.is_set():
            if self._history:
                print(f"[main] conversation reset (manual; sealing {len(self._history)} msgs)")
                self._ui.add_system_text("conversation reset.")
            self.seal_and_refresh()
            self._reset_event.clear()
        elif self._history and (time.time() - self._last_turn_time) > IDLE_RESET_SEC:
            print(f"[main] conversation reset (idle >{IDLE_RESET_SEC:.0f}s)")
            self._ui.add_system_text("conversation reset (idle).")
            self.seal_and_refresh()

        # First turn of a (possibly new) session — capture its language.
        if not self._history:
            self._session_language = language or "en"
            self._session_started_at = datetime.now().isoformat(timespec="seconds")

        print(f"\n[user, {language}] {text}")
        print("[jarvis] ", end="", flush=True)
        self._ui.add_user_text(text, language)

        # Build user-message content. With attachments: list of blocks
        # (each attachment first so Claude has the doc/image in context
        # before the question text). Without: plain string (cheaper).
        if attachments:
            content: list[dict] | str = list(attachments)
            if text:
                content.append({"type": "text", "text": text})
        else:
            content = text

        self._history.append({"role": "user", "content": content})

    def _finalize_turn(self, *, response_chunks: list[str], interrupted: bool,
                       text: str, language: str,
                       reply_text: "Callable[[str], None] | None",
                       reply_audio: "Callable[[bytes], None] | None",
                       speaker: "str | None" = None) -> bool:
        """Assemble the streamed reply, persist + relay it, and (for a phone
        turn) synthesize the reply audio. Returns `interrupted` (the value the
        voice loop uses to decide whether to open a follow-up window). Runs
        under the caller's lock — does NOT acquire self._lock itself.

        An empty reply pops the orphan user message and (for a remote turn)
        emits a short acknowledgement so the user isn't left in silence."""
        full_response = "".join(response_chunks).strip()
        if not full_response:
            self._history.pop()  # nothing came back; drop the orphan user message
            # Don't leave a REMOTE user in silence wondering if the turn
            # even landed — Claude can legitimately return an empty turn
            # (e.g. a tool-only turn that yields no prose). Emit a short
            # acknowledgement to the originating surface, mirroring the
            # apology path's fan-out. The PC/voice path stays silent (the
            # user is present and saw the state indicators); nothing is
            # appended to history (nothing was actually said).
            if not interrupted and (reply_text is not None or reply_audio is not None):
                ack = ("Disculpe, no tengo nada que añadir."
                       if language == "es"
                       else "I didn't have anything to add, sir.")
                self._ui.add_jarvis_text(ack)
                self._emit_remote_reply(reply_text, ack)
            return interrupted

        self._history.append({"role": "assistant", "content": full_response})
        _trim_history(self._history)
        self._last_turn_time = time.time()

        # Persist the completed exchange to today's transcript file. M82 —
        # tag the user side with the voice-identified speaker so
        # recall_conversation can answer "what did Alice ask me to do?".
        self._memory.record_turn(text, full_response, language, speaker=speaker)

        self._ui.add_jarvis_text(full_response)
        self._emit_remote_reply(reply_text, full_response)

        # M48.2b: speak the reply to the phone that asked (UNICAST,
        # mute-independent). Whole-reply-at-completion v1 — matches the
        # at-completion text parity; streamed audio is a documented
        # deferrable. Same edge-tts voice as the PC (VOICE_BY_LANG).
        # Defensive: the text already reached the phone via the fan-out
        # above, so a synth/transport hiccup must NEVER break the turn.
        if reply_audio is not None:
            try:
                import asyncio  # noqa: PLC0415 — lazy per main.py convention
                from src.text_to_speech import (  # noqa: PLC0415
                    DEFAULT_VOICE, VOICE_BY_LANG, _fetch_mp3_with_retry,
                )

                voice = VOICE_BY_LANG.get(language, DEFAULT_VOICE)
                mp3 = asyncio.run(_fetch_mp3_with_retry(full_response, voice))
                reply_audio(mp3)
            except Exception as exc:  # noqa: BLE001
                print(f"[main] phone reply audio failed: {exc}",
                      file=sys.stderr)
        print()
        return interrupted

    def _stream_and_speak(
        self, *, llm_stream, pc_silent: bool, barge_enabled: bool,
        session: "AudioSession | None", interrupt_event: "threading.Event | None",
        language: str, reply_text: "Callable[[str], None] | None",
        aec_barge_on: bool = False,
    ) -> bool:
        """Run the LLM stream and, when audible, speak it — holding the
        cooperative speech gates for the spoken portion and running the
        barge-in monitor for its duration. Returns True if the turn ERRORED
        (the caller then returns False after the apology surfaced here); False
        on a clean run. Runs under the caller's lock — does NOT acquire it.

        For the audible portion it SETS self._pc_speaking (omni-mic echo
        suppression) + self._announce_speaking (the armed PANNs/YOLO CPU loops
        defer so the TTS path doesn't stutter), and clears both in `finally`.
        On an exception it pops the orphan user message and surfaces a brief
        spoken/text apology."""
        speaking_aloud = not pc_silent
        monitor_stop = threading.Event()
        monitor_thread: threading.Thread | None = None

        # Mark "PC is speaking out loud" for the audible portion of this turn so
        # the always-on voice-capture loop (esp. the M51 follow-up window)
        # discards anything it picks up while we talk — the mic stops
        # self-capturing our own reply (the omni-mic echo, 2026-05-29). Silent
        # turns (mute / phone) never set it; cleared in `finally` the instant
        # our audio ends. The announce_speaking gate (set alongside) makes the
        # armed CPU loops defer for the spoken portion, exactly as for a
        # proactive announce (2026-06-01: only announces were gated → stutter).
        if speaking_aloud:
            self._pc_speaking.set()
            self._announce_speaking.set()

        try:
            if pc_silent:
                # Drain the LLM stream silently. response_chunks gets populated
                # inside llm_stream via the print + append, so finalize works
                # unchanged. State stays THINKING — caller flips to IDLE after.
                for _ in llm_stream():
                    pass
            else:
                # M52: run the wake-word barge-in monitor for the duration of
                # playback. It reads the mic on its own thread and, on a "Hey
                # Jarvis", just sets interrupt_event — speak_streaming polls it
                # and performs the sd.stop() cut on its own (stream-owning)
                # thread, the WASAPI thread-affinity fix. M88 Phase 2: when
                # `aec_barge_on` is set we do NOT start this monitor — the duplex
                # AEC stream inside speak_streaming does the detection (hands-free
                # talk-over) and sets the same interrupt_event. (Named *_on to not
                # shadow the imported `aec_barge` module — see process_question.)
                if barge_enabled:
                    monitor_thread = threading.Thread(
                        target=monitor_for_wake_word,
                        args=(session, interrupt_event, monitor_stop),
                        kwargs={"threshold": self._cfg.wake_word_threshold},
                        name="BargeInMonitor",
                        daemon=True,
                    )
                    monitor_thread.start()
                speak_streaming(
                    llm_stream(),
                    language=language,
                    on_first_audio=lambda: self._ui.set_state(State.SPEAKING),
                    on_amplitude=self._ui.set_amplitude,
                    interrupt_event=interrupt_event,
                    aec_barge_device=(self._mic_device if aec_barge_on else None),
                    # M88 Phase 2: on a hands-free cut, flip the orb to LISTENING
                    # the instant we detect — not after the stream teardown — so
                    # it doesn't linger on SPEAKING.
                    on_barge=lambda: self._ui.set_state(State.LISTENING),
                )
        except Exception as exc:
            self._history.pop()  # keep history alternating user/assistant cleanly
            print(f"\n[main] LLM/TTS failed: {exc}")
            # M20: don't leave the user in silence after a partial reply. Speak
            # a brief apology in their language via the simpler Tier A path (no
            # streaming pipeline that could fail again). Wrap in defensive
            # try/except so the apology itself can't break the loop. Add to
            # transcript too so the console reflects it. When silent (muted OR
            # phone-text origin), skip the spoken apology but still surface its
            # text — the phone/console sees the hiccup, the PC stays quiet.
            apology = (
                "Disculpe, tuve un problema técnico. ¿Podría intentarlo de nuevo?"
                if language == "es"
                else "Apologies, a technical hiccup. Could you try that again?"
            )
            if not pc_silent:
                self._ui.set_state(State.SPEAKING)
                try:
                    speak(apology, language=language)
                except Exception as apology_exc:
                    print(f"[main] apology TTS also failed: {apology_exc}")
            # Phone-audio turns: the apology TEXT still reaches the phone via the
            # fan-out; we deliberately don't synth audio on the error path
            # (extra failure surface for little gain — the text conveys it).
            self._ui.add_jarvis_text(apology)
            self._emit_remote_reply(reply_text, apology)
            return True
        finally:
            # Stop suppressing voice capture + release the speech gate the
            # instant our audio ends (so the armed loops resume promptly).
            if speaking_aloud:
                self._pc_speaking.clear()
                self._announce_speaking.clear()
            # M52: always wind the barge-in monitor down — normal end,
            # exception, OR barge-in. monitor_stop makes its next session.read()
            # (≤80ms away) the last; the join is instant on a clean turn and
            # capped short otherwise (daemon thread).
            if monitor_thread is not None:
                monitor_stop.set()
                monitor_thread.join(timeout=1.0)
        print()
        return False

    def process_question(
        self,
        text: str,
        language: str,
        attachments: list[dict] | None = None,
        origin: str = "voice",
        reply_audio: "Callable[[bytes], None] | None" = None,
        reply_text: "Callable[[str], None] | None" = None,
        reply_image: "Callable[[bytes, str], None] | None" = None,
        session: AudioSession | None = None,
        speaker_name: "str | None" = None,
        speaker_lang: "str | None" = None,
        vocal_cue: "str | None" = None,
    ) -> bool:
        """Run one full turn: reset/idle checks, LLM stream, TTS, persist.

        Returns True if the user barged in (M52) and the turn was cut short —
        the voice loop uses that to drop straight into a listening window.
        Every other path (normal completion, errors, the text/phone paths
        that ignore the return) yields False.

        Called from both voice path (after STT, no attachments) and text path
        (after typed Enter, optional attachments). Caller is responsible for
        setting THINKING before calling; SPEAKING is set automatically when
        TTS audio starts.

        `session` (M52) — the PC mic AudioSession, passed ONLY by the voice
        path. Its presence (plus an audible, non-restricted turn) is what
        enables the barge-in monitor: the monitor needs the mic, and only the
        voice path holds the session. Text/phone callers pass None, so the
        interrupt machinery threads through every layer as an inert no-op and
        those paths stay byte-identical to pre-M52.

        `origin` (M48.2/M48.2a): where the turn came from — the single
        honest signal two separate per-turn concerns derive from:
          - "voice"      : PC wake word     → speaks (subject to mute), full tools
          - "console"    : PC typed         → speaks (subject to mute), full tools
          - "phone_text" : phone typed      → text-only, RESTRICTED tools
          - "phone_voice": phone PTT (M48.3)→ audio→phone, RESTRICTED tools
        `text_only` and `restricted` are derived below — they are NOT the
        same axis (phone_voice will speak yet still be restricted), which is
        why the prior single `speak` bool was replaced by `origin`. The phone
        gets a deliberately reduced tool surface (no system_control /
        pc_shell / collector / plex_action; no file/screen/camera) —
        enforced server-side in stream_response, NOT prompt-only
        ([[feedback-jarvis-least-privilege]], [[feedback-diag-vs-action-split]]).

        `speaker_name` / `speaker_lang` (M80): the voice-identified speaker for
        this turn (from M69's identify on the captured clip), or None. Only the
        PC-voice path supplies them; they thread into stream_response as a small
        uncached system note so Claude knows WHO it's talking to (addresses them
        by name) and their usual language. Absent ⇒ no speaker block (unchanged
        behaviour for remote/unrecognized turns).

        When attachments is non-empty, the user message becomes a list of
        content blocks (attachments first, then a text block) instead of a
        plain string. History keeps that structure verbatim, so multi-turn
        Q&A about an attached document works naturally until reset/trim.
        """
        # M48.2/M48.2a — the two per-turn concerns, derived from one origin.
        # text_only: phone-text is text-only (M48.2); everything else speaks
        # (subject to mute). restricted: any phone origin gets the reduced
        # tool surface (M48.2a) — phone_voice (M48.3) will speak AND be
        # restricted, which is why these are separate axes off `origin`.
        # NB: deliberately NOT named `speak` — that shadowed the imported
        # speak() function and crashed the apology path ('bool' not callable).
        # "discord" (2026-06-02): a Discord-channel turn is text-only (the
        # reply posts to the channel, the PC never speaks to its empty room)
        # AND restricted (an internet-relayed, multi-human surface gets the
        # same reduced tool boundary as the phone — no system/shell/file/code).
        text_only = origin in ("phone_text", "discord")
        restricted = origin in ("phone_text", "phone_voice", "discord")

        with self._lock:
            # Seal any stale session, capture session language, surface + append
            # the user message. (Boundary logic lives in _begin_turn.)
            self._begin_turn(text, language, attachments)

            response_chunks: list[str] = []

            # Snapshot engineer-mode state once per turn — same discipline as
            # mute. Mid-turn toggle applies to the next turn (changing thinking
            # budget mid-stream isn't supported).
            engineer = self._ui.is_engineer_mode()

            def on_telemetry(rec: TelemetryRecord) -> None:
                """Called by stream_response at the end of each turn. Surfaces
                the per-turn signal that previously only existed as a stderr
                log line. Format chosen for SRE skim-readability — verb
                (which tools), then iteration count if >1, then how long,
                then how much it cost. The 'thinking' marker calls out
                engineer-mode turns since their token cost is meaningfully
                higher."""
                self._ui.add_session_tokens(rec.total_tokens)
                bits: list[str] = []
                if rec.tools_used:
                    bits.append(", ".join(rec.tools_used))
                if rec.iterations > 1:
                    bits.append(f"{rec.iterations} iters")
                if rec.thinking_enabled:
                    bits.append("thinking")
                bits.append(f"{rec.elapsed_sec:.1f}s")
                bits.append(f"{rec.total_tokens:,} tok")
                if rec.paused:
                    bits.append("PAUSED(10-iter cap)")
                self._ui.add_telemetry_chip(" · ".join(bits))

            def on_image_captured(image_bytes: bytes, media_type: str, tool_name: str) -> None:
                """Called by stream_response when a vision tool returns an
                image. Renders the bytes as an inline thumbnail in the
                console transcript so the user sees *what Jarvis saw*, not
                just the text description. Label maps tool name → emoji +
                source for at-a-glance recognition."""
                label = {
                    "camera_snapshot": "📷 webcam snapshot",
                    "screen_snapshot": "🖥 screen snapshot",
                }.get(tool_name, f"🖼 {tool_name}")
                self._ui.add_image_thumbnail(image_bytes, label)
                # M71 — relay the captured frame to the originating remote
                # surface (Discord) if a per-turn image sink was supplied.
                # The sink BUFFERS it; the photo is posted into the same thread
                # as the reply (see discord_bot._post). Only Discord sets this
                # sink (camera is clawed back for origin="discord" only), so a
                # PC/phone turn never relays its webcam frame anywhere.
                if reply_image is not None:
                    try:
                        reply_image(image_bytes, media_type)
                    except Exception as exc:  # noqa: BLE001 — never break the turn
                        print(f"[main] reply_image sink failed: {exc}",
                              file=sys.stderr)

            def llm_stream():
                # `interrupt_event` is a free variable bound just below (before
                # this generator is ever called) — None unless barge-in is
                # active for this turn, in which case stream_response polls it
                # and closes the HTTP stream when the user cuts in.
                for chunk in stream_response(
                    api_key=self._cfg.anthropic_api_key,
                    messages=self._history,
                    model=self._cfg.claude_model,
                    summaries=self._summaries,
                    plex_client=self._plex_client,
                    plex_laptop_client=self._plex_laptop_client,
                    on_complete=on_telemetry,
                    on_image_captured=on_image_captured,
                    engineer_mode=engineer,
                    restricted=restricted,
                    origin=origin,
                    speaker_name=speaker_name,
                    speaker_lang=speaker_lang,
                    vocal_cue=vocal_cue,
                    interrupt_event=interrupt_event,
                ):
                    response_chunks.append(chunk)
                    print(chunk, end="", flush=True)
                    yield chunk

            # Capture mute state at the start of the turn. If the user
            # toggles mid-response, the change applies to the NEXT turn —
            # mid-stream stop would be jarring and the streamed text is
            # already visible in the console regardless.
            muted = self._ui.is_muted()

            # M48.2: phone-text turns are text-only by origin (text_only),
            # independent of the global mute toggle. Folded into the same
            # silent-drain path mute already uses — one branch, no new code
            # path to keep correct.
            silent = muted or text_only

            # M48.2b: reply_audio set ⇒ this turn's reply audio goes to the
            # ORIGINATING PHONE, never the PC. The PC therefore drains
            # silently for it (like the mute path), and the synth-to-phone
            # below is mute-INDEPENDENT (the locked rule: the remote user
            # must hear the answer they asked for, regardless of PC mute).
            pc_silent = silent or reply_audio is not None

            # Barge-in eligibility: a PC-voice turn that actually speaks aloud
            # (we need the mic + audible playback). M88 Phase 2 — hands-free
            # talk-over (AEC duplex) takes precedence when enabled AND the mic is
            # pinned to a known device (the duplex stream needs an explicit input
            # index); otherwise fall back to M52 wake-word barge-in. Both set the
            # same interrupt_event; phone/console/muted turns get neither.
            _barge_eligible = (
                session is not None and origin == "voice" and not pc_silent
            )
            aec_barge_on = (
                _barge_eligible
                and aec_barge.is_enabled()
                and self._mic_device is not None
            )
            barge_enabled = (
                not aec_barge_on and _BARGE_IN_ENABLED and _barge_eligible
            )
            interrupt_event = (
                threading.Event() if (barge_enabled or aec_barge_on) else None
            )

            # Run the stream and (when audible) speak it — holds the speech
            # gates + barge-in monitor; on error it surfaces an apology, pops
            # the orphan user message, and returns True so we bail here.
            if self._stream_and_speak(
                llm_stream=llm_stream, pc_silent=pc_silent,
                barge_enabled=barge_enabled, session=session,
                interrupt_event=interrupt_event, language=language,
                reply_text=reply_text, aec_barge_on=aec_barge_on,
            ):
                return False

            # M52: did the user barge in? interrupt_event survives the monitor
            # join (Events don't auto-reset), so this read is stable. A
            # barged-in turn keeps whatever partial reply was streamed — it is
            # real context for the follow-up ("as I was saying...") — and
            # signals the voice loop to open a listening window at once.
            interrupted = interrupt_event is not None and interrupt_event.is_set()
            if interrupted:
                print("[main] turn interrupted by barge-in", file=sys.stderr)

            # Assemble + persist + relay the reply (and synth phone audio).
            return self._finalize_turn(
                response_chunks=response_chunks, interrupted=interrupted,
                text=text, language=language,
                reply_text=reply_text, reply_audio=reply_audio,
                speaker=speaker_name,
            )

    def interpret(self, text: str, language: str) -> None:
        """M87 — one interpreter-mode turn: translate `text` (spoken in
        `language`) into the OTHER language of the configured pair and speak it
        in that language's voice. A faithful relay — NO history, memory, tools,
        persona, or barge-in (it's not a conversation). Holds the speech gates
        for the spoken portion exactly like a normal turn (so the omni-mic
        doesn't self-capture the translation and the armed CPU loops defer).
        Fail-soft: a translation/TTS error logs and is swallowed so the
        interpreter loop keeps running. Runs under self._lock so a typed turn
        can't overlap its audio."""
        from src import interpreter as _interp  # noqa: PLC0415
        target = _interp.other_language(language)
        self._ui.add_user_text(text, language)
        print(f"\n[interpret {language}->{target}] {text}", file=sys.stderr)

        chunks: list[str] = []

        def translation_stream():
            for chunk in stream_translation(
                api_key=self._cfg.anthropic_api_key,
                text=text, target_lang=target,
                model=self._cfg.claude_model,
            ):
                chunks.append(chunk)
                yield chunk

        with self._lock:
            self._pc_speaking.set()
            self._announce_speaking.set()
            try:
                speak_streaming(
                    translation_stream(),
                    language=target,
                    on_first_audio=lambda: self._ui.set_state(State.SPEAKING),
                    on_amplitude=self._ui.set_amplitude,
                )
            except Exception as exc:  # noqa: BLE001 — never break the interpreter loop
                print(f"[interpret] failed: {exc}", file=sys.stderr)
            finally:
                self._pc_speaking.clear()
                self._announce_speaking.clear()

        translation = "".join(chunks).strip()
        if translation:
            self._ui.add_jarvis_text(translation)

    def speak_line(self, text: str, language: str = "en") -> None:
        """Speak a fixed line aloud on the calling (audio-owning) thread,
        holding the speech gates so the mic doesn't self-capture it and the
        armed CPU loops defer. Used for interpreter mode's start/stop
        confirmations. Runs under self._lock so it can't overlap a turn's audio.
        Fail-soft."""
        self._ui.add_jarvis_text(text)
        with self._lock:
            self._pc_speaking.set()
            self._announce_speaking.set()
            try:
                self._ui.set_state(State.SPEAKING)
                speak(text, language=language)
            except Exception as exc:  # noqa: BLE001
                print(f"[main] speak_line failed: {exc}", file=sys.stderr)
            finally:
                self._pc_speaking.clear()
                self._announce_speaking.clear()
