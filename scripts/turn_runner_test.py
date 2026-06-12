r"""Regression test for main.TurnRunner (the QoL Tier-2.1-Step-2 extraction).

WHY THIS EXISTS
---------------
`process_question` used to be a closure inside `listen_loop` — untestable
(closed over ~15 locals, no seam). Promoting it to `TurnRunner` gave it an
injectable surface, so this is the FIRST automated coverage of the turn
mechanics: history append/trim/pop, the reset+idle boundaries, the
silent-vs-speak path selection, origin→restricted derivation, and the
fail-soft apology path. These are the invariants a future refactor of the
conversation engine must not break.

Everything is exercised with fakes — no network, no audio, no disk:
  - main.MemoryStore   -> FakeMemory   (no disk I/O at construction)
  - main.stream_response -> a fake generator (configurable chunks / raise)
  - main.speak_streaming / main.speak -> recorded (asserted NOT called on
    silent paths; consumed on the speak path)
  - main._seal_session -> recorded (the boundary tests assert it fired)

    venv\Scripts\python.exe scripts\turn_runner_test.py     # exit 0 = pass
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, condition: bool) -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


# --- Fakes ----------------------------------------------------------------

class FakeUI:
    """Records the UI calls process_question makes; mute/engineer toggled per
    test. Only the methods process_question actually touches are implemented."""

    def __init__(self) -> None:
        self.muted = False
        self.engineer = False
        self.user_texts: list[tuple[str, str]] = []
        self.jarvis_texts: list[str] = []
        self.system_texts: list[str] = []
        self.states: list = []
        self.telemetry: list[str] = []

    def is_muted(self) -> bool:
        return self.muted

    def is_engineer_mode(self) -> bool:
        return self.engineer

    def add_user_text(self, text, language) -> None:
        self.user_texts.append((text, language))

    def add_jarvis_text(self, text) -> None:
        self.jarvis_texts.append(text)

    def add_system_text(self, text) -> None:
        self.system_texts.append(text)

    def add_image_thumbnail(self, image_bytes, label) -> None:
        pass

    def add_session_tokens(self, n) -> None:
        pass

    def add_telemetry_chip(self, text) -> None:
        self.telemetry.append(text)

    def set_state(self, state) -> None:
        self.states.append(state)

    def set_amplitude(self, amp) -> None:
        pass


class FakeMemory:
    """Stands in for MemoryStore — no disk. Records recorded turns."""

    summaries_to_return: list = []

    def __init__(self) -> None:
        self.turns: list[tuple] = []
        self.sealed: list = []

    def prune(self, retain_raw_days=0) -> None:
        pass

    def recent_summaries(self, n) -> list:
        return list(FakeMemory.summaries_to_return)

    def record_turn(self, text, response, language, speaker=None) -> None:
        self.turns.append((text, response, language, speaker))

    def append_summary(self, rec) -> None:
        self.sealed.append(rec)


class FakeStream:
    """Stands in for stream_response. Records the kwargs of the last call so
    tests can assert on `restricted`; yields configured chunks, or raises."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.raises: Exception | None = None
        self.last_kwargs: dict | None = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return iter(list(self.chunks))


def _fake_cfg() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        anthropic_api_key="test-key",
        claude_model="test-model",
        retain_raw_days=30,
        memory_recall_count=10,
        wake_word_threshold=0.5,
    )


# --- Test harness wiring --------------------------------------------------
# Patch the module-level names TurnRunner resolves at call time. MemoryStore
# is patched before construction (its __init__ builds one).
main.MemoryStore = FakeMemory
_orig_seal = main._seal_session
_orig_stream = main.stream_response
_orig_speak_streaming = main.speak_streaming
_orig_speak = main.speak

_seal_calls: list = []
main._seal_session = lambda *a, **k: _seal_calls.append((a, k))

_speak_calls: list = []
main.speak = lambda *a, **k: _speak_calls.append((a, k))


_speak_streaming_calls: list = []
# Observation slots for the pc_speaking gate (echo fix). The test sets
# _pc_event_ref[0] to a runner's _pc_speaking before a speak-path call; the
# fake records whether it was set at the moment "audio" was playing.
_pc_event_ref: list = [None]
_pc_observed: dict = {"set_during_speak": None}
# Same observation pattern for the announce_speaking CPU gate (the 2026-06-01
# turn-reply stutter fix): it MUST be set while a spoken reply 'plays' so the
# armed PANNs/YOLO loops defer.
_ann_event_ref: list = [None]
_ann_observed: dict = {"set_during_speak": None}


def _consuming_speak_streaming(gen, *a, **k):
    """The speak path: real speak_streaming would play audio while pulling the
    generator. We just pull it (to populate response_chunks) and record — and
    snapshot the pc_speaking + announce_speaking gates, which MUST be set while
    we 'play'."""
    if _pc_event_ref[0] is not None:
        _pc_observed["set_during_speak"] = _pc_event_ref[0].is_set()
    if _ann_event_ref[0] is not None:
        _ann_observed["set_during_speak"] = _ann_event_ref[0].is_set()
    for _ in gen:
        pass
    _speak_streaming_calls.append((a, k))


main.speak_streaming = _consuming_speak_streaming


def _new_runner(ui: FakeUI):
    import threading  # noqa: PLC0415
    return main.TurnRunner(_fake_cfg(), ui, threading.Event())


# --- Test 1: construction + empty state -----------------------------------
FakeMemory.summaries_to_return = ["prior summary"]
ui = FakeUI()
runner = _new_runner(ui)
check("new runner -> no active conversation", not runner.has_active_conversation())
check("new runner -> loaded summaries from memory",
      runner._summaries == ["prior summary"])
FakeMemory.summaries_to_return = []


# --- Test 2: a normal silent turn (phone_text) grows history by 2 ---------
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["Hello, ", "sir."]
main.stream_response = stream
_speak_streaming_calls.clear()
ret = runner.process_question("hi", "en", origin="phone_text")
check("normal turn -> returns False (no barge-in)", ret is False)
check("normal turn -> history has user+assistant", len(runner._history) == 2)
check("normal turn -> assistant content assembled from chunks",
      runner._history[1]["content"] == "Hello, sir.")
check("normal turn -> recorded to memory",
      runner._memory.turns == [("hi", "Hello, sir.", "en", None)])
check("normal turn -> jarvis text surfaced", ui.jarvis_texts == ["Hello, sir."])
check("phone_text is silent -> speak_streaming NOT called", _speak_streaming_calls == [])
check("now has an active conversation", runner.has_active_conversation())


# --- Test 3: origin derives `restricted` (phone) --------------------------
check("phone_text turn -> stream_response got restricted=True",
      stream.last_kwargs is not None and stream.last_kwargs.get("restricted") is True)

ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["ok"]
main.stream_response = stream
runner.process_question("hi", "en", origin="console")
check("console turn -> stream_response got restricted=False",
      stream.last_kwargs.get("restricted") is False)


# --- Test 3b: M80 — speaker identity threads through to stream_response ----
# The voice path resolves the recognized speaker and passes it in; it must
# reach stream_response so the per-turn speaker block can be built. Default
# (no speaker) must be None so remote/unrecognized turns add no block.
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["ok"]
main.stream_response = stream
runner.process_question("hola", "es", origin="voice",
                        speaker_name="Bob", speaker_lang="es")
check("voice turn -> stream_response got speaker_name",
      stream.last_kwargs.get("speaker_name") == "Bob")
check("voice turn -> stream_response got speaker_lang",
      stream.last_kwargs.get("speaker_lang") == "es")
# M82: the same speaker must be tagged onto the persisted turn (record_turn).
check("M82: voice turn -> speaker tagged on the recorded turn",
      runner._memory.turns == [("hola", "ok", "es", "Bob")])

ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["ok"]
main.stream_response = stream
runner.process_question("hi", "en", origin="phone_text")
check("turn with no speaker -> stream_response speaker_name defaults None",
      stream.last_kwargs.get("speaker_name") is None)
check("turn with no speaker -> stream_response speaker_lang defaults None",
      stream.last_kwargs.get("speaker_lang") is None)


# --- Test 4: the speak path (console, unmuted) consumes the stream --------
ui = FakeUI(); ui.muted = False
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["spoken ", "reply"]
main.stream_response = stream
_speak_streaming_calls.clear()
runner.process_question("say something", "en", origin="console")
check("console+unmuted -> speak_streaming WAS called (speak path)",
      len(_speak_streaming_calls) == 1)
check("speak path -> history still grew by 2 + assembled",
      len(runner._history) == 2 and runner._history[1]["content"] == "spoken reply")


# --- Test 4b: pc_speaking gate (omni-mic echo fix) ------------------------
# An AUDIBLE turn must SET pc_speaking while it plays (so the voice-capture
# loop suppresses self-audio) and CLEAR it after. A SILENT turn never sets it.
import threading as _threading  # noqa: E402

ui = FakeUI(); ui.muted = False
pc_ev = _threading.Event()
runner = main.TurnRunner(_fake_cfg(), ui, _threading.Event(), pc_speaking=pc_ev)
stream = FakeStream(); stream.chunks = ["hi"]
main.stream_response = stream
_pc_event_ref[0] = pc_ev
_pc_observed["set_during_speak"] = None
runner.process_question("speak up", "en", origin="console")
check("audible turn -> pc_speaking SET while speaking",
      _pc_observed["set_during_speak"] is True)
check("audible turn -> pc_speaking CLEARED after the turn", not pc_ev.is_set())
_pc_event_ref[0] = None

pc_ev_silent = _threading.Event()
runner_s = main.TurnRunner(_fake_cfg(), FakeUI(), _threading.Event(),
                           pc_speaking=pc_ev_silent)
stream = FakeStream(); stream.chunks = ["hi"]
main.stream_response = stream
runner_s.process_question("quietly", "en", origin="phone_text")
check("silent (phone_text) turn -> pc_speaking never set",
      not pc_ev_silent.is_set())


# --- Test 4c: announce_speaking CPU gate (2026-06-01 turn-reply stutter) ---
# An AUDIBLE turn must SET announce_speaking while it plays so the armed PANNs
# (SoundDetector) + YOLO (SecurityWatcher) loops DEFER and don't starve the
# Python-fed TTS path (the live-found stutter: only proactive announces were
# gated, turn replies were not). A SILENT turn never sets it.
ui = FakeUI(); ui.muted = False
ann_ev = _threading.Event()
runner = main.TurnRunner(_fake_cfg(), ui, _threading.Event(),
                         announce_speaking=ann_ev)
stream = FakeStream(); stream.chunks = ["hi"]
main.stream_response = stream
_ann_event_ref[0] = ann_ev
_ann_observed["set_during_speak"] = None
runner.process_question("speak up", "en", origin="console")
check("audible turn -> announce_speaking SET while speaking (CPU loops defer)",
      _ann_observed["set_during_speak"] is True)
check("audible turn -> announce_speaking CLEARED after the turn",
      not ann_ev.is_set())
_ann_event_ref[0] = None

ann_ev_silent = _threading.Event()
runner_s = main.TurnRunner(_fake_cfg(), FakeUI(), _threading.Event(),
                           announce_speaking=ann_ev_silent)
stream = FakeStream(); stream.chunks = ["hi"]
main.stream_response = stream
runner_s.process_question("quietly", "en", origin="phone_text")
check("silent (phone_text) turn -> announce_speaking never set",
      not ann_ev_silent.is_set())


# --- Test 5: empty response pops the orphan user message ------------------
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = []          # nothing comes back
main.stream_response = stream
ret = runner.process_question("hi", "en", origin="phone_text")
check("empty response -> history left empty (orphan user popped)",
      runner._history == [])
check("empty response -> nothing recorded to memory", runner._memory.turns == [])
check("empty response -> returns False", ret is False)


# --- Test 6: an exception speaks/surfaces an apology + pops the user ------
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.raises = RuntimeError("boom")
main.stream_response = stream
_speak_calls.clear()
ret = runner.process_question("hi", "en", origin="phone_text")
check("exception -> history left empty (user popped)", runner._history == [])
check("exception -> apology surfaced to transcript",
      len(ui.jarvis_texts) == 1 and "hiccup" in ui.jarvis_texts[0].lower())
check("exception on a silent turn -> apology NOT spoken aloud", _speak_calls == [])
check("exception -> returns False", ret is False)

# Spanish apology variant
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.raises = RuntimeError("boom")
main.stream_response = stream
runner.process_question("hola", "es", origin="phone_text")
check("exception (es) -> Spanish apology surfaced",
      "Disculpe" in ui.jarvis_texts[0])


# --- Test 7: manual reset boundary seals before the next turn -------------
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["one"]
main.stream_response = stream
runner.process_question("first", "en", origin="phone_text")     # history -> 2
_seal_calls.clear()
runner._reset_event.set()
stream.chunks = ["two"]
runner.process_question("second", "en", origin="phone_text")
check("reset set -> seal fired before the new turn", len(_seal_calls) == 1)
check("reset -> reset_event cleared", not runner._reset_event.is_set())
check("reset -> history holds only the new pair",
      len(runner._history) == 2 and runner._history[0]["content"] == "second")


# --- Test 8: idle boundary seals before the next turn ---------------------
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["one"]
main.stream_response = stream
runner.process_question("first", "en", origin="phone_text")     # history -> 2
runner._last_turn_time = 0.0                                    # far in the past
_seal_calls.clear()
stream.chunks = ["two"]
runner.process_question("second", "en", origin="phone_text")
check("idle elapsed -> seal fired before the new turn", len(_seal_calls) == 1)


# --- Test 9: attachments become a content-block list ----------------------
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["ok"]
main.stream_response = stream
block = {"type": "image", "source": {"data": "x"}}
runner.process_question("what is this", "en",
                        attachments=[block], origin="phone_text")
user_content = runner._history[0]["content"]
check("attachments -> user content is a block list (attachment first)",
      isinstance(user_content, list)
      and user_content[0] is block
      and user_content[-1] == {"type": "text", "text": "what is this"})


# --- Test 10: seal_and_refresh clears history + reloads summaries ---------
ui = FakeUI()
runner = _new_runner(ui)
stream = FakeStream(); stream.chunks = ["x"]
main.stream_response = stream
runner.process_question("hi", "en", origin="phone_text")
FakeMemory.summaries_to_return = ["reloaded"]
runner.seal_and_refresh()
check("seal_and_refresh -> history cleared", runner._history == [])
check("seal_and_refresh -> summaries reloaded", runner._summaries == ["reloaded"])
check("seal_and_refresh -> no active conversation after",
      not runner.has_active_conversation())
FakeMemory.summaries_to_return = []


# --- Restore patched globals (tidy, in case of import reuse) --------------
main._seal_session = _orig_seal
main.stream_response = _orig_stream
main.speak_streaming = _orig_speak_streaming
main.speak = _orig_speak

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
