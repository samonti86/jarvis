r"""Regression test for the omni-mic echo fix (2026-05-29).

Exercises `_record_until_silence`'s `suppress_event` abort in isolation — the
load-bearing new safety: while the PC is speaking (a turn reply or a proactive
announce), the always-on voice-capture loop must DISCARD what it hears instead
of transcribing Jarvis's own voice as a question.

Uses a fake AudioSession (silence chunks, no real mic / sleep) so the VAD loop
runs deterministically and fast.

    venv\Scripts\python.exe scripts\stt_suppress_test.py     # exit 0 = pass
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.speech_to_text import _record_until_silence  # noqa: E402

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


class FakeSession:
    """Minimal AudioSession stand-in: returns silence chunks, counts reads,
    and can set an event on the Nth read (to simulate the PC starting to speak
    mid-capture). 1 s chunks @ 16 kHz keep the pre-speech window to ~5 reads."""

    def __init__(self, set_event_on_read: int | None = None,
                 event: threading.Event | None = None) -> None:
        self.sample_rate = 16000
        self.chunk_samples = 16000          # 1 s/chunk → ~5 pre-speech reads
        self.read_count = 0
        self._set_on = set_event_on_read
        self._event = event

    def read(self) -> np.ndarray:
        self.read_count += 1
        if (self._set_on is not None and self._event is not None
                and self.read_count == self._set_on):
            self._event.set()
        return np.zeros(self.chunk_samples, dtype=np.int16)  # silence


# --- Test 1: event already set -> abort before reading at all -------------
ev = threading.Event()
ev.set()
sess = FakeSession()
out = _record_until_silence(sess, suppress_event=ev)
check("pre-set suppress -> empty array (aborted)", out.size == 0)
check("pre-set suppress -> session never read", sess.read_count == 0)


# --- Test 2: event set mid-capture -> abort promptly ----------------------
ev = threading.Event()
sess = FakeSession(set_event_on_read=2, event=ev)   # PC 'starts speaking' at read 2
out = _record_until_silence(sess, suppress_event=ev)
check("mid-capture suppress -> empty array (aborted)", out.size == 0)
check("mid-capture suppress -> aborted right after the event set (~2 reads)",
      sess.read_count == 2)


# --- Test 3: no suppression -> normal no-speech path still returns empty ---
# (silence only -> pre-speech window elapses -> empty, exactly as before;
# proves the new param didn't change the default behaviour.)
sess = FakeSession()
out = _record_until_silence(sess, suppress_event=None)
check("no suppress + silence -> empty (no-speech timeout, unchanged)",
      out.size == 0)
check("no suppress -> ran the full pre-speech window (read > 0)",
      sess.read_count > 0)

# A clear (unset) event behaves like None — no spurious abort.
ev = threading.Event()  # never set
sess = FakeSession()
out = _record_until_silence(sess, suppress_event=ev)
check("clear suppress event -> no spurious abort (ran the window)",
      out.size == 0 and sess.read_count > 0)


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
