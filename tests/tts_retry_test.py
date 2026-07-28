r"""Regression test for the edge-tts bounded-retry helper (2026-06-02).

WHY THIS EXISTS:
A live `NoAudioReceived` flake dropped one sentence from a spoken reply on the
streaming TTS path, which (unlike the one-shot speak() path) had no retry or
fallback. The fix wraps `_fetch_mp3` in `_fetch_mp3_with_retry`. This suite
locks in the helper's contract WITHOUT touching the network: it monkeypatches
the inner `_fetch_mp3` with a scripted sequence and zeroes the backoff.

Contract:
  - returns on first success (no needless retries);
  - retries a raised exception and succeeds if a later attempt does;
  - retries an empty-bytes result (defensive, no-exception flake);
  - re-raises the LAST error after `attempts` tries (so each caller's terminal
    behaviour — drop / pyttsx3 fallback / log — still runs on a real failure);
  - honors a custom `attempts`.

    python tests/tts_retry_test.py     # exit 0 = pass
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.text_to_speech as tts  # noqa: E402

tts._TTS_FETCH_BACKOFF_S = 0  # no real sleeps — keep the suite instant

PASSED = 0
FAILED = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok: {label}")
    else:
        FAILED += 1
        print(f"  FAIL: {label}  {detail}")


class _NoAudio(Exception):
    """Stand-in for edge_tts.exceptions.NoAudioReceived."""


class _FakeFetch:
    """Scripted async replacement for _fetch_mp3. Each script item is either
    `bytes` to return or an `Exception` to raise; the last item repeats if the
    helper calls more times than the script length."""

    def __init__(self, script: list) -> None:
        self.script = script
        self.calls = 0

    async def __call__(self, text: str, voice: str) -> bytes:
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _install(script: list) -> _FakeFetch:
    fake = _FakeFetch(script)
    tts._fetch_mp3 = fake  # _fetch_mp3_with_retry resolves this at call time
    return fake


def _call(attempts: int | None = None):
    if attempts is None:
        return asyncio.run(tts._fetch_mp3_with_retry("hello", "v"))
    return asyncio.run(tts._fetch_mp3_with_retry("hello", "v", attempts=attempts))


print("\n[group] _fetch_mp3_with_retry")

# 1) first-try success — no retries
f = _install([b"AUDIO"])
check("first-try success returns bytes", _call() == b"AUDIO")
check("first-try success makes exactly 1 call", f.calls == 1, f"calls={f.calls}")

# 2) fail twice then succeed (default attempts=3)
f = _install([_NoAudio("no audio"), _NoAudio("no audio"), b"AUDIO"])
check("retries a raised error then succeeds", _call() == b"AUDIO")
check("used all 3 attempts", f.calls == 3, f"calls={f.calls}")

# 3) persistent failure re-raises the last exception
f = _install([_NoAudio("boom")])
raised = None
try:
    _call()
except Exception as exc:  # noqa: BLE001
    raised = exc
check("persistent failure re-raises", isinstance(raised, _NoAudio), repr(raised))
check("persistent failure tried 3 times", f.calls == 3, f"calls={f.calls}")

# 4) empty bytes are treated as a (retryable) failure, then succeed
f = _install([b"", b"AUDIO"])
check("empty-bytes flake is retried then succeeds", _call() == b"AUDIO")
check("empty-then-ok took 2 calls", f.calls == 2, f"calls={f.calls}")

# 5) persistent empty bytes -> raises the synthesized RuntimeError
f = _install([b"", b"", b""])
raised = None
try:
    _call()
except Exception as exc:  # noqa: BLE001
    raised = exc
check("persistent empty raises RuntimeError",
      isinstance(raised, RuntimeError) and "no audio bytes" in str(raised),
      repr(raised))

# 6) custom attempts honored (stop early)
f = _install([_NoAudio("x")])
try:
    _call(attempts=2)
except _NoAudio:
    pass
check("custom attempts=2 makes exactly 2 calls", f.calls == 2, f"calls={f.calls}")


print("\n" + "=" * 50)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 50)
sys.exit(1 if FAILED else 0)
