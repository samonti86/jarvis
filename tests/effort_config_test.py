r"""Regression test for the output_config.effort plumbing (M89).

WHY THIS EXISTS:
`effort` controls how hard the model thinks AND acts per turn. Two ways to get
it wrong are both silent-ish and both expensive:

  1. Sending it to a model that rejects it. Haiku 4.5 returns a 400 for
     output_config.effort, and the background jobs (session summariser,
     prediction miner) run on Haiku. A stray effort there doesn't degrade —
     it takes those jobs off the air entirely.
  2. A typo in .env. An invalid effort value is a hard API error on EVERY
     turn, so `JARVIS_VOICE_EFFORT=meduim` would silence the assistant
     completely. That must fail soft to the default, per the project's
     "an optional component logs and degrades" contract.

Neither is caught by py_compile or by importing the module — `import src.llm`
succeeded happily while `os` was unimported, because the failure was at call
time. So the wiring is asserted here, with a fake client: no network, no key.

    venv\Scripts\python.exe scripts\effort_config_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.llm as llm  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")
        if detail:
            print(f"          {detail}")


# --- 1. model guard -------------------------------------------------------
for model, expected in [
    ("claude-sonnet-5", True),
    ("claude-opus-5", True),
    ("claude-sonnet-4-6", True),
    ("claude-haiku-4-5-20251001", False),   # 400s on effort
    ("claude-haiku-4-5", False),
    ("", False),
]:
    got = llm._supports_effort(model)
    check(f"_supports_effort({model or '(empty)'}) is {expected}", got == expected)


# --- 2. env parsing fails soft -------------------------------------------
def with_env(value: str | None):
    key = "JARVIS_TEST_EFFORT"
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return llm._env_effort(key, "medium")


check("unset -> default", with_env(None) == "medium")
check("blank -> default", with_env("") == "medium")
check("valid value honoured", with_env("low") == "low")
check("uppercase normalised", with_env("HIGH") == "high")
check("surrounding whitespace tolerated", with_env("  xhigh  ") == "xhigh")
check("typo falls back to default (does NOT 400 every turn)",
      with_env("meduim") == "medium")
check("junk falls back to default", with_env("turbo") == "medium")
os.environ.pop("JARVIS_TEST_EFFORT", None)


# --- 3. the wiring: what actually reaches the API ------------------------
class _Stop(Exception):
    """Raised by the fake to end the turn once kwargs are captured."""


class _FakeStream:
    def __init__(self, captured: dict, kwargs: dict) -> None:
        captured.update(kwargs)

    def __enter__(self):
        raise _Stop

    def __exit__(self, *a):
        return False


def capture_kwargs(model: str, engineer: bool = False,
                   effort: str | None = None) -> dict:
    """Run one turn against a fake client; return the kwargs it would send."""
    captured: dict = {}

    class FakeMessages:
        def stream(self, **kwargs):
            return _FakeStream(captured, kwargs)

    class FakeClient:
        messages = FakeMessages()

    orig = llm._get_client
    llm._get_client = lambda _key: FakeClient()
    try:
        gen = llm.stream_response(
            api_key="test-key",
            messages=[{"role": "user", "content": "hello"}],
            model=model,
            engineer_mode=engineer,
            effort=effort,
        )
        try:
            for _ in gen:
                pass
        except _Stop:
            pass
        except Exception:
            pass
    finally:
        llm._get_client = orig
    return captured


k = capture_kwargs("claude-sonnet-5")
check("sonnet voice turn sends output_config.effort",
      k.get("output_config", {}).get("effort") == llm._VOICE_EFFORT,
      f"got {k.get('output_config')!r}")
check("sonnet voice turn keeps thinking disabled",
      k.get("thinking") == {"type": "disabled"}, f"got {k.get('thinking')!r}")

k = capture_kwargs("claude-sonnet-5", engineer=True)
check("engineer turn sends the engineer effort",
      k.get("output_config", {}).get("effort") == llm._ENGINEER_EFFORT,
      f"got {k.get('output_config')!r}")
check("engineer turn uses adaptive thinking",
      k.get("thinking") == {"type": "adaptive"}, f"got {k.get('thinking')!r}")

k = capture_kwargs("claude-haiku-4-5-20251001")
check("HAIKU turn omits output_config entirely (would 400)",
      "output_config" not in k, f"got {k.get('output_config')!r}")

k = capture_kwargs("claude-sonnet-5", effort="xhigh")
check("explicit effort argument overrides the default",
      k.get("output_config", {}).get("effort") == "xhigh",
      f"got {k.get('output_config')!r}")

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
