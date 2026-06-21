r"""Regression test for config's defensive numeric env parsing (2026-06-21).

WHY THIS EXISTS:
config.load() parsed ~8 tunables (SMTP_PORT, JARVIS_REMOTE_PORT, the thresholds,
etc.) with a bare int()/float(). A single typo'd value in .env (e.g.
SMTP_PORT=587x) raised ValueError at import and took the WHOLE assistant down —
the opposite of the "graceful fallback, never crash the listening loop" rule.
_int_env/_float_env fall back to the default on a missing/garbage value. This
suite locks that: valid parses, blank -> default, garbage -> default (no raise),
and an end-to-end load() with a poisoned env still returns a usable Config.

    python scripts/config_env_test.py     # exit 0 = pass
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config  # noqa: E402

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


def _set(name: str, val: str | None) -> None:
    if val is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = val


print("\n[group] _int_env")
_set("X_TEST_INT", "42")
check("valid int parses", config._int_env("X_TEST_INT", 7) == 42)
_set("X_TEST_INT", None)
check("missing -> default", config._int_env("X_TEST_INT", 7) == 7)
_set("X_TEST_INT", "   ")
check("blank/whitespace -> default", config._int_env("X_TEST_INT", 7) == 7)
_set("X_TEST_INT", "587x")
check("garbage -> default (no raise)", config._int_env("X_TEST_INT", 587) == 587)
_set("X_TEST_INT", "3.5")
check("float string is not an int -> default", config._int_env("X_TEST_INT", 7) == 7)
_set("X_TEST_INT", " 100 ")
check("surrounding whitespace tolerated", config._int_env("X_TEST_INT", 7) == 100)
_set("X_TEST_INT", None)


print("\n[group] _float_env")
_set("X_TEST_FLOAT", "0.72")
check("valid float parses", config._float_env("X_TEST_FLOAT", 0.5) == 0.72)
_set("X_TEST_FLOAT", None)
check("missing -> default", config._float_env("X_TEST_FLOAT", 0.5) == 0.5)
_set("X_TEST_FLOAT", "abc")
check("garbage -> default (no raise)", config._float_env("X_TEST_FLOAT", 0.5) == 0.5)
_set("X_TEST_FLOAT", "9")
check("int string parses as float", config._float_env("X_TEST_FLOAT", 0.5) == 9.0)
_set("X_TEST_FLOAT", None)


print("\n[group] load() survives a poisoned .env (the real regression)")
poison = {
    "SMTP_PORT": "587x",
    "JARVIS_REMOTE_PORT": "not-a-port",
    "JARVIS_SPEAKER_THRESHOLD": "high",
    "WAKE_WORD_THRESHOLD": "",
    "JARVIS_PRESENCE_ARM_DELAY": "soon",
}
saved = {k: os.environ.get(k) for k in poison}
for k, v in poison.items():
    os.environ[k] = v
try:
    cfg = config.load()  # must NOT raise
    check("load() returns a Config despite garbage values", cfg is not None)
    check("SMTP_PORT fell back to 587", cfg.smtp_port == 587)
    check("remote_port fell back to 8765", cfg.remote_port == 8765)
    check("speaker_threshold fell back to 0.60", cfg.speaker_threshold == 0.60)
    check("wake_word_threshold blank -> 0.5 default", cfg.wake_word_threshold == 0.5)
    check("presence_arm_delay fell back to 60.0", cfg.presence_arm_delay == 60.0)
except Exception as exc:  # noqa: BLE001
    check("load() returns a Config despite garbage values", False, f"raised {exc!r}")
finally:
    for k, v in saved.items():
        _set(k, v)


print("\n" + "=" * 50)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 50)
sys.exit(1 if FAILED else 0)
