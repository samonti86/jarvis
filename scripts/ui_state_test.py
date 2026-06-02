r"""Regression test for the persisted UI-toggle store (2026-06-02).

WHY THIS EXISTS:
The M69 voice-lock gate was runtime-only — a tray toggle ON didn't survive a
restart (it re-read the env default, which is off). The fix persists the toggle
via src/ui_state.py. This suite locks in that store's contract: get/set round
-trip, default fallback, the "persisted wins over the env default" semantics
used at startup, atomic-write durability, and fail-soft on a corrupt file.

Hermetic: redirects LOCALAPPDATA to a temp dir BEFORE importing the module, and
asserts the store path is isolated, so it never touches real UI state.

    python scripts/ui_state_test.py     # exit 0 = pass
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = tempfile.mkdtemp(prefix="jarvis_ui_state_test_")
os.environ["LOCALAPPDATA"] = _TMP

import src.ui_state as ui_state  # noqa: E402 — must follow the env redirect

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


def _clear() -> None:
    ui_state._store_path().unlink(missing_ok=True)


assert str(ui_state._store_path()).startswith(_TMP), "store not isolated!"


print("\n[group] get_flag defaults")
_clear()
check("missing file -> default False", ui_state.get_flag("speaker_gate") is False)
check("missing file -> explicit default True honored",
      ui_state.get_flag("speaker_gate", True) is True)
check("unknown key -> default", ui_state.get_flag("nope", False) is False)


print("\n[group] set/get round-trip + persistence semantics")
_clear()
ui_state.set_flag("speaker_gate", True)
check("set True then get -> True", ui_state.get_flag("speaker_gate") is True)
check("persisted value IGNORES a False default (the restart case)",
      ui_state.get_flag("speaker_gate", False) is True)
ui_state.set_flag("speaker_gate", False)
check("set False then get -> False (toggle off is sticky)",
      ui_state.get_flag("speaker_gate") is False)
check("persisted False IGNORES a True env default",
      ui_state.get_flag("speaker_gate", True) is False)

# The exact startup expression in main.py: get_flag(name, env_default).
# Fresh install (no file) -> env default; after a toggle -> persisted value.
_clear()
check("startup w/ env default True, no file -> True",
      ui_state.get_flag("speaker_gate", True) is True)
ui_state.set_flag("speaker_gate", False)
check("startup w/ env default True, but toggled OFF -> False (user wins)",
      ui_state.get_flag("speaker_gate", True) is False)


print("\n[group] durability + independence + robustness")
_clear()
ui_state.set_flag("speaker_gate", True)
ui_state.set_flag("other_toggle", False)
check("file actually written", ui_state._store_path().exists())
check("two keys independent (a)", ui_state.get_flag("speaker_gate") is True)
check("two keys independent (b)", ui_state.get_flag("other_toggle", True) is False)

# Corrupt the file -> fail-soft to default, no raise.
with open(ui_state._store_path(), "w", encoding="utf-8") as f:
    f.write("{ this is not json ")
check("corrupt file -> default (fail-soft, no raise)",
      ui_state.get_flag("speaker_gate", False) is False)
# ...and a subsequent set recovers (overwrites the garbage).
ui_state.set_flag("speaker_gate", True)
check("set after corruption recovers the store",
      ui_state.get_flag("speaker_gate") is True)


print("\n" + "=" * 50)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 50)
sys.exit(1 if FAILED else 0)
