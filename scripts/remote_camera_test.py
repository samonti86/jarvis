"""M71 — tests for the per-origin tool claw-back (Discord webcam access).

The security-critical core: camera_snapshot is re-allowed for origin="discord"
ONLY, while every other restricted tool stays denied for Discord and the phone
origins get nothing back. Covered here without touching real hardware (we never
execute camera_snapshot — we assert the GATE decision via _effective_deny and
the denial path via the denied-tool combos that return before executing).

    python scripts/remote_camera_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import (  # noqa: E402
    _RESTRICTED_ALLOW_BY_ORIGIN,
    _RESTRICTED_DENY,
    _effective_deny,
    _execute_client_tool,
    build_system_prompt,
)


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


# --- _effective_deny: the per-origin claw-back ----------------------------

discord_deny = _effective_deny("discord")
check("discord: camera_snapshot is CLAWED BACK (not denied)",
      "camera_snapshot" not in discord_deny)
check("discord: screen_snapshot STILL denied",
      "screen_snapshot" in discord_deny)
check("discord: system_control STILL denied",
      "system_control" in discord_deny)
check("discord: run_code STILL denied", "run_code" in discord_deny)
check("discord: update_jarvis STILL denied", "update_jarvis" in discord_deny)
check("discord: read_local_file STILL denied",
      "read_local_file" in discord_deny)
check("discord: claw-back removes EXACTLY one tool",
      discord_deny == (_RESTRICTED_DENY - {"camera_snapshot"}))

for origin in ("phone_text", "phone_voice", "", "unknown"):
    d = _effective_deny(origin)
    check(f"{origin or 'empty'}: camera_snapshot STILL denied (no claw-back)",
          "camera_snapshot" in d)
    check(f"{origin or 'empty'}: deny == full _RESTRICTED_DENY",
          d == _RESTRICTED_DENY)

check("only 'discord' has a claw-back entry",
      set(_RESTRICTED_ALLOW_BY_ORIGIN) == {"discord"})
check("the claw-back is exactly {camera_snapshot}",
      _RESTRICTED_ALLOW_BY_ORIGIN["discord"] == frozenset({"camera_snapshot"}))


# --- _execute_client_tool gate (hermetic: only DENIED combos, which return
#     the denial string BEFORE executing — never touches hardware) ----------

r = _execute_client_tool("system_control", {}, restricted=True, origin="discord")
check("discord + system_control -> denied (string, not executed)",
      isinstance(r, str) and "isn't available" in r)

r = _execute_client_tool("camera_snapshot", {}, restricted=True, origin="phone_text")
check("phone_text + camera_snapshot -> denied (camera not clawed back for phone)",
      isinstance(r, str) and "isn't available" in r)

r = _execute_client_tool("screen_snapshot", {}, restricted=True, origin="discord")
check("discord + screen_snapshot -> denied (only camera was clawed back)",
      isinstance(r, str) and "isn't available" in r)


# --- system prompt addendum variant ---------------------------------------

discord_prompt = build_system_prompt(remote_restricted=True, restricted_origin="discord")
check("discord prompt advertises the webcam (camera_snapshot named)",
      "camera_snapshot" in discord_prompt)
check("discord prompt does NOT forbid the camera",
      "screen/camera capture" not in discord_prompt)
check("discord prompt still forbids screen capture",
      "screen capture" in discord_prompt)

phone_prompt = build_system_prompt(remote_restricted=True, restricted_origin="phone_text")
check("phone prompt uses the base addendum (forbids screen/camera)",
      "screen/camera capture" in phone_prompt)
check("phone prompt does NOT invite camera use",
      "look through the webcam" not in phone_prompt)

unrestricted = build_system_prompt(remote_restricted=False, restricted_origin="")
check("unrestricted (PC) prompt has NEITHER restricted addendum",
      "Remote session" not in unrestricted)


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
