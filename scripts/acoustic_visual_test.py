"""M72 — tests for the multimodal acoustic alert (look + describe on a sound).

Hermetic: no real Anthropic call (we test the fail-soft guards + the pure
prompt builder), no network (send_discord_photo guards), no audio hardware
(SoundDetector constructs without starting capture). The _fire -> on_visual_alert
wiring is exercised with a recording hook.

    python scripts/acoustic_visual_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.notifications import send_discord_photo  # noqa: E402
from src.sound_detector import SoundDetector, _DEFAULT_RULES  # noqa: E402
from src.vision_describe import build_prompt, describe_scene  # noqa: E402


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


# --- build_prompt (pure) ---------------------------------------------------

check("prompt weaves in the heard event", "doorbell" in build_prompt("doorbell"))
check("prompt replaces underscores in the event name",
      "smoke alarm" in build_prompt("smoke_alarm"))
check("prompt asks for ONE concise sentence", "ONE concise" in build_prompt("knock"))
check("empty event falls back to 'a sound'", "a sound" in build_prompt(""))
check("non-str-ish event doesn't crash the builder",
      isinstance(build_prompt(None), str))  # type: ignore[arg-type]


# --- describe_scene fail-soft guards (no live API) -------------------------

check("describe_scene: empty jpeg -> None",
      describe_scene(b"", "doorbell", api_key="k", model="m") is None)
check("describe_scene: missing api key -> None",
      describe_scene(b"\xff\xd8\xff", "doorbell", api_key="", model="m") is None)


# --- send_discord_photo guards (no network) --------------------------------

check("send_discord_photo: blank webhook -> False",
      send_discord_photo("", "hi", b"\xff\xd8\xff") is False)
check("send_discord_photo: empty image -> False",
      send_discord_photo("https://example.test/hook", "hi", b"") is False)


# --- _fire -> on_visual_alert wiring ---------------------------------------

fired = threading.Event()
seen: list[str] = []


def _recorder(event_name: str) -> None:
    seen.append(event_name)
    fired.set()


det = SoundDetector(announce=lambda _t: None, on_visual_alert=_recorder)
rule = _DEFAULT_RULES[0]  # e.g. doorbell
det._fire(rule, 0.9, 0.1)
fired.wait(timeout=3.0)
# join the AcousticVisual thread(s) so we don't leak into other tests
for t in threading.enumerate():
    if t.name == "AcousticVisual":
        t.join(timeout=2.0)
check("_fire spawns the visual hook with the rule name",
      seen == [rule.name])


# A hook that raises must be swallowed (never breaks the inference loop).
det2 = SoundDetector(
    announce=lambda _t: None,
    on_visual_alert=lambda _e: (_ for _ in ()).throw(RuntimeError("boom")),
)
try:
    det2._safe_visual("glass_breaking")
    check("a raising visual hook is swallowed", True)
except Exception:
    check("a raising visual hook is swallowed", False)

# No hook wired ⇒ _fire still works (text-only, pre-M72 behaviour).
det3 = SoundDetector(announce=lambda _t: None)  # no on_visual_alert
try:
    det3._fire(rule, 0.9, 0.1)
    check("_fire with no visual hook is fine (text-only path)", True)
except Exception:
    check("_fire with no visual hook is fine (text-only path)", False)


# --- summary --------------------------------------------------------------

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
