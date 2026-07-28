"""M84 — tests for the ambient HUD's pure helpers.

The overlay itself is a Windows graphics surface (transparent, click-through)
that can't be exercised headless — it's live-validated by eye. What we CAN pin
here is the colour math the animation depends on (a bad _dim/_blend would make
the orb render wrong or crash on an out-of-range channel) plus the env gate.

    python tests/hud_test.py    # exit 0 = all pass, 1 = any failed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hud import _blend, _dim, _hex_to_rgb, _rgb_to_hex, is_enabled  # noqa: E402

_passed = 0
_failed = 0


def check(label: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


# --- hex<->rgb round trip --------------------------------------------------
check("hex_to_rgb parses cyan", _hex_to_rgb("#22d3ee") == (0x22, 0xd3, 0xee))
check("hex_to_rgb tolerates no-hash", _hex_to_rgb("22d3ee") == (0x22, 0xd3, 0xee))
check("rgb_to_hex formats lowercase 6-digit", _rgb_to_hex((34, 211, 238)) == "#22d3ee")
check("rgb_to_hex round-trips", _rgb_to_hex(_hex_to_rgb("#7df3ff")) == "#7df3ff")


# --- _dim ------------------------------------------------------------------
check("dim by 1.0 is unchanged", _dim("#22d3ee", 1.0) == "#22d3ee")
check("dim by 0.0 is black", _dim("#22d3ee", 0.0) == "#000000")
# Use all-even channels so halving is exact (no rounding ambiguity).
check("dim by 0.5 halves channels", _dim("#20d2ee", 0.5) == _rgb_to_hex((0x10, 0x69, 0x77)))
# Out-of-range factor must CLAMP, never produce an invalid colour (the orb's
# pulse can briefly push >1.0; a crash there would kill the animation).
out = _dim("#22d3ee", 2.0)
check("dim by >1.0 clamps to valid hex (no overflow)", out == "#22d3ee" or len(out) == 7)
check("dim by >1.0 each channel <= 255",
      all(c <= 255 for c in _hex_to_rgb(_dim("#ffffff", 3.0))))


# --- _blend ----------------------------------------------------------------
check("blend t=0 is colour A", _blend("#000000", "#ffffff", 0.0) == "#000000")
check("blend t=1 is colour B", _blend("#000000", "#ffffff", 1.0) == "#ffffff")
check("blend t=0.5 is the midpoint", _blend("#000000", "#ffffff", 0.5) == _rgb_to_hex((127.5, 127.5, 127.5)))
check("blend clamps t<0 to A", _blend("#101010", "#202020", -1.0) == "#101010")
check("blend clamps t>1 to B", _blend("#101010", "#202020", 2.0) == "#202020")


# --- is_enabled (default OFF) ----------------------------------------------
_old = os.environ.get("JARVIS_HUD")
os.environ["JARVIS_HUD"] = "1"
check("is_enabled True when =1", is_enabled() is True)
os.environ["JARVIS_HUD"] = "on"
check("is_enabled True when =on", is_enabled() is True)
os.environ["JARVIS_HUD"] = "0"
check("is_enabled False when =0", is_enabled() is False)
os.environ.pop("JARVIS_HUD", None)
check("is_enabled False when unset (default OFF)", is_enabled() is False)
if _old is not None:
    os.environ["JARVIS_HUD"] = _old


print(f"\n{_passed} passed, {_failed} failed")
sys.exit(0 if _failed == 0 else 1)
