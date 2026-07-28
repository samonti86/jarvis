r"""Arc-reactor renderer — the shared visual for the HUD and console (M98).

WHY PIL AND NOT THE CANVAS:
Tk's `create_oval` / `create_arc` are ALIASED. Every ring and arc came out with
stair-stepped edges, which is most of why the old reactor read as flat clip-art
rather than a piece of hardware. Pillow (already a dependency, via pystray)
supersamples and downsamples, so the same shapes come out smooth, and it can do
a real glow falloff — dozens of stacked translucent discs — which the canvas
cannot express at all.

WHY FRAMES ARE PRE-RENDERED, NOT DRAWN LIVE:
Measured before building it, because this project has a scar here. Rendering
one 224px reactor costs 9-25 ms depending on supersample; the HUD redraws at
30 fps, so drawing live would burn **55-74% of a core, continuously**, forever.
That is precisely the always-on CPU load that starved the real-time audio
threads in M67/M68 and produced the armed-mode TTS stutter.

So a full rotation cycle is rendered ONCE per (size, colour) into a list of
frames, and the widget just swaps a PhotoImage each tick. Steady-state cost is
an array index. The one-time cost is paid on a DAEMON THREAD at startup, and
until it finishes the caller keeps drawing its old canvas primitives — so the
UI never blocks and a slow machine simply gets the plain look for a second.

_FRAMES is deliberately small: these arcs rotate slowly, and at 12 frames the
cycle reads as smooth motion while keeping the warm cost and memory modest.
"""

from __future__ import annotations

import math
import sys
import threading

_FRAMES = 12          # frames per full rotation cycle
_SUPERSAMPLE = 3      # 3 is the knee: visually ~4, ~35% cheaper to warm

# The HUD's chroma key. Transparency there is BINARY — Windows punches out
# pixels of exactly this colour and leaves every other pixel fully opaque, so a
# soft alpha glow is impossible: it composites to an opaque near-black disc,
# which is invisible on a dark desktop but an ugly blob over a light window.
# Anything that composites to within _KEY_CUT of the key is therefore forced to
# alpha 0 — a hard edge exactly where the glow stopped being perceptible.
_KEY_RGB = (1, 1, 1)
_KEY_CUT = 13

# (size, color, frames) -> [PIL.Image]. Warmed off-thread, read on the UI thread;
# dict get/set is atomic under the GIL so no lock is needed for correctness.
_cache: dict[tuple, list] = {}
_warming: set[tuple] = set()
# Keys whose render RAISED. Failure is recorded and never retried, because the
# caller is an animation tick: frames() misses 20 times a second, and each miss
# calls warm(). Without this, one broken render (no Pillow, an OOM) becomes a
# permanent 20-thread-per-second storm. The cost of giving up is that the
# widget draws its vector fallback until restart — which is the correct trade.
_failed: set[tuple] = set()
_lock = threading.Lock()


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def _render(size: int, rgb: tuple[int, int, int], phase: float):
    """One frame. `phase` is 0..1 through the rotation cycle."""
    from PIL import Image, ImageDraw  # noqa: PLC0415 — lazy; keeps import light

    S = _SUPERSAMPLE
    W = size * S
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = W / 2
    R = W / 2
    hot = (125, 243, 255)
    ring = (36, 74, 94)
    t = phase * 360

    # Glow: stacked translucent discs with a quadratic falloff. This is the
    # part the canvas simply cannot do. Kept tight and bright rather than wide
    # and faint, because _KEY_CUT will clip the faint tail off anyway.
    for i in range(24, 0, -1):
        f = i / 24
        r = R * (0.44 + 0.24 * f)
        a = int(40 * (1 - f) ** 1.8)
        if a > 0:
            d.ellipse([c - r, c - r, c + r, c + r], fill=(*rgb, a))

    def arcs(radius, width, col, n, span, deg):
        step = 360 / n
        for k in range(n):
            a0 = deg + k * step
            d.arc([c - radius, c - radius, c + radius, c + radius],
                  a0, a0 + span, fill=(*col, 255), width=max(1, int(width * S)))

    # Three counter-rotating rings.
    #
    # The rotation amounts are not free. A ring of n arcs is symmetric every
    # 360/n degrees, so it maps onto ITSELF partway through the cycle unless
    # the numbers are chosen against it. Writing q for how many symmetry
    # periods a ring turns per cycle, frame i repeats frame 0 as soon as
    # i*q is a multiple of _FRAMES — so the cycle only yields _FRAMES distinct
    # frames when gcd(q, _FRAMES) == 1.
    #
    # The first draft used q = 2, 6, 18 against _FRAMES = 12 and got SIX
    # distinct frames, two of them rendered twelve times: half the warm cost
    # and half the memory bought nothing, and the animation ran at double the
    # intended speed. q = 1, -5, 7 are all coprime with 12, so every frame is
    # distinct and the loop still closes seamlessly. Pinned by reactor_test.
    arcs(R * 0.94, 1.5, ring, 2, 104, t * 0.5)                        # q = +1
    arcs(R * 0.84, 2.0, _lerp(ring, rgb, 0.55), 3, 34, -t * (5 / 3))  # q = -5
    arcs(R * 0.72, 1.2, _lerp(ring, rgb, 0.30), 6, 8, t * (7 / 6))    # q = +7

    # Graduated tick ring — static, reads as instrumentation. 36 ticks, not 60:
    # at 190 px, 60 ticks land ~6 px apart and read as noise rather than a scale.
    rt0, rt1 = R * 0.62, R * 0.67
    for k in range(36):
        ang = math.radians(k * 10)
        long = (k % 3 == 0)
        r0 = rt0 - (R * 0.03 if long else 0)
        col = _lerp(ring, rgb, 0.5) if long else ring
        d.line([c + r0 * math.cos(ang), c + r0 * math.sin(ang),
                c + rt1 * math.cos(ang), c + rt1 * math.sin(ang)],
               fill=(*col, 255), width=max(1, int((1.6 if long else 1.0) * S)))

    # Core: radial ramp out from a near-white centre.
    r_core = R * 0.30
    for i in range(int(r_core), 0, -1):
        f = i / r_core
        d.ellipse([c - i, c - i, c + i, c + i],
                  fill=(*_lerp(hot, rgb, f ** 0.6), 255))
    r_hot = R * 0.11
    d.ellipse([c - r_hot, c - r_hot, c + r_hot, c + r_hot], fill=(*hot, 255))

    out = img.resize((size, size), Image.LANCZOS)
    return _clip_to_key(out)


def _clip_to_key(img):
    """Force every pixel that would composite to ~the chroma key to alpha 0.

    Without this the glow's faint tail lands a few units off the key colour, so
    Windows keeps it — an opaque near-black disc the width of the whole halo.
    """
    import numpy as np  # noqa: PLC0415 — lazy; a hard dep, but only needed here
    from PIL import Image  # noqa: PLC0415

    arr = np.asarray(img, dtype=np.float32)
    key = np.array(_KEY_RGB, dtype=np.float32)
    a = arr[:, :, 3:4] / 255.0
    # What Windows will actually see, then how far that is from the key.
    composited = arr[:, :, :3] * a + key * (1.0 - a)
    faint = np.abs(composited - key).max(axis=2) <= _KEY_CUT
    out = arr.astype(np.uint8)
    out[:, :, 3][faint] = 0
    return Image.fromarray(out, "RGBA")


def frames(size: int, color: str) -> list | None:
    """Rendered cycle for (size, colour), or None if not warmed yet.

    Never renders on the calling thread — a 200 ms stall on the Tk thread would
    be a visible hitch, and this is decoration. Returns None until ready and the
    caller falls back to its own drawing.
    """
    key = (size, color, _FRAMES)
    got = _cache.get(key)
    if got is not None:
        return got
    warm(size, color)
    return None


def warm(size: int, color: str) -> None:
    """Render (size, colour) in the background if it isn't already queued."""
    key = (size, color, _FRAMES)
    with _lock:
        if key in _cache or key in _warming or key in _failed:
            return
        _warming.add(key)

    def _work() -> None:
        try:
            rgb = _hex_to_rgb(color)
            out = [_render(size, rgb, i / _FRAMES) for i in range(_FRAMES)]
            _cache[key] = out
        except Exception as exc:  # noqa: BLE001 — decoration must never crash
            with _lock:
                _failed.add(key)
            print(f"[reactor] render failed ({size}px {color}): {exc} "
                  f"— falling back to vector drawing", file=sys.stderr)
        finally:
            with _lock:
                _warming.discard(key)

    threading.Thread(target=_work, name=f"reactor-warm-{size}",
                     daemon=True).start()


def warm_all(size: int, colors) -> None:
    """Pre-warm every state colour so the first state change is already smooth."""
    for c in colors:
        warm(size, c)
