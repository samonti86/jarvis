"""screen_snapshot — give Jarvis eyes on the digital world.

Captures the user's primary monitor and returns it as a Messages-API image
content block, so Claude *sees* the screen and can answer in voice ("that's
a 502 from your reverse proxy, sir — the upstream service is unreachable").
The SRE companion to M29's camera_snapshot: same plumbing, different sensor.

Why this exists (M30, post-M29):
- The "Jarvis, what's on my screen?" / "what does this error mean?" loop is
  the highest-leverage vision use case for a Technical-Support-Engineer
  user. A screenshot of a terminal trace is faster to caption than to read.
- It's almost free: PIL.ImageGrab is in Pillow, which is already a dep
  (pystray pulls it for tray-icon rendering). M29 built the image-block
  plumbing and `_ClientTool.execute` already accepts `str | list[dict]`,
  so this slots in clean.

Design notes:
- **Open-on-demand, never persistent.** Same privacy posture as the camera —
  capture, encode, release. No screen-recording daemon running in the
  background; the user invokes it explicitly each time.
- **Primary monitor only for v1.** PIL.ImageGrab.grab() defaults to the
  primary monitor; `all_screens=True` stitches the whole virtual desktop
  into one wide image, but for multi-monitor users that produces a 3840+px
  panorama that Anthropic downscales aggressively (text becomes unreadable).
  Single monitor keeps text legible. A `monitor` param can be added later
  without breaking callers — same forward-compatible shape as cameras.py's
  `camera` slot.
- **PNG, not JPEG.** Screen content is text + UI — JPEG's chroma subsampling
  blurs small text noticeably even at quality 90. PNG is lossless and the
  file-size cost (~300-800 KB at 1568px long edge) is well under Anthropic's
  5 MB inline cap. Different default than cameras.py (JPEG q80) on purpose:
  webcam frames are photographic, screen frames are typographic.
- **Resize to ≤1568px long edge.** Anthropic's vision sweet spot — going
  larger gets downscaled server-side anyway, so we save bandwidth by doing
  it locally with a higher-quality filter (LANCZOS) than the server picks.
- **Defensive contract:** never raises. Locked workstation / no display /
  Pillow missing / capture failure all become readable strings Claude relays
  apologetically. Same shape as cameras.py.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime

from src.attachments import image_block


# --- Anthropic tool definition ---------------------------------------------
SCREEN_SNAPSHOT_TOOL = {
    "name": "screen_snapshot",
    "description": (
        "Capture a screenshot of the user's primary monitor and look at it — "
        "this is how you SEE what they're doing on their PC. Call it whenever "
        "the user asks what's on their screen, asks you to read or explain "
        "something they're looking at (an error message, a terminal trace, a "
        "config file, a webpage, a chat), wants help with a UI (\"where do I "
        "click?\"), or asks for a second opinion on what they're seeing. "
        "Each call captures a fresh frame. After looking, describe or explain "
        "what's there concisely and conversationally — like glancing at their "
        "screen over their shoulder, not narrating a photo. Don't announce "
        "that you're taking a screenshot; just look and tell them. Pairs with "
        "camera_snapshot (physical world); this one is the digital world."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


# Anthropic vision sweet spot — going past ~1568 on the long edge gets
# downscaled server-side anyway. Doing it locally with LANCZOS gives a
# higher-quality result than the server's default resampler.
_MAX_LONG_EDGE = 1568

# Below this mean pixel value (0-255) the capture is effectively black — almost
# always a locked workstation, an off display, or a permissions failure (e.g.
# capturing from a session with no desktop). Same threshold as cameras.py.
_BLACK_FRAME_MEAN = 8.0


def execute_screen_snapshot(params: dict) -> str | list[dict]:
    """Grab a screenshot of the primary monitor. Returns a list of content
    blocks (image + caption) on success, or a readable error string. Never
    raises."""
    try:
        from PIL import ImageGrab, Image  # type: ignore
    except ImportError:
        return (
            "Screen-capture support isn't installed (Pillow missing). "
            "Run: pip install Pillow"
        )

    try:
        img = ImageGrab.grab()
        if img is None:
            return (
                "Couldn't capture the screen — the workstation may be locked "
                "or there's no active display session."
            )

        # Black-frame guard — locked-screen captures sometimes come back as a
        # uniform black image rather than failing outright. Cheap: convert a
        # downscaled grayscale copy and inspect the mean.
        probe = img.convert("L").resize((64, 64))
        if sum(probe.getdata()) / (64 * 64) < _BLACK_FRAME_MEAN:
            return (
                "The screen capture came back black — the workstation is "
                "probably locked or the display is off."
            )

        w, h = img.size
        if max(w, h) > _MAX_LONG_EDGE:
            img.thumbnail((_MAX_LONG_EDGE, _MAX_LONG_EDGE), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png = buf.getvalue()
    except Exception as exc:  # noqa: BLE001 — defensive; this tool must never raise
        print(f"[screen] snapshot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return f"Screen-capture error: {exc}"

    final_w, final_h = img.size
    print(
        f"[screen] snapshot ok: source={w}x{h} sent={final_w}x{final_h} {len(png)} bytes",
        file=sys.stderr,
    )
    when = datetime.now().strftime("%I:%M %p").lstrip("0")  # "2:15 PM" — cross-platform
    return [
        image_block(png, "image/png"),
        {"type": "text", "text": f"Screenshot of the primary monitor, captured just now ({when})."},
    ]
