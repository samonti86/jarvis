"""External notifications — currently Discord webhooks (M38).

The bluff path of M35 (deterrent + evidence-snapshot save) is theatrical:
it scares intruders but only the user-on-their-PC sees the result. This
module layers a *real* alert on top — when the deterrent fires, post the
evidence JPEG to a Discord webhook so the user gets a push on their phone
via the Discord mobile app.

Why Discord webhooks (vs email, Pushover, etc.):
- Free forever (no per-message cost, no monthly cap that matters).
- Instant delivery (sub-second typical).
- Image attachments native — Discord auto-embeds the snapshot.
- No bot framework required, no OAuth, no API tokens — just a webhook URL.
- Phone push works out of the box if the user has Discord mobile installed.
- The webhook URL is the only secret (lives in `.env`, gitignored).

Defensive contract: this module NEVER raises into the SecurityWatcher
thread. The deterrent path has already done its job (announce + save
local evidence) by the time we're called — a failed Discord post is
a real loss but mustn't break the security flow. All failures log to
stderr (-> jarvis.log) for audit.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import httpx


# 10s is generous — Discord webhook POSTs typically complete in <500 ms,
# but we'd rather absorb a slow upload than timeout in the middle.
_DISCORD_TIMEOUT_SECONDS = 10.0


def send_discord_alert(
    webhook_url: str,
    image_bytes: bytes,
    image_filename: str = "evidence.jpg",
    when: datetime | None = None,
) -> bool:
    """Post the security-alert message + evidence image to a Discord webhook.

    Returns True if the post succeeded (HTTP 2xx), False on any failure
    (no network, invalid URL, Discord rate-limited, HTTP non-2xx, etc.).
    Never raises — callers can fire-and-forget.

    `image_bytes`: JPEG bytes (typically the evidence frame saved by
    SecurityWatcher._save_evidence). If empty, the message still sends
    but without an attachment.

    `when`: timestamp of the triggering detection (NOT the moment this
    notification fires — they can differ by a few seconds depending on
    how quickly the deterrent path executes). Defaults to now.
    """
    if not webhook_url:
        return False

    when = when or datetime.now()
    time_str = when.strftime("%I:%M %p on %b %d").lstrip("0")

    # The message format. Markdown-aware: Discord renders **bold** and the
    # 🚨 emoji at the start makes the push notification preview eye-catching
    # on the user's lock screen.
    content = (
        f"🚨 **Security Alert** · {time_str}\n\n"
        f"A person was detected in the monitored space and did not authenticate "
        f"within the 15-second challenge window. Evidence attached."
    )
    payload = {
        "content": content,
        "username": "Jarvis",
        # Discord clamps usernames to 80 chars and forbids "discord" etc.
        # — "Jarvis" is safe.
    }

    # Discord's webhook expects multipart/form-data when an attachment is
    # present. The payload_json field carries the message; files[N] carry
    # the attachments (Discord references them as attachment://<filename>
    # in embeds if you want to inline them — we just use the default
    # auto-embed behaviour which works fine for one image).
    files = {
        "payload_json": (None, json.dumps(payload), "application/json"),
    }
    if image_bytes:
        files["files[0]"] = (image_filename, image_bytes, "image/jpeg")

    try:
        resp = httpx.post(
            webhook_url,
            files=files,
            timeout=_DISCORD_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        print(f"[notify] discord POST failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False

    if resp.status_code >= 400:
        # Body is usually a small JSON error. Log first 200 chars for
        # debugging without flooding the log if Discord returns HTML.
        print(
            f"[notify] discord returned HTTP {resp.status_code}: "
            f"{resp.text[:200]}",
            file=sys.stderr,
        )
        return False

    print(
        f"[notify] discord alert sent ({len(image_bytes)} bytes evidence, "
        f"HTTP {resp.status_code})",
        file=sys.stderr,
    )
    return True


def send_discord_alert_for_path(
    webhook_url: str,
    image_path: Path,
    when: datetime | None = None,
) -> bool:
    """Convenience wrapper: read the JPEG from disk and post.

    Useful when the caller has already saved the evidence file and just
    has the path — saves a round-trip of passing bytes in memory. Read
    failure is treated like any other notification failure: log + return
    False, don't raise.
    """
    if not webhook_url or image_path is None:
        return False
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        print(f"[notify] couldn't read evidence file {image_path}: {exc}",
              file=sys.stderr)
        return False
    return send_discord_alert(
        webhook_url, image_bytes, image_filename=image_path.name, when=when,
    )
