"""External notifications — Discord webhook + SMTP email (M38).

The bluff path of M35 (deterrent + evidence-snapshot save) is theatrical:
it scares intruders but only the user-on-their-PC sees the result. This
module layers *real* alerts on top — when the deterrent fires, push the
evidence JPEG out through one or both channels so the user gets a phone
notification.

Two channels, fired in parallel from src/security.py:
- Discord webhook (instant, native image embed, free, single-URL secret)
- SMTP email (works without any mobile-app dependency; Gmail App Password
  setup required for the most common provider; comma-separated recipient
  list supported)

Defensive contract: this module NEVER raises into the SecurityWatcher
thread. The deterrent path has already done its job (announce + save
local evidence) by the time we're called — a failed Discord post or
SMTP send is a real loss but mustn't break the security flow. All
failures log to stderr (-> jarvis.log) for audit.
"""

from __future__ import annotations

import json
import smtplib
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import httpx


# 10s is generous — Discord webhook POSTs typically complete in <500 ms,
# but we'd rather absorb a slow upload than timeout in the middle.
_DISCORD_TIMEOUT_SECONDS = 10.0

# Discord's hard limit on message `content` is 2000 chars (a 400 over it). One
# constant for every sender so the clamp can't drift (send_discord_message used
# to clamp 1900 while send_discord_photo clamped 2000, both docstringed "2000").
_DISCORD_CONTENT_LIMIT = 2000


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


def send_discord_photo(
    webhook_url: str,
    content: str,
    image_bytes: bytes,
    image_filename: str = "snapshot.jpg",
) -> bool:
    """M72 — post caller-supplied text WITH a photo attachment to a Discord
    webhook. The generic photo counterpart to send_discord_message (text only)
    and send_discord_alert (fixed security text): used for the multimodal
    acoustic alert ("I heard X — here's what I see" + the camera frame).

    Returns True on HTTP 2xx, False on any failure. Never raises — callers
    fire-and-forget. `content` is clamped to Discord's 2000-char limit.
    """
    if not webhook_url or not image_bytes:
        return False
    payload = {"content": (content or "")[:_DISCORD_CONTENT_LIMIT], "username": "Jarvis"}
    files = {
        "payload_json": (None, json.dumps(payload), "application/json"),
        "files[0]": (image_filename, image_bytes, "image/jpeg"),
    }
    try:
        resp = httpx.post(webhook_url, files=files, timeout=_DISCORD_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        print(f"[notify] discord photo POST failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False
    if resp.status_code >= 400:
        print(f"[notify] discord photo returned HTTP {resp.status_code}: "
              f"{resp.text[:200]}", file=sys.stderr)
        return False
    print(f"[notify] discord photo sent ({len(image_bytes)} bytes, "
          f"HTTP {resp.status_code})", file=sys.stderr)
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


def send_discord_message(webhook_url: str, content: str) -> bool:
    """Post a plain-text message to a Discord webhook.

    The generic counterpart to send_discord_alert, which is hardwired to the
    M35 intruder deterrent (fixed text + an image attachment). M56's homelab
    monitor needs to push arbitrary up/down alert lines, so this is the
    no-attachment, caller-supplied-text path.

    Returns True on HTTP 2xx, False on any failure. Never raises — callers
    fire-and-forget. `content` is clamped to Discord's 2000-char limit.
    """
    if not webhook_url or not content:
        return False

    payload = {"content": content[:_DISCORD_CONTENT_LIMIT], "username": "Jarvis"}
    try:
        resp = httpx.post(
            webhook_url, json=payload, timeout=_DISCORD_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        print(f"[notify] discord message failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False

    if resp.status_code >= 400:
        print(
            f"[notify] discord message returned HTTP {resp.status_code}: "
            f"{resp.text[:200]}",
            file=sys.stderr,
        )
        return False

    print(f"[notify] discord message sent (HTTP {resp.status_code})",
          file=sys.stderr)
    return True


# 20s — Gmail SMTP usually completes in <2s, but we'd rather absorb a slow
# TLS handshake on a flaky network than timeout during send. Still well
# under the watcher's tolerance for a backgrounded task.
_SMTP_TIMEOUT_SECONDS = 20.0


def send_email_alert(
    smtp_host: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    smtp_to: str,
    image_bytes: bytes,
    image_filename: str = "evidence.jpg",
    when: datetime | None = None,
) -> bool:
    """Send the security-alert email with the evidence JPEG attached.

    Returns True on success, False on any failure (missing creds,
    auth rejection, network error, etc.). Never raises.

    `smtp_to` may be a single address or a comma-separated list. The
    function splits and trims; empty entries are dropped silently.

    Transport selection by port:
      - 465  → implicit SSL (smtplib.SMTP_SSL)
      - else → STARTTLS upgrade on a plaintext SMTP socket
    This matches the standard convention used by Gmail, Outlook, Fastmail.
    """
    if not (smtp_host and smtp_username and smtp_password and smtp_to):
        # Any missing piece → email path is not configured. Silent skip
        # (caller already gates on smtp_username, but defensive double-check
        # so direct callers don't trip an obscure SMTP error).
        return False

    recipients = [addr.strip() for addr in smtp_to.split(",") if addr.strip()]
    if not recipients:
        return False

    when = when or datetime.now()
    time_str = when.strftime("%I:%M %p on %b %d").lstrip("0")

    msg = EmailMessage()
    msg["Subject"] = f"Jarvis Security Alert — {time_str}"
    msg["From"] = smtp_username
    # Comma-joined list is the standards-correct way to put N addresses in
    # a single To: header (RFC 5322 §3.6.3). Gmail / Outlook handle it fine.
    msg["To"] = ", ".join(recipients)
    msg.set_content(
        f"Security Alert — {time_str}\n\n"
        "A person was detected in the monitored space and did not authenticate "
        "within the 15-second challenge window. Evidence attached.\n\n"
        "— Jarvis"
    )
    if image_bytes:
        msg.add_attachment(
            image_bytes,
            maintype="image",
            subtype="jpeg",
            filename=image_filename,
        )

    # ssl.create_default_context() picks up the system CA bundle and enforces
    # certificate verification + hostname matching. Gmail's cert is fine; if
    # the user ever points this at a self-signed SMTP server, the failure is
    # loud (cert verify error in the log) rather than silently insecure.
    tls_context = ssl.create_default_context()

    try:
        if smtp_port == 465:
            # Implicit TLS — socket is encrypted from the first byte.
            with smtplib.SMTP_SSL(
                smtp_host,
                smtp_port,
                timeout=_SMTP_TIMEOUT_SECONDS,
                context=tls_context,
            ) as smtp:
                smtp.login(smtp_username, smtp_password)
                smtp.send_message(msg, to_addrs=recipients)
        else:
            # Plaintext socket → EHLO → STARTTLS → re-EHLO → AUTH → send.
            # smtplib.SMTP.starttls() handles the upgrade dance; we just
            # supply the context for cert verification.
            with smtplib.SMTP(
                smtp_host, smtp_port, timeout=_SMTP_TIMEOUT_SECONDS,
            ) as smtp:
                smtp.starttls(context=tls_context)
                smtp.login(smtp_username, smtp_password)
                smtp.send_message(msg, to_addrs=recipients)
    except (OSError, smtplib.SMTPException, ssl.SSLError) as exc:
        # OSError covers connect refused / DNS failure / timeout; SMTPException
        # covers auth rejection, recipient refused, malformed response;
        # SSLError covers handshake / cert issues. Anything else propagates
        # — but per the never-raises contract, defensively catch too.
        print(
            f"[notify] email send failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False
    except Exception as exc:  # noqa: BLE001 — defensive ceiling
        print(
            f"[notify] email send unexpected error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False

    print(
        f"[notify] email alert sent ({len(image_bytes)} bytes evidence, "
        f"to {len(recipients)} recipient(s))",
        file=sys.stderr,
    )
    return True
