"""Discord bot bridge — a private Discord channel as a two-way Jarvis client.

The webhook (src/notifications.py) is one-way (Jarvis → channel). To RECEIVE
messages you need a bot: a token + a persistent Gateway WebSocket + the
privileged Message Content intent. This module runs discord.py's asyncio loop
in its OWN daemon thread — exactly the shape src/remote_console.py uses — and
bridges to the brain through the SAME seam as the phone:

  inbound : on_message → (channel + allowlist + non-bot filter) → on_text(text,
            reply) which main.py wires onto text_queue with origin="discord".
  outbound: a PER-TURN reply sink posts the answer back to the originating
            channel. NOT a JarvisUI broadcast sink — a PC/voice/phone turn must
            never echo into the shared channel where the household can read it.

Why a bot, not the webhook, and why the gateway (not HTTP interactions): the
gateway is an OUTBOUND connection, so it needs no inbound port / Tailscale /
TLS — it works from anywhere the PC has internet, and your household can talk
to Jarvis off-network for free.

Security: Discord is an internet-relayed, MULTI-human surface, so it's a
RESTRICTED origin (main.py derives restricted=True → no system/shell/file/code
tools). Two access gates: the configured CHANNEL (coarse) and the user-ID
ALLOWLIST (fine). Bot/webhook authors are ignored, so the bot can't react to
its own replies or the reminder webhook (no loop). Availability is optional +
graceful (the lazy/AVAILABLE pattern): no token ⇒ it never starts; any
connect/import error is logged, never fatal to Jarvis.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import Callable, Iterable

# Discord's hard per-message ceiling is 2000 chars; stay under it with margin.
_DISCORD_MSG_LIMIT = 1900


def should_handle(
    *,
    author_id: int,
    author_is_bot: bool,
    is_webhook: bool,
    channel_id: int,
    allowed_users: frozenset[int],
    allowed_channel: int,
) -> bool:
    """Pure decision: should this inbound message be routed to the brain?

    True only when it's a human (not a bot/webhook — the loop guard), in the
    configured channel, from an allowlisted user. Factored out so the gating
    is unit-testable without a live gateway connection."""
    if author_is_bot or is_webhook:
        return False
    if channel_id != allowed_channel:
        return False
    return author_id in allowed_users


def chunk_message(text: str, limit: int = _DISCORD_MSG_LIMIT) -> list[str]:
    """Split a reply into Discord-sized pieces (≤ limit chars), preferring line
    boundaries; hard-slice any single line that's itself too long. Always
    returns at least one chunk for non-empty text."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    buf = ""
    for line in text.split("\n"):
        # A single oversized line: flush the buffer, then hard-slice the line.
        while len(line) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(line[:limit])
            line = line[limit:]
        # Would appending this line overflow the buffer? Flush first.
        candidate = f"{buf}\n{line}" if buf else line
        if len(candidate) > limit:
            chunks.append(buf)
            buf = line
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


class DiscordBot:
    """Runs a discord.py client on a daemon thread; bridges to text_queue.

    main.py wires set_on_text(fn) where fn(text, reply) enqueues the turn with
    origin="discord" and `reply` as the per-turn text sink. The brain calls
    `reply(answer)` once the turn completes; we marshal that onto the bot's
    loop and post it (chunked) back to the originating channel."""

    def __init__(
        self,
        token: str,
        channel_id: int,
        allowed_user_ids: Iterable[int],
    ) -> None:
        self._token = token
        self._channel_id = int(channel_id)
        self._allowed: frozenset[int] = frozenset(int(u) for u in allowed_user_ids)
        self._on_text: Callable[[str, Callable[[str], None]], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def set_on_text(
        self, fn: Callable[[str, Callable[[str], None]], None]
    ) -> None:
        """Wire the inbound handler. fn(text, reply): enqueue the turn; the
        brain later calls reply(answer) to post back to the channel."""
        self._on_text = fn

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="DiscordBot", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        # Imported here (not at module top) so a missing/broken discord.py
        # never breaks `import main` — the optional-component contract.
        try:
            import discord  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[discord] discord.py unavailable — bot disabled: {exc}",
                  file=sys.stderr)
            return

        intents = discord.Intents.default()
        intents.message_content = True  # privileged — must be enabled in the portal
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready() -> None:  # noqa: D401
            self._loop = asyncio.get_running_loop()
            print(
                f"[discord] connected as {client.user} — channel "
                f"{self._channel_id}, {len(self._allowed)} allowed user(s)",
                file=sys.stderr,
            )

        @client.event
        async def on_message(message: "discord.Message") -> None:
            if not should_handle(
                author_id=message.author.id,
                author_is_bot=bool(getattr(message.author, "bot", False)),
                is_webhook=message.webhook_id is not None,
                channel_id=message.channel.id,
                allowed_users=self._allowed,
                allowed_channel=self._channel_id,
            ):
                return
            content = (message.content or "").strip()
            if not content or self._on_text is None:
                return

            # Per-turn reply sink bound to THIS channel. Called by the brain on
            # the text_input_loop thread → marshal onto the bot's loop.
            def _reply(answer: str, _ch=message.channel) -> None:
                self._post(_ch, answer)

            try:
                self._on_text(content, _reply)
            except Exception as exc:  # noqa: BLE001 — never break the gateway loop
                print(f"[discord] on_text raised: {exc}", file=sys.stderr)

        try:
            # log_handler=None: don't let discord.py install its own root log
            # handler (we have our own stderr logging).
            client.run(self._token, log_handler=None)
        except Exception as exc:  # noqa: BLE001 — a dead bot must not kill Jarvis
            print(f"[discord] bot stopped: {exc}", file=sys.stderr)

    def _post(self, channel: object, answer: str) -> None:
        """Thread-safe: post `answer` (chunked) back to `channel`. Called from
        the brain's worker thread; marshals onto the bot's loop. Fail-soft."""
        if self._loop is None:
            print("[discord] reply dropped — bot loop not ready", file=sys.stderr)
            return
        chunks = chunk_message(answer)
        if not chunks:
            return

        async def _send() -> None:
            for piece in chunks:
                await channel.send(piece)  # type: ignore[attr-defined]

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception as exc:  # noqa: BLE001
            print(f"[discord] post failed: {exc}", file=sys.stderr)
