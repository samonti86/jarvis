"""Discord bot bridge — a private Discord channel as a two-way Jarvis client.

The webhook (src/notifications.py) is one-way (Jarvis → channel). To RECEIVE
messages you need a bot: a token + a persistent Gateway WebSocket + the
privileged Message Content intent. This module runs discord.py's asyncio loop
in its OWN daemon thread — exactly the shape src/remote_console.py uses — and
bridges to the brain through the SAME seam as the phone:

  inbound : on_message → (scope + allowlist + non-bot + ADDRESSED filter) →
            on_text(text, reply) which main.py wires onto text_queue with
            origin="discord".
  outbound: a PER-TURN reply sink posts the answer back in a THREAD off the
            triggering message (or in the existing thread). NOT a JarvisUI
            broadcast sink — a PC/voice/phone turn must never echo into the
            shared channel where the household can read it.

Addressed-only (2026-06-02): in a channel where the household also chats with
each other, the bot replies ONLY when explicitly addressed — the message
@-mentions the bot OR contains the word "jarvis". The one exception is inside a
thread the bot itself started: there, every message is for Jarvis, so no
re-addressing is needed (a natural back-and-forth).

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
import re
import sys
import threading
from typing import Callable, Iterable

# Discord's hard per-message ceiling is 2000 chars; stay under it with margin.
_DISCORD_MSG_LIMIT = 1900

# Word-boundary match on the wake name, case-insensitive — "Jarvis", "hey
# jarvis,", "JARVIS?" all hit; "jarvison" does not.
_NAME_RE = re.compile(r"\bjarvis\b", re.IGNORECASE)


def should_handle(
    *,
    author_id: int,
    author_is_bot: bool,
    is_webhook: bool,
    channel_id: int,
    parent_id: int | None,
    allowed_users: frozenset[int],
    allowed_channel: int,
) -> bool:
    """Pure decision: is this message IN SCOPE for Jarvis?

    True only when it's a human (not a bot/webhook — the loop guard), from an
    allowlisted user, in the configured channel OR a thread hanging off it
    (`parent_id` == the configured channel; None for non-thread messages).
    Note: 'in scope' ≠ 'reply' — see is_addressed for the addressed-only gate.
    Factored out so gating is unit-testable without a live gateway."""
    if author_is_bot or is_webhook:
        return False
    in_scope = channel_id == allowed_channel or parent_id == allowed_channel
    if not in_scope:
        return False
    return author_id in allowed_users


def is_addressed(content: str, bot_user_id: int, mention_ids: Iterable[int]) -> bool:
    """True if the message explicitly addresses Jarvis — an @-mention of the
    bot, or the word 'jarvis' anywhere. Lets the household chat freely in the
    channel without Jarvis butting in on every line."""
    if bot_user_id in set(mention_ids):
        return True
    return _NAME_RE.search(content or "") is not None


def strip_bot_mention(content: str, bot_user_id: int) -> str:
    """Remove the bot's own @-mention token (<@id> / <@!id>) so the LLM sees
    clean text ('what's the weather') not raw mention markup. Leaves the
    literal name 'Jarvis' in place (harmless, natural address)."""
    return re.sub(rf"<@!?{bot_user_id}>", "", content or "").strip()


def thread_name(content: str, limit: int = 90) -> str:
    """A Discord thread title (≤100 chars) from the triggering message; falls
    back to 'Jarvis' for an empty/mention-only prompt."""
    one_line = " ".join((content or "").split())
    return one_line[:limit] if one_line else "Jarvis"


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
        # M71: on_text now carries TWO per-turn sinks — reply_text (post the
        # answer) and reply_image (buffer a webcam frame for the same threaded
        # reply). reply_image exists because camera_snapshot is clawed back for
        # origin="discord" so the household can check the home while away.
        self._on_text: "Callable[[str, Callable[[str], None], Callable[[bytes, str], None]], None] | None" = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # Thread IDs the bot itself created — inside these, every message is for
        # Jarvis, so the addressed-only gate is skipped (natural follow-ups).
        self._bot_threads: set[int] = set()

    def set_on_text(
        self,
        fn: "Callable[[str, Callable[[str], None], Callable[[bytes, str], None]], None]",
    ) -> None:
        """Wire the inbound handler. fn(text, reply_text, reply_image): enqueue
        the turn; the brain calls reply_text(answer) to post back, and (M71)
        reply_image(bytes, media_type) to buffer a captured webcam frame so it's
        attached to that same threaded reply."""
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
            is_thread = isinstance(message.channel, discord.Thread)
            parent_id = message.channel.parent_id if is_thread else None
            if not should_handle(
                author_id=message.author.id,
                author_is_bot=bool(getattr(message.author, "bot", False)),
                is_webhook=message.webhook_id is not None,
                channel_id=message.channel.id,
                parent_id=parent_id,
                allowed_users=self._allowed,
                allowed_channel=self._channel_id,
            ):
                return
            if self._on_text is None:
                return

            # Addressed-only gate — reply only when explicitly addressed
            # (@-mention or the word "jarvis"), EXCEPT inside a thread the bot
            # started, where every message is already for Jarvis.
            in_bot_thread = is_thread and message.channel.id in self._bot_threads
            bot_id = client.user.id if client.user else 0
            if not in_bot_thread and not is_addressed(
                message.content, bot_id, [m.id for m in message.mentions]
            ):
                return

            # Strip the bot's own @-mention so the LLM sees clean text.
            content = strip_bot_mention(message.content, bot_id)
            if not content:  # mention/name only, nothing to answer
                return

            # Per-turn reply sink bound to THIS message. Called by the brain on
            # the text_input_loop thread → marshal onto the bot's loop. Passes
            # the whole message (so _post can decide thread-vs-channel) plus the
            # CLEANED text as the thread title (so the title reads "what is the
            # weather", not the raw "<@id> what is the weather" mention markup).
            # M71 — per-turn webcam-frame buffer. The brain calls reply_image
            # mid-turn (when camera_snapshot fires) BEFORE reply_text at the
            # end, so by the time we post the answer this list holds any photo
            # to attach. Buffering (vs posting the image immediately) keeps the
            # photo + the description in ONE thread — posting separately would
            # race _post's thread creation and split them.
            pending_images: list[tuple[bytes, str]] = []

            def _reply_image(img: bytes, media_type: str) -> None:
                pending_images.append((img, media_type))

            def _reply(answer: str, _msg=message, _title=content) -> None:
                self._post(_msg, answer, _title, pending_images)

            try:
                self._on_text(content, _reply, _reply_image)
            except Exception as exc:  # noqa: BLE001 — never break the gateway loop
                print(f"[discord] on_text raised: {exc}", file=sys.stderr)

        try:
            # log_handler=None: don't let discord.py install its own root log
            # handler (we have our own stderr logging).
            client.run(self._token, log_handler=None)
        except Exception as exc:  # noqa: BLE001 — a dead bot must not kill Jarvis
            print(f"[discord] bot stopped: {exc}", file=sys.stderr)

    def _post(self, message: object, answer: str, title_hint: str = "",
              images: "list[tuple[bytes, str]] | None" = None) -> None:
        """Thread-safe: post `answer` (chunked) back as a THREADED reply to
        `message`. Called from the brain's worker thread; marshals onto the
        bot's loop. Fail-soft.

        Target resolution: if the triggering message is already in a thread,
        reply there; otherwise create a thread off the message (titled from
        `title_hint` — the cleaned text, no mention markup — and set to
        AUTO-ARCHIVE after 1h idle so it self-clears from the channel sidebar;
        recording its id so follow-ups skip the addressed-gate) and reply
        inside it. If the bot lacks thread permissions, fall back to an inline
        channel reply and log what to grant — degrade, don't break."""
        if self._loop is None:
            print("[discord] reply dropped — bot loop not ready", file=sys.stderr)
            return
        chunks = chunk_message(answer)
        if not chunks and not images:
            return

        async def _send() -> None:
            import io  # noqa: PLC0415
            import discord  # noqa: PLC0415 — cached; needed for Thread/Forbidden/File
            try:
                channel = message.channel  # type: ignore[attr-defined]
                if isinstance(channel, discord.Thread):
                    target = channel
                else:
                    try:
                        # auto_archive_duration=60 (minutes): the thread drops
                        # off the channel sidebar ~1h after the last message
                        # (still readable under "Archived Threads"; a new reply
                        # un-archives it). Keeps #general uncluttered.
                        target = await message.create_thread(  # type: ignore[attr-defined]
                            name=thread_name(title_hint),
                            auto_archive_duration=60,
                        )
                        self._bot_threads.add(target.id)
                    except discord.Forbidden:
                        print(
                            "[discord] no thread permission — replying inline. "
                            "Grant the bot 'Create Public Threads' + 'Send "
                            "Messages in Threads' to enable threaded replies.",
                            file=sys.stderr,
                        )
                        target = channel
                # M71 — build the webcam attachment(s) once and attach to the
                # FIRST message so the photo + Jarvis's description land together
                # in the same thread (not split across separate posts).
                files = []
                for i, (img, media_type) in enumerate(images or []):
                    ext = "png" if "png" in (media_type or "") else "jpg"
                    suffix = "" if i == 0 else str(i)
                    files.append(
                        discord.File(io.BytesIO(img), filename=f"snapshot{suffix}.{ext}")
                    )
                if not chunks:
                    # Image(s) only (empty text reply) — post the photo alone.
                    if files:
                        await target.send(files=files)
                    return
                for idx, piece in enumerate(chunks):
                    if idx == 0 and files:
                        await target.send(content=piece, files=files)
                    else:
                        await target.send(piece)
            except Exception as exc:  # noqa: BLE001
                print(f"[discord] send failed: {exc}", file=sys.stderr)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception as exc:  # noqa: BLE001
            print(f"[discord] post failed: {exc}", file=sys.stderr)
