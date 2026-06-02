r"""Regression test for the Discord bot bridge (2026-06-02).

The access gating and reply-chunking are factored into PURE functions
(should_handle / chunk_message) so the security-critical logic is testable
without a live gateway connection. Also covers the config parsing of the
channel ID + allowlist (fail-closed on garbage). No network, no discord.py
runtime needed (the lib is imported lazily inside the bot thread).

    python scripts/discord_bot_test.py     # exit 0 = pass
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.discord_bot import (  # noqa: E402
    should_handle, chunk_message, is_addressed, strip_bot_mention, thread_name,
)
from src.config import _parse_user_ids, _parse_channel_id  # noqa: E402

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


ME = 111111111111111111
USER_B = 222222222222222222
STRANGER = 999999999999999999
CHAN = 555555555555555555
OTHER_CHAN = 444444444444444444
ALLOW = frozenset({ME, USER_B})


THREAD = 777777777777777777  # a thread's own (channel) id


def _h(**kw) -> bool:
    base = dict(
        author_id=ME, author_is_bot=False, is_webhook=False,
        channel_id=CHAN, parent_id=None, allowed_users=ALLOW, allowed_channel=CHAN,
    )
    base.update(kw)
    return should_handle(**base)


print("\n[group] should_handle — scope + access gating")
check("allowed user, right channel, human -> in scope", _h() is True)
check("a second allowlisted user -> in scope", _h(author_id=USER_B) is True)
check("non-allowlisted user -> out", _h(author_id=STRANGER) is False)
check("wrong channel -> out", _h(channel_id=OTHER_CHAN) is False)
check("bot author -> out (loop guard)", _h(author_is_bot=True) is False)
check("webhook author -> out (reminder-webhook loop guard)",
      _h(is_webhook=True) is False)
check("allowed user but EMPTY allowlist -> out (fail-closed)",
      should_handle(author_id=ME, author_is_bot=False, is_webhook=False,
                    channel_id=CHAN, parent_id=None,
                    allowed_users=frozenset(), allowed_channel=CHAN) is False)
check("right user, wrong channel AND bot -> out",
      _h(channel_id=OTHER_CHAN, author_is_bot=True) is False)
# Threads: a message in a thread has channel_id=thread, parent_id=parent chan.
check("thread UNDER the configured channel -> in scope",
      _h(channel_id=THREAD, parent_id=CHAN) is True)
check("thread under a DIFFERENT channel -> out",
      _h(channel_id=THREAD, parent_id=OTHER_CHAN) is False)


print("\n[group] is_addressed — addressed-only gate")
BOT = 100000000000000000
check("@-mention of the bot -> addressed", is_addressed("hello", BOT, [BOT]) is True)
check("word 'jarvis' -> addressed", is_addressed("jarvis what's the weather", BOT, []) is True)
check("'Hey Jarvis,' (case/punct) -> addressed", is_addressed("Hey Jarvis, status?", BOT, []) is True)
check("trailing 'JARVIS?' -> addressed", is_addressed("what's up JARVIS?", BOT, []) is True)
check("no name, no mention -> NOT addressed", is_addressed("what do you think honey?", BOT, []) is False)
check("someone else mentioned -> NOT addressed", is_addressed("hi", BOT, [STRANGER]) is False)
check("'jarvison' (substring) -> NOT addressed (word boundary)", is_addressed("jarvison", BOT, []) is False)
check("empty -> NOT addressed", is_addressed("", BOT, []) is False)


print("\n[group] strip_bot_mention")
BOT = 100000000000000000
check("strips <@id>", strip_bot_mention(f"<@{BOT}> what's the weather", BOT) == "what's the weather")
check("strips nickname <@!id>", strip_bot_mention(f"<@!{BOT}> hi", BOT) == "hi")
check("leaves the literal name", strip_bot_mention("Jarvis, hello", BOT) == "Jarvis, hello")
check("mention-only -> empty", strip_bot_mention(f"<@{BOT}>", BOT) == "")
check("leaves other mentions", strip_bot_mention(f"<@{BOT}> tell <@999> hi", BOT) == "tell <@999> hi")


print("\n[group] thread_name")
check("short content kept", thread_name("what's the weather") == "what's the weather")
check("collapses whitespace/newlines", thread_name("line one\n  line two") == "line one line two")
check("truncates to limit", len(thread_name("x" * 200)) == 90)
check("empty -> 'Jarvis'", thread_name("") == "Jarvis")


print("\n[group] chunk_message — Discord 2000-char limit")
check("empty -> []", chunk_message("") == [])
check("whitespace-only -> []", chunk_message("   \n  ") == [])
check("short text -> single chunk", chunk_message("hello sir") == ["hello sir"])

short = "x" * 1900
check("exactly at limit -> single chunk", chunk_message(short, 1900) == [short])

# Over the limit across multiple lines -> multiple chunks, each within limit.
many_lines = "\n".join(["line " + str(i) for i in range(800)])
parts = chunk_message(many_lines, 1900)
check("multiline overflow -> splits", len(parts) > 1)
check("every chunk within limit", all(len(p) <= 1900 for p in parts),
      f"max={max(len(p) for p in parts)}")
check("no content lost across chunks",
      sum(p.count("line ") for p in parts) == 800)

# A single line longer than the limit -> hard-sliced.
one_huge = "y" * 5000
hp = chunk_message(one_huge, 1900)
check("single oversized line hard-sliced", len(hp) == 3 and all(len(p) <= 1900 for p in hp))
check("hard-slice preserves length", sum(len(p) for p in hp) == 5000)


print("\n[group] config parsing — allowlist + channel id")
check("two ids parse", _parse_user_ids("111,222") == (111, 222))
check("spaces tolerated", _parse_user_ids(" 111 , 222 ,333") == (111, 222, 333))
check("empty -> ()", _parse_user_ids("") == ())
check("garbage tokens dropped (fail-closed)", _parse_user_ids("abc,111,@me") == (111,))
check("channel id parses", _parse_channel_id("123") == 123)
check("blank channel -> 0", _parse_channel_id("") == 0)
check("garbage channel -> 0", _parse_channel_id("not-a-number") == 0)


print("\n" + "=" * 50)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 50)
sys.exit(1 if FAILED else 0)
