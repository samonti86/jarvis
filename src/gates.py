"""CountedEvent — a threading.Event whose set/clear NEST (2026-07-02 QA).

The two cooperative speech gates (`pc_speaking` — the omni-mic self-capture
suppressor, and `announce_speaking` — the armed-CPU-burst gate) are raised by
MULTIPLE unserialized speakers: every spoken turn reply (TurnRunner), the
interpreter relay, speak_line confirmations, and the proactive Announcer
thread. A plain Event can't express overlap: if a reminder announce fires two
seconds into a twenty-second turn reply, the announce's `finally` clears the
gate while the reply is still playing — dropping the echo suppression for the
rest of the reply (the 2026-05-29 self-capture bug re-opened) and letting the
armed PANNs/YOLO loops resume mid-speech (the M68 stutter re-exposed).

CountedEvent keeps the exact Event API so every existing producer/consumer
call site works unchanged: set() increments a counter (the flag goes up on
0→1), clear() decrements (the flag drops only on the LAST clear). Defensive
clears with no matching set clamp at zero rather than going negative — the
same "clear in a finally, even on an error path" idiom the callers use today
stays safe.

Consumers (`is_set`, `wait`) are inherited untouched.
"""

from __future__ import annotations

import threading


class CountedEvent(threading.Event):
    """An Event where set()/clear() calls nest via a counter."""

    def __init__(self) -> None:
        super().__init__()
        self._count_lock = threading.Lock()
        self._count = 0

    def set(self) -> None:  # noqa: A003 — matching threading.Event's API
        with self._count_lock:
            self._count += 1
            super().set()

    def clear(self) -> None:
        with self._count_lock:
            # Clamp: a defensive clear with no matching set must not push the
            # counter negative (it would make a later set/clear pair no-op).
            self._count = max(0, self._count - 1)
            if self._count == 0:
                super().clear()

    def force_clear(self) -> None:
        """Reset regardless of nesting — for shutdown/abnormal paths only."""
        with self._count_lock:
            self._count = 0
            super().clear()
