"""Crash-safe, concurrency-safe atomic file writes.

Several small JSON stores (reminders, predictions, the quiet-hours catch-up,
the UI-toggle state) persist with a temp-file + os.replace dance. os.replace
makes the *rename* atomic — an interrupted write can't leave a half-written
file in the target's place — but two gaps remained, both fixed here in one
place so every store inherits the fix:

1. DURABILITY (fsync). os.replace does not guarantee the new file's DATA is on
   disk before the rename commits. On a hard power loss (this machine has no
   UPS — see the project notes) the directory entry can end up pointing at a
   file whose blocks were never flushed: a zero-length or torn store that
   survives as the "committed" copy. flush()+fsync() before the replace makes
   the data durable on the platter first.

2. CONCURRENCY (unique temp name). The old pattern used a single fixed temp
   path (`store.json.tmp`). If two threads ever save the same store at once
   (the background prediction cycle vs. the briefing surfacing its results),
   they'd write the SAME temp file and corrupt it. A unique temp per write
   removes that: the two replaces race harmlessly (each leaves a complete
   file; last writer wins, which is benign for these stores).

Stdlib only — kept dependency-free like logfile.py.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def _replace_with_retry(src: str, dst: Path, *, attempts: int = 25,
                        delay: float = 0.01) -> None:
    """os.replace, retried on the transient Windows ERROR_ACCESS_DENIED /
    ERROR_SHARING_VIOLATION (surfaced as PermissionError) that fires when
    another thread's concurrent replace — or an antivirus / Search indexer —
    briefly holds the destination. POSIX rename is atomic and never hits this,
    so the loop succeeds on the first try there. After the last attempt the
    error propagates (the caller fail-soft-wraps if it must not crash)."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Durably and atomically overwrite `path` with `text`.

    Writes to a UNIQUE sibling temp file, flushes + fsyncs it to disk, then
    os.replace()s it over the target (atomic on Windows and POSIX). After a
    crash the target is either the complete old file or the complete new one —
    never torn or zero-length. Safe for two threads to call concurrently on the
    same path (each gets its own temp; the replaces race benignly).

    The temp lives in the SAME directory as the target so os.replace is a
    same-filesystem rename (atomic); a cross-device temp would fall back to a
    non-atomic copy. On any failure the temp is cleaned up and the error
    propagates (callers that must never raise wrap this themselves).
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp_name, path)
    except BaseException:
        # Don't leave a stray temp behind on error (full disk, permission, etc.).
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
