"""Manual size-based rotation for the Jarvis logfile.

Called from jarvis.pyw and main.setup_logging() at startup, BEFORE the log
file is opened. We rotate when the file exceeds max_bytes, keeping up to
`keep` historical backups (.1, .2, .3, ...). The oldest gets dropped.

Why not logging.handlers.RotatingFileHandler? We don't use the logging
module — we redirect stdout/stderr directly to a file (TeeStream in
console mode, plain file under pythonw). RotatingFileHandler only knows
about logging-module records. Manual rotation at startup is also safer
on Windows: renaming a file with an open handle fails.

Stdlib only — this module is imported by jarvis.pyw before the wrong-
interpreter re-exec resolves, so it must run on either system Python or
the venv's. No numpy/anthropic/etc. allowed in the import chain.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path


DEFAULT_MAX_BYTES = 5_000_000   # 5 MB — months of moderate use
DEFAULT_KEEP = 3                # log + 3 backups = ~20 MB max footprint


def _line_stamp() -> str:
    """The per-line prefix: a compact, sortable, second-resolution wall clock.
    Matches the bracketed style of the '--- Jarvis started ... ---' markers and
    the Plex MCP child's own log lines, so the whole file reads uniformly."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")


def stamp_chunk(data: str, at_line_start: bool, stamp: str) -> tuple[str, bool]:
    """Pure core of TimestampStream.write: prefix `stamp` at the start of every
    NON-EMPTY line in `data`. Returns (stamped_text, at_line_start_after).

    Separated from the stream and the clock so the line-splitting logic is
    deterministically testable. Stateful by necessity — print() emits a line as
    two calls ("text" then "\\n") and a traceback arrives as one multi-line
    write, so we carry whether the next char opens a fresh line across calls.
    A lone newline (blank line) is left UNstamped so the log's blank separators
    stay clean.
    """
    if not data:
        return "", at_line_start
    out: list[str] = []
    for ch in data:
        if at_line_start and ch != "\n":
            out.append(stamp)
            at_line_start = False
        out.append(ch)
        if ch == "\n":
            at_line_start = True
    return "".join(out), at_line_start


class TimestampStream:
    """Wraps a writable text stream and prefixes every emitted line with a
    wall-clock timestamp. The Jarvis log is a redirected stdout/stderr of bare
    print() calls, so historically only session markers carried a time — which
    made a long-running soak impossible to reason about temporally (two lines
    being file-adjacent did NOT mean they happened close in time, e.g. across an
    overnight shutdown gap). One stamp per line closes that.

    fileno() deliberately DELEGATES to the wrapped stream: in production a
    subprocess inherits the parent's stderr by file descriptor (the Plex MCP
    child's logs reach jarvis.log this way), and that must keep working. Such
    output bypasses stamping — fine, it carries its own format, exactly as
    before this wrapper existed. Only Python-level writes (.write) get stamped.
    """

    def __init__(self, stream) -> None:
        self._stream = stream
        self._at_line_start = True
        # 2026-07-02 QA: many threads print() through the ONE shared instance,
        # and print() emits a line as two write() calls ("text", then "\n").
        # Unlocked, a racing thread could observe at_line_start=False mid-pair
        # and emit its whole line UNstamped — defeating the per-line timestamp
        # the wrapper exists to guarantee (the log is the forensic artifact
        # for soak debriefs). Lock the state update + underlying write.
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        if not data:
            return 0
        with self._lock:
            text, self._at_line_start = stamp_chunk(
                data, self._at_line_start, _line_stamp()
            )
            self._stream.write(text)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._stream.fileno()


def _backup_path(log_path: Path, n: int) -> Path:
    """jarvis.log -> jarvis.log.1, jarvis.log.2, ..."""
    return log_path.with_name(f"{log_path.name}.{n}")


def rotate_if_needed(
    log_path: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    keep: int = DEFAULT_KEEP,
) -> bool:
    """Rotate `log_path` if it exceeds max_bytes. Returns True if a rotation
    actually happened. Idempotent and best-effort: any OS error is swallowed
    so a stuck rotation never blocks startup."""
    try:
        if not log_path.exists() or log_path.stat().st_size < max_bytes:
            return False
    except OSError:
        return False

    # Shift from oldest to newest so we never overwrite a backup we still
    # need. Drop the very oldest (would-be .keep+1) first, then .keep -> drop,
    # .keep-1 -> .keep, ..., .1 -> .2, current -> .1.
    try:
        oldest = _backup_path(log_path, keep)
        if oldest.exists():
            oldest.unlink()
    except OSError:
        pass

    for i in range(keep - 1, 0, -1):
        src = _backup_path(log_path, i)
        dst = _backup_path(log_path, i + 1)
        try:
            if src.exists():
                src.rename(dst)
        except OSError:
            pass

    try:
        log_path.rename(_backup_path(log_path, 1))
        return True
    except OSError:
        return False
