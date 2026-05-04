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

from pathlib import Path


DEFAULT_MAX_BYTES = 5_000_000   # 5 MB — months of moderate use
DEFAULT_KEEP = 3                # log + 3 backups = ~20 MB max footprint


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
