"""File → Anthropic content-block conversion for M16 attachments.

Handles three file families that Claude reads natively:
  - PDFs       → {"type": "document", ...} (base64-encoded)
  - Images     → {"type": "image", ...}    (base64-encoded; PNG/JPEG/GIF/WebP)
  - Text files → {"type": "text", ...}     (UTF-8 decoded, with filename header)

Returns (block, error) — exactly one of the two is non-None. Caller displays
the error in the system transcript if non-None.

Size cap is 25 MB to stay safely under Anthropic's 32 MB inline document limit
and avoid blowing up memory during base64 encoding (which roughly inflates 4:3).

This module is deliberately UI-agnostic and synchronous — it returns plain dicts
that drop straight into a Messages API user message's `content` list. No
network, no threading, no Tk.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB

# Mimetypes Anthropic accepts on `image` content blocks.
_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}

# Common text/code file extensions we'll embed inline as a `text` block.
# mimetypes.guess_type() misses some of these (.md, .yml, .log, .toml) so we
# extension-hint as well as MIME-check.
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".env",
    ".log", ".html", ".htm", ".xml", ".css",
    ".js", ".ts", ".tsx", ".jsx",
    ".py", ".rb", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".sh", ".ps1", ".bat", ".sql",
}


def load_attachment(path: str) -> tuple[dict | None, str | None]:
    """Load a file and return (content_block, error). Exactly one is None.

    The returned content_block (if any) is a dict that drops straight into
    the `content` list of a Messages API user message — no further processing
    needed by the caller.
    """
    p = Path(path)
    if not p.is_file():
        return None, f"not a file: {p.name}"

    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        mb = size // (1024 * 1024)
        cap = MAX_FILE_BYTES // (1024 * 1024)
        return None, f"{p.name} is {mb} MB (limit {cap} MB)"

    suffix = p.suffix.lower()
    mime, _ = mimetypes.guess_type(str(p))

    # PDFs — extension OR sniffed MIME, since some downloads come through with
    # no extension at all.
    if suffix == ".pdf" or mime == "application/pdf":
        b64, err = _read_b64(p)
        if err is not None:
            return None, err
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64,
            },
        }, None

    # Images
    if mime in _IMAGE_MIMES:
        b64, err = _read_b64(p)
        if err is not None:
            return None, err
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": b64,
            },
        }, None

    # Text / code files — decode and embed with a filename header so Claude
    # has context about what it's reading.
    if suffix in _TEXT_EXTENSIONS or (mime and mime.startswith("text/")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return None, f"failed to read {p.name}: {exc}"
        return {
            "type": "text",
            "text": f"[Attached file: {p.name}]\n\n{text}",
        }, None

    return None, f"unsupported file type: {p.name}"


def _read_b64(p: Path) -> tuple[str, str | None]:
    """Read bytes and base64-encode. Returns (b64, error)."""
    try:
        return base64.b64encode(p.read_bytes()).decode("ascii"), None
    except OSError as exc:
        return "", f"failed to read {p.name}: {exc}"
