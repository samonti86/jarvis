"""read_local_file — let Jarvis read a text file you point it at.

Use cases:
  - ad-hoc config/log inspection ("check my docker-compose for issues",
    "what's in my .ssh/config", "tail my error log")
  - reading individual files out of a diagnostics bundle after
    run_pc_diagnostics_collector has produced one (M28's other half)

Sibling, deliberately not shared: src/attachments.py's load_attachment turns
a file into a Messages-API *content block* (handles PDF/image base64 too) for
the 📎 upload flow. This module returns *plain text for a tool result* with a
different size budget and a binary-sniff + credential denylist instead of an
extension allowlist. Different jobs — merging them would help neither.

Design / safety posture (the M23 philosophy applied to reads):
  - No confirmation gate. The user named the file; reading is low-impact.
    Confirmation is reserved for destructive / expensive operations
    (kill_process, the diagnostics collector). Friction here would be
    annoying with no security payoff.
  - Read-only by construction. This module never writes, deletes, or
    executes anything.
  - Text-only. A NUL byte in the first chunk → treated as binary, refused.
    Voice/LLM context has no use for raw bytes anyway.
  - Size cap. Truncated to MAX_READ_BYTES of decoded text, with a note.
    We truncate from the FRONT (keep the beginning) — and for the one big
    category (event-log dumps), Get-WinEvent emits newest-first, so the
    front is the recent stuff. Claude can request a more specific file if
    it needs more.
  - Credential-material denylist. Private keys, keystores, and the Windows
    SAM/SECURITY/SYSTEM registry hives are refused by filename pattern,
    regardless of location. This is defense in depth: a hallucinated or
    misguided tool call cannot slurp key material into the conversation
    transcript. It is deliberately NARROW — `.env`, `.ssh/config`, etc.
    are allowed (the user controls this machine, and a `.env` read leaks
    nothing new to Anthropic that the request's own API key doesn't).
  - Every read is logged to stderr (→ jarvis.log) with the resolved path
    and size — an audit trail for "what files has Jarvis looked at".

Defensive contract, same as the rest of the toolkit: never raises, always
returns a readable string.
"""

from __future__ import annotations

import sys
from pathlib import Path


# --- Anthropic tool definition ---------------------------------------------
READ_LOCAL_FILE_TOOL = {
    "name": "read_local_file",
    "description": (
        "Read the contents of a text file on THIS PC and return it. Use when "
        "the user points you at a specific file — a config file, a log, a "
        "Dockerfile, a script, a .ssh/config — or to read an individual file "
        "out of a diagnostics bundle produced by run_pc_diagnostics_collector. "
        "Accepts absolute paths or paths starting with ~ (home). Text files "
        "only; binary files are refused. Large files are truncated. Whatever "
        "you read becomes part of this conversation — don't read files the "
        "user didn't ask about. This tool ONLY reads — it never modifies "
        "anything."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file. Absolute (C:\\Users\\...\\file.txt) or "
                    "~-relative (~/.ssh/config). Forward or back slashes both "
                    "work on Windows."
                ),
            },
        },
        "required": ["path"],
    },
}


# 64 KB of decoded text. ~16k tokens worst case — already plenty for a voice
# context, and the diagnostics-bundle files are designed to be small. The
# only files that routinely exceed this are event-log dumps; front-truncation
# keeps the most-recent entries (Get-WinEvent emits newest-first).
MAX_READ_BYTES = 64 * 1024

# How much of the file to sniff for a NUL byte before deciding "binary".
_BINARY_SNIFF_BYTES = 8192

# Filename patterns that name genuine credential material. Matched case-
# insensitively against the file NAME (not the full path). Narrow on purpose
# — see the module docstring. `.pub` files are public keys → allowed.
_DENIED_EXACT_NAMES = {
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ed25519_sk", "id_ecdsa_sk",
    "sam", "security", "system", "ntuser.dat",  # Windows registry hives
    ".htpasswd",
}
_DENIED_SUFFIXES = {
    ".pem", ".key", ".pfx", ".p12", ".keystore", ".jks", ".kdbx", ".ppk",
    ".asc", ".gpg",
}


def _is_denied(p: Path) -> bool:
    if p.name.lower() in _DENIED_EXACT_NAMES:
        return True
    suffix = p.suffix.lower()
    # `.pub` overrides a denied stem: id_ed25519.pub is a PUBLIC key.
    if suffix == ".pub":
        return False
    return suffix in _DENIED_SUFFIXES


def _looks_binary(raw: bytes) -> bool:
    """A NUL byte in the sniffed prefix is the standard 'this is binary'
    heuristic (git uses it too). Text files essentially never contain \\x00."""
    return b"\x00" in raw[:_BINARY_SNIFF_BYTES]


def execute_read_local_file(params: dict) -> str:
    """Run the tool. Always returns a string — never raises."""
    raw_path = (params.get("path") or "").strip()
    if not raw_path:
        return "read_local_file needs a 'path'."

    try:
        # expanduser handles ~ and ~user; Windows accepts / and \ alike.
        p = Path(raw_path).expanduser()
    except (ValueError, RuntimeError) as exc:
        return f"Bad path '{raw_path}': {exc}"

    # Resolve for display + symlink-following, but don't fail if it doesn't
    # exist yet — let the is_file() check below produce the clearer message.
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p

    if not resolved.exists():
        return f"No such file: {resolved}"
    if resolved.is_dir():
        return f"That's a directory, not a file: {resolved}. Name a file inside it."
    if not resolved.is_file():
        return f"Not a regular file: {resolved}"

    if _is_denied(resolved):
        print(f"[file_reader] DENIED (credential material): {resolved}", file=sys.stderr)
        return (
            f"Refusing to read {resolved.name} — that looks like private key "
            f"or credential material, which I won't pull into the conversation."
        )

    try:
        size = resolved.stat().st_size
    except OSError as exc:
        return f"Could not stat {resolved}: {exc}"

    try:
        with open(resolved, "rb") as fh:
            raw = fh.read(MAX_READ_BYTES + 1)  # +1 so we can tell "exactly at cap" from "over"
    except OSError as exc:
        return f"Could not read {resolved}: {exc}"

    if _looks_binary(raw):
        print(f"[file_reader] refused (binary): {resolved} ({size} bytes)", file=sys.stderr)
        return (
            f"{resolved.name} looks like a binary file ({size} bytes) — "
            f"I can only read text."
        )

    truncated = len(raw) > MAX_READ_BYTES
    if truncated:
        raw = raw[:MAX_READ_BYTES]

    # utf-8-sig strips a leading BOM if present — many Windows tools (incl. the
    # diagnostics collector) save UTF-8 with a BOM, and a stray U+FEFF at the
    # start of the tool result is just noise. errors='replace' keeps us safe on
    # any odd bytes after that.
    text = raw.decode("utf-8-sig", errors="replace")
    line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)

    print(
        f"[file_reader] read {resolved} ({size} bytes, {line_count}+ lines"
        f"{', TRUNCATED' if truncated else ''})",
        file=sys.stderr,
    )

    header = f"[{resolved}] — {size} bytes"
    if truncated:
        header += f" (showing first {MAX_READ_BYTES // 1024} KB; file is larger)"
    body = f"{header}\n\n{text}"
    if truncated:
        body += f"\n\n[...truncated — {size - MAX_READ_BYTES} more bytes not shown]"
    return body
