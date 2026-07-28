r"""Cross-session self-review — "how have you been, Jarvis?" (M93).

WHAT M60 ALREADY DOES, AND WHAT IT CANNOT:
`status_report` counts concerning lines SINCE THE CURRENT SESSION MARKER. That
is the right window for "are you healthy right now" and the wrong one for "is
something quietly wrong with you". A fault that appears once per run is
invisible per-run and obvious across fifty.

Measured on the real log before this was written: 31,427 lines, 58 sessions,
73 concerning lines — which collapse to **19 distinct signatures**, one of
which (`[outlook] ical fetch failed: ConnectTimeout`) accounts for 47 of them.
Nobody would find that by reading a session at a time, and nobody was going to
read 31,000 lines. That gap is this module.

WHY SIGNATURES, NOT LINES:
Every log line carries a timestamp, so raw counting reports every line as
unique and the recurring fault hides in the noise. Normalising away the
volatile parts — timestamp, durations, ids, paths, retry suffixes — turns 73
lines into 19 things that are actually wrong, ranked.

RECURRENCE IS MEASURED IN SESSIONS, NOT OCCURRENCES:
"47 times" could be one bad afternoon. "In 12 of the last 58 runs" is a
standing defect. The second number is the one that tells you whether to care,
so both are reported and the ranking uses sessions first.

READ-ONLY, DELIBERATELY:
This reads logs and reports. It does not change code, open pull requests, or
restart anything. An assistant that diagnoses itself is useful; one that edits
itself unattended is the thing this project has consistently refused to build
([[feedback-jarvis-least-privilege]]). The fix stays a human decision.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

# A line worth looking at. Deliberately broader than a strict ERROR level —
# this codebase logs failures as prose ("could not read", "fetch err").
_CONCERNING = re.compile(
    r"error|failed|exception|traceback|could not|unable to|refused|timed out",
    re.IGNORECASE,
)

# Lines that match the above but are NOT faults: the word appears in ordinary
# reporting. Without this the report is dominated by its own vocabulary.
_BENIGN = re.compile(
    r"errors?=0|0 errors|no errors|error-free|"
    r"ignoring JARVIS_|--- Jarvis started",
    re.IGNORECASE,
)

_SESSION_MARKER = "--- Jarvis started"
_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)[^\]]*\]\s*")

# A crash leaves a traceback; a clean stop does not. Used to tell "ended" from
# "died", which is the difference between a restart and an incident.
_CRASH_HINT = re.compile(r"traceback \(most recent call last\)", re.IGNORECASE)

# Traceback SCAFFOLDING. The first run of this against the real log ranked
# "Traceback (most recent call last):" and "The above exception was the direct
# cause..." as the top two issues — which is noise, not insight: they are the
# frame around an error, never the error. These lines are skipped, and the
# traceback is attributed to its terminating exception line instead (below).
_TB_NOISE = re.compile(
    r"^\s*(?:file \"|\^{2,}|~+\^|during handling of the above|"
    r"the above exception was the direct cause|traceback \(most recent call last\))",
    re.IGNORECASE,
)

# The line that actually names the failure: `ValueError: ...`,
# `websockets.exceptions.InvalidMessage: ...`, `PortAudioError: ...`.
_EXC_LINE = re.compile(r"^\s*(?:[A-Za-z_][\w.]*\.)?([A-Z]\w*(?:Error|Exception|Warning))\b")

# The session banner is written directly by setup_logging, so unlike every
# other line it has NO "[timestamp]" prefix — it carries its own ISO stamp
# instead. Parsing that is what makes the day-window honest; without it a
# marker inherits whatever window the previous line happened to be in.
_SESSION_STAMP = re.compile(
    r"--- Jarvis started (\d{4}-\d\d-\d\d)[T ](\d\d:\d\d:\d\d)"
)


def _log_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Jarvis"


def log_files() -> list[Path]:
    """Current log first, then rotations (jarvis.log.1, .2, ...) oldest last.

    Rotations matter: at 5 MB a busy fortnight can roll, and the trend this
    module exists to find is exactly what would be cut in half by a rotation.
    """
    base = _log_dir() / "jarvis.log"
    files = [base] if base.exists() else []
    for i in range(1, 6):
        p = _log_dir() / f"jarvis.log.{i}"
        if p.exists():
            files.append(p)
    return files


def signature(line: str) -> str:
    """Collapse one log line to what is actually wrong with it.

    Strips the parts that differ between two occurrences of the SAME fault:
    timestamp, durations, numeric ids, absolute paths, and the retry suffix
    this codebase appends to transient failures.
    """
    s = _TS.sub("", line)
    s = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ms|s)\b", "<dur>", s)
    s = re.sub(r"\b0x[0-9a-fA-F]+\b", "<addr>", s)
    s = re.sub(r"[A-Za-z]:\\[^\s]+", "<path>", s)
    s = re.sub(r"/[\w./-]{8,}", "<path>", s)
    s = re.sub(r"\b\d{3,}\b", "<n>", s)
    s = re.sub(r"\s+[—-]\s+retrying.*$", "", s)
    return s.strip()[:140]


def _line_date(line: str) -> datetime | None:
    m = _TS.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def scan(days: int = 7) -> dict:
    """Walk the logs and return a structured health picture.

    Never raises: an unreadable or absent log yields an empty-but-valid result,
    because a self-check that crashes is worse than one that says "no data".
    """
    cutoff = datetime.now() - timedelta(days=max(days, 1))
    sessions = 0
    crashes = 0
    total_lines = 0
    concerning = 0
    # signature -> [count, first_seen, last_seen, {session indices}]
    sigs: dict[str, list] = {}
    current_session = 0
    session_had_crash = False
    in_window = True  # lines before the first timestamp are assumed in-window

    for path in log_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"[selfreview] cannot read {path.name}: {exc}", file=sys.stderr)
            continue

        for line in text.splitlines():
            total_lines += 1
            stamp = _line_date(line)
            if stamp is not None:
                in_window = stamp >= cutoff

            if _SESSION_MARKER in line:
                if session_had_crash:
                    crashes += 1
                session_had_crash = False
                # The banner carries its own ISO stamp; trust that over the
                # inherited window, or a rotated log's ancient sessions get
                # counted as recent.
                m = _SESSION_STAMP.search(line)
                if m:
                    try:
                        in_window = datetime.strptime(
                            f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S"
                        ) >= cutoff
                    except ValueError:
                        pass
                if in_window:
                    sessions += 1
                    current_session += 1
                continue

            if not in_window:
                continue
            if _CRASH_HINT.search(line):
                session_had_crash = True
            # Skip the frame around an exception — it is not the exception.
            if _TB_NOISE.match(line):
                continue
            if not _CONCERNING.search(line) or _BENIGN.search(line):
                continue

            concerning += 1
            sig = signature(line)
            entry = sigs.get(sig)
            if entry is None:
                sigs[sig] = [1, stamp, stamp, {current_session}]
            else:
                entry[0] += 1
                if stamp is not None:
                    entry[1] = entry[1] or stamp
                    entry[2] = stamp
                entry[3].add(current_session)

    if session_had_crash:
        crashes += 1

    ranked = sorted(
        (
            {
                "signature": s,
                "count": v[0],
                "sessions": len(v[3]),
                "first": v[1].isoformat(timespec="minutes") if v[1] else None,
                "last": v[2].isoformat(timespec="minutes") if v[2] else None,
            }
            for s, v in sigs.items()
        ),
        # Sessions first: a fault in many runs is a standing defect; the same
        # count inside one run is a bad afternoon.
        key=lambda d: (d["sessions"], d["count"]),
        reverse=True,
    )
    return {
        "days": days,
        "sessions": sessions,
        "crashes": crashes,
        "lines": total_lines,
        "concerning": concerning,
        "signatures": ranked,
    }


def format_review(data: dict, top: int = 4) -> str:
    """Render for speech: a verdict first, then only what earns a mention."""
    sessions = data["sessions"]
    if not sessions and not data["concerning"]:
        return "I've no log activity to review, sir."

    sigs = data["signatures"]
    days = data["days"]

    if not sigs:
        return (f"Nothing of concern in the last {days} days, sir — "
                f"{sessions} sessions, no errors logged.")

    # Spoken aloud, so singulars have to be real singulars: "I've run 1 times
    # and logged 1 concerning lines across 1 distinct issues" is the kind of
    # sentence that makes an assistant sound like a database.
    def plural(n: int, one: str, many: str) -> str:
        return f"{n} {one}" if n == 1 else f"{n} {many}"

    runs = "once" if sessions == 1 else f"{sessions} times"
    head = (f"In the last {days} days I've run {runs} "
            f"and logged {plural(data['concerning'], 'concerning line', 'concerning lines')} "
            f"across {plural(len(sigs), 'distinct issue', 'distinct issues')}.")
    if data["crashes"]:
        head += f" {data['crashes']} of those runs ended in an unhandled exception."

    lines = [head]
    for item in sigs[:top]:
        # This is spoken aloud, so "1 times in one session" is not acceptable.
        how_many = "Once" if item["count"] == 1 else f"{item['count']} times"
        where = (f"in {item['sessions']} sessions"
                 if item["sessions"] > 1 else "in one session")
        lines.append(f"{how_many} {where}: {item['signature']}")

    remaining = len(sigs) - min(top, len(sigs))
    if remaining > 0:
        lines.append(f"Plus {remaining} less frequent issues.")
    return "\n".join(lines)


SELF_REVIEW_TOOL = {
    "name": "self_review",
    "description": (
        "Jarvis's own health across RECENT DAYS AND RESTARTS — recurring "
        "faults, how many sessions each affected, and whether any run ended in "
        "a crash. Use for 'how have you been?', 'any problems lately?', 'have "
        "you been having trouble?', 'how are you running?', or when the user "
        "asks what keeps going wrong. "
        "Distinct from status_report, which answers 'are you healthy RIGHT "
        "NOW' for the current session only — use this one for trends and "
        "anything spanning restarts. Read-only: it reports, it does not fix."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "How far back to look. Default 7.",
            },
        },
        "required": [],
    },
}


def execute_self_review(params: dict) -> str:
    """Never raises — a self-check that crashes is worse than one that shrugs."""
    try:
        days = int(params.get("days") or 7)
    except (TypeError, ValueError):
        days = 7
    days = max(1, min(days, 90))
    try:
        return format_review(scan(days))
    except Exception as exc:  # noqa: BLE001
        print(f"[selfreview] scan failed: {exc}", file=sys.stderr)
        return "I couldn't review my logs just now, sir."
