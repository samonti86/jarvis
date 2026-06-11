"""Conversation recall — full-text search over past conversation transcripts (C / M78).

The episodic-memory analog of src/knowledge.py. Where knowledge_search reaches
the user's CURATED facts, recall_conversation reaches the VERBATIM record of
what was actually said in earlier sessions — the detail the lossy session
summaries (src/memory.py) deliberately strip.

Why this exists (the bug that motivated it): the episodic recall path injects
only the last N *summaries* into the system prompt, and the summarizer omits
time-sensitive values to avoid stale-memory pollution. That combination meant
Jarvis could not recall, e.g., a pre-Finals "Spurs in six" prediction he had
actually made — the raw exchange was on disk the whole time, but nothing ever
read it back. This tool reads it back: Claude can search the full transcript
when the user asks "what did you predict?" / "what did we decide about X?".

DESIGN — direct scan, NOT an index (the load-bearing decision):

  Unlike the knowledge corpus (hand-authored, changes rarely → a DROP+rebuild
  FTS5 index is cheap and bug-free), the transcript corpus is APPEND-ONLY and
  grows every single turn. An index over it would force either a rebuild on
  every startup (cost grows with history; today's turns aren't searchable until
  a restart) or an incremental-sync pipeline (exactly the bug-class knowledge.py
  avoids). A direct scan of the per-day JSONL files sidesteps both: it reads the
  files that exist, so it is ALWAYS current — "what did we just discuss an hour
  ago" works, which a stale index would miss. At personal scale (a year of text
  is a few MB) a linear scan is microseconds.

  Keyword (token) matching is v1. Recall queries are keyword-friendly: the user
  asks about a concrete noun (a team, film, person, decision) that appears
  verbatim in the transcript, and crude prefix-stemming covers
  predict/predicted/prediction. The semantic (embedding) half is a deliberate,
  MEASURED follow-on — the M45→M46 progression repeated: ship the simple correct
  thing, add vectors only once keyword recall is the measured limiter. The
  ranking is structured around an RRF fusion seam (_rank_exchanges) so that
  later addition is a clean drop-in over the same candidate set, no rearchitecting.

Defensive contract, identical to src/knowledge.py / src/tmdb.py: every public
entry point never raises and always returns a readable string Claude can
paraphrase. Missing sessions dir, unreadable file, malformed JSONL line, no
matches — all degrade to a voice-friendly message, never an exception into the
listen loop.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.memory import default_base_dir, format_relative_time


# --- Anthropic tool definition ---------------------------------------------
RECALL_CONVERSATION_TOOL = {
    "name": "recall_conversation",
    "description": (
        "Search the FULL TEXT of your past conversations with the user and "
        "read back what was actually said. This reaches the VERBATIM record — "
        "use it when the user asks about a specific detail from an earlier "
        "chat that your short 'Recent conversations' summaries don't contain: "
        "'what did you predict for the Finals?', 'what did we decide about the "
        "server?', 'what did I say about X last week?', 'what was that movie "
        "you recommended?'. Those summaries are lossy (they omit specific "
        "values and details on purpose); this tool searches the actual "
        "transcript. It is your EPISODIC memory of past chats — distinct from "
        "knowledge_search (the user's curated facts about their setup) and from "
        "web_search (public facts). Staleness rule still applies: a recalled "
        "time-sensitive VALUE (an old score, price, weather) may be out of date "
        "— but a recalled STANCE, prediction, decision, or recommendation is "
        "exactly what this tool is for. If it finds nothing, say so plainly — "
        "don't invent a past conversation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to look for, in natural words or key terms (e.g. "
                    "'NBA finals prediction', 'pet names', 'plex server "
                    "decision'). Matched against the words in past exchanges, "
                    "so include the concrete nouns the conversation would have "
                    "used."
                ),
            },
            "days_back": {
                "type": "integer",
                "description": (
                    "Optional. Only search conversations from the last this-"
                    "many days (e.g. 7 for 'last week', 1 for 'yesterday'). "
                    "Omit to search the whole retained history."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Optional. Max past exchanges to return (default 4, max "
                    "10). Voice replies want few; raise only if the first "
                    "search was too narrow."
                ),
            },
        },
        "required": ["query"],
    },
}


# Voice cap. Each returned exchange carries both sides, so the default is
# leaner than knowledge_search's passages. Claude paraphrases anyway.
_DEFAULT_LIMIT = 4
_MAX_LIMIT = 10

# Per-side excerpt caps in the result payload (voice-lean). The user's question
# is short; Jarvis's reply can ramble, so it gets more room but still bounded.
_USER_EXCERPT_CHARS = 200
_ASSISTANT_EXCERPT_CHARS = 500

# Tokenization mirrors src/knowledge.py: alnum tokens, stopwords stripped so
# they don't dominate the match. Kept deliberately consistent with the
# knowledge tool so the two memory surfaces behave the same to the user.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset(
    "the a an of to in on at is are was were be been being and or not for "
    "with my your you i it this that what whats how do does me we our can "
    "could would should will did say said tell told about".split()
)
# Below this length a token must match EXACTLY (prefix-stemming short tokens
# produces noise — 'in' would prefix-match 'india').
_MIN_PREFIX_LEN = 4


def _sessions_dir() -> Path:
    """The raw per-day transcript folder, co-located with the other Jarvis
    runtime state under %LOCALAPPDATA%\\Jarvis (the same path src/memory.py
    writes to via MemoryStore)."""
    return default_base_dir() / "sessions"


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _query_terms(raw: str) -> list[str]:
    """Reduce a natural query to the content tokens we'll match on. Stopwords
    dropped; falls back to the bare tokens if stripping nuked everything (a
    very short query). Capped so a rambling spoken query stays sane."""
    tokens = _tokenize(raw)
    terms = [t for t in tokens if t not in _STOP and len(t) > 1]
    if not terms:
        terms = [t for t in tokens if len(t) > 1]
    return terms[:12]


def _term_matches(term: str, tokens: set[str]) -> bool:
    """A query term matches an exchange if it appears, or prefix-stems to, any
    token in it. Prefix-stemming (only for tokens >= _MIN_PREFIX_LEN) is the
    crude-but-effective stand-in for a real stemmer: 'predict' matches
    'predicted'/'prediction', 'final' matches 'finals'. The embedding layer is
    the upgrade if this ever proves too blunt."""
    if term in tokens:
        return True
    if len(term) < _MIN_PREFIX_LEN:
        return False
    for tok in tokens:
        if len(tok) >= _MIN_PREFIX_LEN and (
            tok.startswith(term) or term.startswith(tok)
        ):
            return True
    return False


def _iter_session_files(days_back: int | None):
    """Yield (file_date, path) for date-named session files within the window,
    newest file first. Non-date-named files are skipped (same tolerance as
    MemoryStore.prune)."""
    sessions = _sessions_dir()
    if not sessions.is_dir():
        return
    cutoff: datetime | None = None
    if days_back is not None and days_back > 0:
        cutoff = datetime.now() - timedelta(days=days_back)
    dated: list[tuple[datetime, Path]] = []
    for path in sessions.glob("*.jsonl"):
        try:
            file_date = datetime.strptime(path.stem, "%Y-%m-%d")
        except ValueError:
            continue
        if cutoff is not None and file_date.date() < cutoff.date():
            continue
        dated.append((file_date, path))
    dated.sort(key=lambda t: t[0], reverse=True)
    yield from dated


def _collect_exchanges(days_back: int | None) -> list[dict]:
    """Read the in-window session files and reconstruct (ts, user, assistant)
    exchanges. record_turn writes a user line immediately followed by an
    assistant line (same ts), so we pair a buffered user turn with the next
    assistant turn. Malformed lines are skipped, never fatal."""
    exchanges: list[dict] = []
    cutoff: datetime | None = None
    if days_back is not None and days_back > 0:
        cutoff = datetime.now() - timedelta(days=days_back)

    for _file_date, path in _iter_session_files(days_back):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"[recall] skip unreadable {path}: {exc}", file=sys.stderr)
            continue
        pending_user: dict | None = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue  # malformed JSONL line — skip silently
            role = rec.get("role")
            content = rec.get("content")
            if not isinstance(content, str):
                content = _flatten_content(content)
            if role == "user":
                # Two users in a row (no assistant): keep the latest.
                pending_user = rec
            elif role == "assistant" and pending_user is not None:
                ts = pending_user.get("ts") or rec.get("ts") or ""
                # Per-exchange window filter (finer than the per-file one).
                if cutoff is not None:
                    try:
                        if datetime.fromisoformat(ts) < cutoff:
                            pending_user = None
                            continue
                    except (ValueError, TypeError):
                        pass
                exchanges.append({
                    "ts": ts,
                    "user": pending_user.get("content") if isinstance(
                        pending_user.get("content"), str
                    ) else _flatten_content(pending_user.get("content")),
                    "assistant": content,
                })
                pending_user = None
    return exchanges


def _flatten_content(content) -> str:
    """An attachment turn (M31) stores content as a list of blocks. Flatten to
    text for matching — image blocks become a terse marker (we never match on
    base64). Mirrors memory._content_to_text's intent."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "image":
                    parts.append("[image]")
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _rank_exchanges(exchanges: list[dict], terms: list[str]) -> list[dict]:
    """Score each exchange by how many distinct query terms it matches (in
    either side), drop zero-score, rank by score then recency.

    This is the FUSION SEAM: today it returns a single keyword-ranked list. A
    future semantic layer would compute a second (embedding-ranked) list over
    the SAME `exchanges` and merge the two by Reciprocal Rank Fusion here —
    exactly how src/knowledge.py._rrf_fuse combines its keyword and vector
    lists. The callers below depend only on the ranked output, so dropping that
    in changes nothing downstream."""
    scored: list[tuple[int, str, dict]] = []
    for ex in exchanges:
        tokens = set(_tokenize(ex["user"])) | set(_tokenize(ex["assistant"]))
        score = sum(1 for term in terms if _term_matches(term, tokens))
        if score > 0:
            scored.append((score, ex["ts"], ex))
    # Higher score first; within a score, newer (larger ISO ts) first.
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [ex for _score, _ts, ex in scored]


def _truncate(text: str, limit: int) -> str:
    """Collapse whitespace for voice and cap length at a word boundary."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > limit // 2 else cut) + " …"


def _search(query: str, days_back: int | None, limit: int) -> str:
    """Direct-scan keyword recall. Never raises."""
    sessions = _sessions_dir()
    if not sessions.is_dir():
        return (
            "We don't have any recorded conversation history yet, sir."
        )

    terms = _query_terms(query)
    if not terms:
        return (
            f"I couldn't make a searchable query out of '{query}', sir — "
            f"try naming what it was about."
        )

    try:
        exchanges = _collect_exchanges(days_back)
    except Exception as exc:  # noqa: BLE001 — recall must never break a turn
        print(f"[recall] collect failed: {exc}", file=sys.stderr)
        return "I couldn't read our conversation history just now, sir."

    if not exchanges:
        window = f" in the last {days_back} days" if days_back else ""
        return (
            f"I don't have any recorded conversations{window} to search, sir."
        )

    ranked = _rank_exchanges(exchanges, terms)
    if not ranked:
        window = f" from the last {days_back} days" if days_back else ""
        return (
            f"I couldn't find anything in our past conversations{window} "
            f"about '{query}', sir."
        )

    hits = ranked[:limit]
    now = datetime.now()
    out = [
        f"Found {len(hits)} relevant exchange"
        f"{'s' if len(hits) != 1 else ''} from our past conversations:"
    ]
    for ex in hits:
        when = format_relative_time(ex["ts"], now) if ex["ts"] else "earlier"
        user = _truncate(ex["user"], _USER_EXCERPT_CHARS)
        asst = _truncate(ex["assistant"], _ASSISTANT_EXCERPT_CHARS)
        out.append(f'- [{when}] You asked: "{user}" — I replied: "{asst}"')
    return "\n".join(out)


def execute_recall_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises. Same
    shape as execute_knowledge_tool: validate args, dispatch, every failure
    path is a readable string Claude can paraphrase for voice."""
    query = (params.get("query") or "").strip()
    if not query:
        return "A search query is required to recall a past conversation."

    days_back: int | None
    raw_days = params.get("days_back")
    if raw_days in (None, ""):
        days_back = None
    else:
        try:
            days_back = int(raw_days)
            if days_back <= 0:
                days_back = None
        except (TypeError, ValueError):
            days_back = None

    try:
        limit = int(params.get("limit") or _DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    limit = max(1, min(limit, _MAX_LIMIT))

    return _search(query, days_back, limit)
