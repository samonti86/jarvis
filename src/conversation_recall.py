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

  Retrieval is HYBRID (M78.1): keyword (token) matching fused with local-
  embedding cosine via Reciprocal Rank Fusion — the same shape as
  src/knowledge.py's M46 layer. Keyword carries the concrete nouns recall
  queries usually contain (a team, film, person, decision) plus crude prefix-
  stemming (predict/predicted/prediction); the vector pass catches paraphrases
  keyword structurally misses. Unlike the knowledge corpus there is NO
  persistent embedding index — the vector pass embeds a BOUNDED candidate set
  (keyword hits ∪ a recent-history window) at query time, so recall stays
  always-current (a new exchange is searchable the moment a query includes it)
  while the embed cost stays bounded. If sentence-transformers is absent the
  vector list is empty and search degrades cleanly to keyword-only. A full-
  corpus embedding cache (to catch paraphrases of OLD, non-keyword, out-of-
  window exchanges) is the measured follow-on if real use shows that gap.

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

from src import embeddings
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
        "web_search (public facts). Each past turn is tagged with WHO said it "
        "(the household member whose voice was recognized), so you can scope a "
        "search to a specific person via the `speaker` argument — use it when "
        "the user asks what a NAMED person said or asked ('what did Alice ask "
        "me to do?', 'what did Alice want?'). Staleness rule still applies: a "
        "recalled time-sensitive VALUE (an old score, price, weather) may be out "
        "of date — but a recalled STANCE, prediction, decision, or "
        "recommendation is exactly what this tool is for. If it finds nothing, "
        "say so plainly — don't invent a past conversation."
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
            "speaker": {
                "type": "string",
                "description": (
                    "Optional. Restrict the search to turns spoken by a "
                    "specific enrolled person, by name (e.g. 'Alice'). Use "
                    "when the user asks what a NAMED person said or asked. Omit "
                    "for the user's own history or when no specific person is "
                    "named — an omitted value searches everyone."
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

# --- M78.1 semantic layer ---------------------------------------------------
# Hybrid retrieval: keyword scoring fused with local-embedding cosine via RRF
# (the same shape as src/knowledge.py's M46 layer). Unlike the knowledge corpus,
# transcripts are append-only and there is NO persistent embedding index — we
# embed a BOUNDED candidate set at query time (keyword hits ∪ a recent-history
# window), which preserves the direct-scan property (always-current — a brand-new
# exchange is embedded the moment a query includes it) AND keeps the per-query
# embed cost bounded. The deliberate v1 limitation: a paraphrase of an OLD
# exchange that shares no keywords and falls outside the recent window won't be
# found semantically — a full-corpus embedding cache is the MEASURED follow-on
# if real use shows that gap (the M45→M46 discipline, repeated). Embeddings
# unavailable ⇒ vector list is empty ⇒ clean keyword-only fallback.
_CANDIDATES = 25            # top-N from each retriever fed into the fusion
_RRF_K = 60                 # Reciprocal Rank Fusion damping (canonical default)
_RECENT_VECTOR_WINDOW = 80  # most-recent exchanges always eligible for vectors
_MAX_EMBED = 150            # hard cap on exchanges embedded per query (latency)
# Cosine floor: only genuinely-similar exchanges enter the vector list. Without
# it, the recent-window candidates would all fuse in at low similarity and
# surface irrelevant recent chatter as "relevant". MiniLM puts related text
# ~0.4-0.7 and unrelated ~0.0-0.2, so 0.30 cleanly separates them. (knowledge.py
# omits this because its small curated corpus wants max recall; recall over a
# large transcript history wants precision instead.)
_VECTOR_SIM_FLOOR = 0.30


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
                    # M82 — who spoke this turn (None for old/typed/remote
                    # records that carry no speaker tag).
                    "speaker": pending_user.get("speaker"),
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


def _rrf_fuse(keyword_ids: list[int], vector_ids: list[int]) -> list[int]:
    """Reciprocal Rank Fusion of two ranked index lists (mirrors
    src/knowledge.py._rrf_fuse). Each list contributes 1/(_RRF_K + rank) per
    item; an item both retrievers surface rises. With one list empty this
    degrades cleanly to the other's order — exactly the keyword-only fallback
    when embeddings are unavailable."""
    scores: dict[int, float] = {}
    for ranked in (keyword_ids, vector_ids):
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def _vector_rank(
    exchanges: list[dict], query: str, keyword_ids: list[int]
) -> list[int]:
    """Top exchange indices by cosine similarity to the query, over a BOUNDED
    candidate set (keyword hits ∪ the most-recent window, capped at _MAX_EMBED)
    and above _VECTOR_SIM_FLOOR. Returns [] when embeddings are unavailable
    (→ the caller runs keyword-only) or nothing clears the floor. Never raises."""
    if not exchanges:
        return []
    # Most-recent-by-ts indices (ISO ts strings sort lexicographically).
    recent = sorted(
        range(len(exchanges)), key=lambda i: exchanges[i]["ts"], reverse=True
    )[:_RECENT_VECTOR_WINDOW]
    # Keyword hits first (keep their order), then the recent window; de-duped,
    # capped. dict.fromkeys preserves first-seen order.
    cand = list(dict.fromkeys(list(keyword_ids) + recent))[:_MAX_EMBED]
    if not cand:
        return []
    texts = [f'{exchanges[i]["user"]} {exchanges[i]["assistant"]}' for i in cand]
    mat = embeddings.embed([query] + texts)  # one batched call; row 0 = query
    if mat is None or len(mat) != len(texts) + 1:
        return []
    try:
        import numpy as np  # noqa: PLC0415
        qv = mat[0]
        sims = mat[1:] @ qv                   # both L2-normalized → cosine
        out: list[int] = []
        for j in np.argsort(-sims):
            if sims[j] < _VECTOR_SIM_FLOOR:
                break                          # sorted desc — nothing better left
            out.append(cand[int(j)])
            if len(out) >= _CANDIDATES:
                break
        return out
    except Exception as exc:  # noqa: BLE001 — degrade to keyword-only
        print(f"[recall] vector rank failed ({exc})", file=sys.stderr)
        return []


def _rank_exchanges(
    exchanges: list[dict], terms: list[str], query: str
) -> list[dict]:
    """Hybrid rank: a keyword pass fused with a semantic (embedding) pass by
    RRF. The keyword pass scores each exchange by how many distinct query terms
    it matches (either side), ranked score-then-recency. The vector pass
    (_vector_rank) ranks by cosine over a bounded candidate set. When the
    embedder is unavailable the vector list is empty and this returns the pure
    keyword ranking — identical to the pre-M78.1 behaviour."""
    # Keyword pass — index-based so it can fuse with the vector pass.
    kw_scored: list[tuple[int, str, int]] = []
    for i, ex in enumerate(exchanges):
        tokens = set(_tokenize(ex["user"])) | set(_tokenize(ex["assistant"]))
        score = sum(1 for term in terms if _term_matches(term, tokens))
        if score > 0:
            kw_scored.append((score, ex["ts"], i))
    # Higher score first; within a score, newer (larger ISO ts) first.
    kw_scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    keyword_ids = [i for _score, _ts, i in kw_scored]

    vector_ids = _vector_rank(exchanges, query, keyword_ids)
    if not vector_ids:
        return [exchanges[i] for i in keyword_ids]  # keyword-only fallback

    fused = _rrf_fuse(keyword_ids, vector_ids)
    return [exchanges[i] for i in fused]


def _norm_speaker(name) -> str:
    """Normalize a speaker name for case-insensitive matching."""
    return str(name or "").strip().lower()


def _speaker_matches(exchange_speaker, wanted: str) -> bool:
    """M82 — does this exchange's tagged speaker match the wanted name?
    Case-insensitive. An exchange with NO speaker tag (None — old/typed/remote
    turns) never matches a specific-speaker filter, so 'what did Alice ask?'
    returns only turns actually tagged Alice, never untagged ones."""
    if not wanted:
        return True  # no filter → everything passes
    es = _norm_speaker(exchange_speaker)
    return bool(es) and es == _norm_speaker(wanted)


def _who_label(speaker) -> str:
    """How to introduce the asker in a result line. The tagged name when known
    (so a recalled Alice turn reads 'Alice asked'), else the neutral 'You'
    (untagged turns are the user's own typed/remote history)."""
    name = str(speaker or "").strip()
    return name if name else "You"


def _truncate(text: str, limit: int) -> str:
    """Collapse whitespace for voice and cap length at a word boundary."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > limit // 2 else cut) + " …"


def _search(query: str, days_back: int | None, limit: int,
            speaker: str = "") -> str:
    """Direct-scan keyword recall. Never raises. `speaker` (M82) optionally
    restricts the search to exchanges spoken by that enrolled person."""
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

    # M82 — speaker filter applied BEFORE ranking, so a per-person query only
    # ranks (and embeds) that person's turns.
    if speaker:
        exchanges = [ex for ex in exchanges
                     if _speaker_matches(ex.get("speaker"), speaker)]
        if not exchanges:
            return (
                f"I don't have any recorded turns from {speaker} to search, "
                f"sir."
            )

    ranked = _rank_exchanges(exchanges, terms, query)
    if not ranked:
        who = f" from {speaker}" if speaker else ""
        window = f" from the last {days_back} days" if days_back else ""
        return (
            f"I couldn't find anything{who} in our past conversations{window} "
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
        who = _who_label(ex.get("speaker"))
        user = _truncate(ex["user"], _USER_EXCERPT_CHARS)
        asst = _truncate(ex["assistant"], _ASSISTANT_EXCERPT_CHARS)
        out.append(f'- [{when}] {who} asked: "{user}" — I replied: "{asst}"')
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

    speaker = (params.get("speaker") or "").strip()

    return _search(query, days_back, limit, speaker=speaker)
