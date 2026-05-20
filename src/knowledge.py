"""Personal knowledge base — curated/private facts Claude can search (M45).

This is the gap web_search structurally CANNOT fill: private knowledge no
public tool can reach — homelab topology, the Plex/MEDIA-HOST runbook,
3D-printer profiles, work deployment notes, anything the user
explicitly teaches Jarvis. Textbook RAG, slotted in as one more M13-pattern
client-side tool (same shape as src/tmdb.py / src/games.py).

Two stores, deliberately separated (the load-bearing design decision):

  Corpus  — the SOURCE OF TRUTH. Plain *.md / *.txt under a git-versioned
            folder (`~/repos/jarvis-knowledge/` by default — the user's repo
            convention, so it's offsite-replicated like every other repo).
            Hand-authored; survives a disk death via `git clone`.
  Index   — a DERIVED cache. A SQLite FTS5 database at
            %LOCALAPPDATA%\\Jarvis\\knowledge.db. Rebuildable from the corpus
            at any time, so it is intentionally NOT versioned (gitignored).

This is the same relationship a search engine has with its documents vs. its
inverted index, or that a package manager has with a lockfile vs. node_modules:
one is authored and backed up, the other is regenerated on demand. `reindex()`
is the regenerate step.

Retrieval is HYBRID (M46): FTS5 keyword search + local-embedding vector
search, merged by Reciprocal Rank Fusion. M45 shipped keyword-only — `sqlite3`
with FTS5 is stdlib, zero heavy deps — and proved the whole ingest -> tool ->
"update" UX loop. M46 added the semantic half once a measured probe showed
keyword recall was the real limiter: paraphrased queries ('microphone' for
'mic', 'home theater' for 'Plex') that an inverted index structurally cannot
match. The embedding model runs LOCALLY (sentence-transformers,
all-MiniLM-L6-v2) — the private corpus never leaves the machine; API
embeddings would violate privacy-by-design. If sentence-transformers is
absent the vector half is simply skipped and search degrades cleanly to the
M45 keyword-only behaviour — M46 never breaks M45.

This is SEMANTIC/DECLARATIVE memory ("curated facts/docs"), kept strictly
distinct from the M6/M10 EPISODIC conversational memory ("what we discussed").
Different stores, different lifecycles — do not merge them.

Same defensive contract as src/tmdb.py / src/games.py: every public entry
point never raises and always returns a readable string Claude can paraphrase
(missing corpus, empty index, FTS5 unavailable, disk error all degrade to a
voice-friendly message). A knowledge-base failure must never break a turn.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.memory import default_base_dir


# --- Anthropic tool definition ---------------------------------------------
KNOWLEDGE_SEARCH_TOOL = {
    "name": "knowledge_search",
    "description": (
        "Search the user's PRIVATE, curated knowledge base — facts and "
        "documents they have personally written down or explicitly taught "
        "you: their homelab/network topology, the Plex and MEDIA-HOST "
        "runbook, 3D-printer profiles, work deployment notes, "
        "personal preferences and setup decisions. ALWAYS prefer this over "
        "web_search or training/memory for any question about THE USER'S OWN "
        "setup, environment, equipment, or anything they've told you to "
        "remember permanently ('how is my homelab wired?', 'what's my "
        "printer's profile for PETG?', 'what did I decide about X?', 'what's "
        "the runbook for Y?'). This is private knowledge no public source "
        "has. It is NOT conversational memory of past chats (that's separate) "
        "and NOT public facts (use web_search for those). If it returns "
        "nothing, say so plainly — don't invent an answer from training."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to look up, in natural words — the user's question "
                    "or its key terms (e.g. 'plex laptop service name', "
                    "'3d printer petg temperature', 'home network vlan "
                    "layout'). Matched by hybrid keyword + semantic search, "
                    "so natural paraphrases work — you need not echo the "
                    "corpus's exact wording."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Optional. Max passages to return (default 5, max 10). "
                    "Raise it only if the first search was too narrow."
                ),
            },
        },
        "required": ["query"],
    },
}


# Default corpus location. Read at call time (NOT cached in the frozen Config
# dataclass) for the same reason src/tmdb.py reads TMDB_API_KEY fresh each
# call: it stays hot-swappable in dev, and the user can `git clone` the
# knowledge repo into place without restarting Jarvis. Override via
# JARVIS_KNOWLEDGE_DIR in .env.
_DEFAULT_CORPUS = Path.home() / "repos" / "jarvis-knowledge"

# Text extensions we ingest. Plain text only by design (v1 scope): zero
# extraction deps, fully local. PDF is a documented, scoped follow-on, not a
# v1 dependency.
_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}

# Chunk sizing. Personal-scale corpus, so this is tuned for retrieval quality,
# not ingest throughput. ~1200 chars ≈ a few paragraphs — large enough to
# carry a self-contained fact, small enough that a hit is a focused passage
# rather than a whole document. We never split mid-paragraph unless a single
# paragraph alone exceeds the cap.
_CHUNK_CHAR_CAP = 1200

# Voice cap on returned passages. Claude summarizes for speech anyway; a lean
# payload also keeps prompt-cache write cost down (same reasoning as
# tmdb._MAX_RESULTS).
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 10

# --- M46 hybrid semantic search --------------------------------------------
# Local embedding model: all-MiniLM-L6-v2 — 384-dim, ~80 MB, CPU-fast on the
# Ryzen. Local by REQUIREMENT, not convenience: an API embedding endpoint
# would ship the private corpus to a third party, violating privacy-by-design.
_EMBED_MODEL = "all-MiniLM-L6-v2"
# Per-retriever candidate depth fed into the rank fusion — well above
# _MAX_LIMIT so fusion has real material to merge.
_CANDIDATES = 25
# Reciprocal Rank Fusion constant — 60 is the canonical default; it damps the
# weight of any single list's top ranks so neither retriever dominates.
_RRF_K = 60
# Per-passage excerpt cap in the result payload (voice-lean; Claude paraphrases).
_EXCERPT_CHARS = 420


# --- Path resolution -------------------------------------------------------

def _corpus_dir() -> Path:
    """The source-of-truth folder. Env override, else the repo-convention
    default. Not required to exist — callers handle absence gracefully."""
    raw = os.getenv("JARVIS_KNOWLEDGE_DIR", "").strip()
    return Path(raw).expanduser() if raw else _DEFAULT_CORPUS


def _db_path() -> Path:
    """The derived index. Lives with the other Jarvis runtime state
    (sessions, summaries) under %LOCALAPPDATA%\\Jarvis — co-located because
    it's a rebuildable cache, not authored data. default_base_dir() is the
    same helper src/memory.py uses, so the whole runtime state tree stays in
    one place."""
    return default_base_dir() / "knowledge.db"


# --- Ingestion -------------------------------------------------------------

def _iter_corpus_files(corpus: Path):
    """Yield ingestible text files under the corpus, skipping VCS/hidden
    directories (.git, .obsidian, etc.). rglob keeps subfolders working so
    the user can organize the corpus however they like."""
    for path in sorted(corpus.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(corpus).parts):
            continue
        # Skip README.md — it is meta-documentation ABOUT the knowledge base,
        # not knowledge. Indexing it lets its generic prose ("keep facts
        # atomic", "make it survive a disk death") win on common query words
        # and bury real facts — measured directly in the M46 retrieval probe.
        # The corpus is for knowledge; the README explains the corpus.
        if path.name.lower() == "readme.md":
            continue
        if path.suffix.lower() in _TEXT_SUFFIXES:
            yield path


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")


def _chunk_text(text: str) -> list[str]:
    """Split a document into retrieval chunks on blank-line (paragraph)
    boundaries, accumulating up to _CHUNK_CHAR_CAP.

    We also carry the nearest preceding Markdown heading into each chunk as a
    one-line context prefix. This is the cheapest possible retrieval-quality
    win: a chunk that reads 'GPU passthrough' under a buried '## Proxmox VM
    config' heading becomes findable by a 'proxmox gpu' query even though the
    word 'proxmox' isn't in the paragraph itself. ~15 lines for a real recall
    improvement; the embedding layer (M46) is what we'd reach for if this
    heuristic ever stops being enough.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    heading = ""

    def flush() -> None:
        nonlocal current, size
        if current:
            body = "\n\n".join(current).strip()
            if body:
                chunks.append(f"[{heading}]\n{body}" if heading else body)
            current = []
            size = 0

    # Split into paragraph blocks on one-or-more blank lines.
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue

        # A pure heading line updates context and starts a fresh chunk so the
        # heading binds to what follows it, not what preceded it.
        m = _HEADING_RE.match(block)
        if m and "\n" not in block:
            flush()
            heading = m.group(1)
            continue

        # An over-long single paragraph (e.g. a giant pasted log line) gets
        # hard-split so one pathological block can't blow the cap.
        if len(block) > _CHUNK_CHAR_CAP:
            flush()
            for i in range(0, len(block), _CHUNK_CHAR_CAP):
                piece = block[i : i + _CHUNK_CHAR_CAP]
                chunks.append(f"[{heading}]\n{piece}" if heading else piece)
            continue

        if size + len(block) > _CHUNK_CHAR_CAP:
            flush()
        current.append(block)
        size += len(block) + 2

    flush()
    return chunks


def _fts5_available(conn: sqlite3.Connection) -> bool:
    """CPython's official Windows builds ship sqlite3 with FTS5 compiled in,
    but a custom Python might not. Probe once and degrade gracefully rather
    than crashing the tool — same fail-soft posture as the rest of Jarvis."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp._fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


# --- M46 embedding layer ---------------------------------------------------
# Lazy: sentence-transformers pulls in torch, far too heavy to import at
# module load (knowledge.py is imported by llm.py on every startup). Loaded
# on first use and cached. Sentinel: None = not yet tried, False = load
# failed (→ permanent keyword-only fallback for the process), else the model.
_embedder: object | None = None


def _get_embedder():
    """The embedding model, lazily loaded and cached. Returns None if
    sentence-transformers isn't installed or the model can't load — callers
    then fall back to keyword-only search. The optional-component contract:
    M46 must never break M45."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            print(
                f"[knowledge] loading embedding model ({_EMBED_MODEL})…",
                file=sys.stderr,
            )
            _embedder = SentenceTransformer(_EMBED_MODEL)
            print("[knowledge] embedding model ready", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            print(
                f"[knowledge] embeddings unavailable "
                f"({type(exc).__name__}: {exc}) — keyword-only search",
                file=sys.stderr,
            )
            _embedder = False
    return _embedder or None


def _embed(texts: list[str]):
    """Encode texts → an (N, 384) float32 array, L2-normalized so cosine
    similarity reduces to a plain dot product. Returns None on any failure
    (caller falls back to keyword-only)."""
    model = _get_embedder()
    if model is None or not texts:
        return None
    try:
        import numpy as np  # noqa: PLC0415
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[knowledge] encode failed ({exc})", file=sys.stderr)
        return None


@dataclass
class ReindexResult:
    """Outcome of a reindex, shaped for both a log line and a spoken summary."""
    ok: bool
    files: int
    chunks: int
    message: str  # voice-ready, e.g. "Indexed 42 passages from 7 documents."


def reindex() -> ReindexResult:
    """Rebuild the FTS5 index from scratch off the current corpus.

    Full DROP + rebuild rather than incremental sync. For a personal-scale
    corpus this is sub-second, and it sidesteps an entire class of bugs:
    stale rows for deleted/renamed files, partial-update inconsistency,
    change detection. The corpus is the source of truth; the index is
    disposable and cheap to regenerate — so we regenerate it whole. (If the
    corpus ever grows past 'personal-scale', incremental sync is the obvious
    optimization — but that's a real-usage trigger, not a v1 guess.)

    Never raises. Every failure path returns a readable ReindexResult.
    """
    corpus = _corpus_dir()
    if not corpus.exists():
        return ReindexResult(
            False, 0, 0,
            f"No knowledge folder yet at {corpus}. Create it and add some "
            f"notes, sir, then ask me to update my knowledge.",
        )

    db = _db_path()
    try:
        db.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[knowledge] cannot create db dir: {exc}", file=sys.stderr)
        return ReindexResult(False, 0, 0, "I couldn't write the knowledge index, sir.")

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db))
        if not _fts5_available(conn):
            print("[knowledge] sqlite3 built without FTS5", file=sys.stderr)
            return ReindexResult(
                False, 0, 0,
                "This Python's database engine lacks full-text search, so I "
                "can't index knowledge, sir.",
            )

        # Rebuild atomically: build into the live table inside one
        # transaction so a crash mid-reindex can't leave a half-empty index
        # that silently returns nothing.
        conn.execute("DROP TABLE IF EXISTS chunks")
        # path UNINDEXED: stored so we can name the source file in results,
        # but not tokenized (we never search on the path). porter stemming so
        # 'restarting' matches a query for 'restart' — high-value on a small
        # corpus where exact-form misses hurt recall.
        conn.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "path UNINDEXED, content, tokenize='porter unicode61')"
        )
        # M46 — companion table holding one embedding per chunk, keyed by the
        # same rowid we assign explicitly into `chunks` below. A plain table,
        # not a vector DB: at personal scale (tens to low-thousands of chunks)
        # a brute-force cosine over a numpy array is microseconds — same
        # "stdlib/minimal when it's sufficient" judgement as M45's sqlite3.
        conn.execute("DROP TABLE IF EXISTS vectors")
        conn.execute(
            "CREATE TABLE vectors (rowid INTEGER PRIMARY KEY, embedding BLOB)"
        )

        n_files = n_chunks = 0
        chunk_rows: list[tuple[int, str]] = []  # (rowid, text) for embedding
        for path in _iter_corpus_files(corpus):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"[knowledge] skip unreadable {path}: {exc}", file=sys.stderr)
                continue
            rel = str(path.relative_to(corpus))
            file_had_chunk = False
            for chunk in _chunk_text(text):
                n_chunks += 1  # explicit, deterministic rowid
                conn.execute(
                    "INSERT INTO chunks(rowid, path, content) VALUES (?, ?, ?)",
                    (n_chunks, rel, chunk),
                )
                chunk_rows.append((n_chunks, chunk))
                file_had_chunk = True
            if file_had_chunk:
                n_files += 1

        # M46 — embed every chunk in one batch and store the vectors. If
        # sentence-transformers is unavailable, _embed returns None, the
        # vectors table simply stays empty, and search runs keyword-only.
        embedded = False
        if chunk_rows:
            vecs = _embed([c for _, c in chunk_rows])
            if vecs is not None and len(vecs) == len(chunk_rows):
                conn.executemany(
                    "INSERT INTO vectors(rowid, embedding) VALUES (?, ?)",
                    [
                        (rid, vecs[i].tobytes())
                        for i, (rid, _) in enumerate(chunk_rows)
                    ],
                )
                embedded = True
        conn.commit()
    except sqlite3.Error as exc:
        print(f"[knowledge] reindex sqlite error: {exc}", file=sys.stderr)
        return ReindexResult(False, 0, 0, "The knowledge index hit a database error, sir.")
    finally:
        if conn is not None:
            conn.close()

    if n_chunks == 0:
        return ReindexResult(
            True, 0, 0,
            "Your knowledge folder is there but empty, sir — nothing to index yet.",
        )
    msg = (
        f"Knowledge updated, sir — indexed {n_chunks} "
        f"passage{'s' if n_chunks != 1 else ''} from {n_files} "
        f"document{'s' if n_files != 1 else ''}"
        + (", with semantic search."
           if embedded else
           " (keyword-only — semantic search is unavailable).")
    )
    print(
        f"[knowledge] reindex ok: {n_files} files, {n_chunks} chunks, "
        f"semantic={'on' if embedded else 'off'}",
        file=sys.stderr,
    )
    return ReindexResult(True, n_files, n_chunks, msg)


# --- Search ----------------------------------------------------------------

# FTS5's MATCH grammar treats " * ( ) : - ^ AND OR NOT NEAR as operators.
# Voice/transcribed queries contain those freely ("what's my printer's
# temp?"), and an unescaped one raises `sqlite3.OperationalError: fts5:
# syntax error`. We defuse this by reducing the query to bare alnum tokens
# and OR-joining them, each double-quoted so FTS5 treats it as a literal
# term. OR (not AND) because on a small personal corpus recall matters more
# than precision — better to surface a loosely-related passage and let Claude
# judge it than to return nothing on a slightly-off phrasing.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
# Stopwords would otherwise dominate the OR and drown the real terms.
_STOP = frozenset(
    "the a an of to in on at is are was were be been being and or not for "
    "with my your you i it this that what whats how do does my me we our "
    "can could would should will".split()
)


def _build_match_query(raw: str) -> str | None:
    """Sanitize arbitrary voice text into a safe FTS5 MATCH expression.
    None if nothing usable survives (caller treats that as 'no match')."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(raw or "")]
    terms = [t for t in tokens if t not in _STOP and len(t) > 1]
    # If stopword-stripping nuked everything (e.g. a very short query), fall
    # back to the raw tokens so we still attempt a search.
    if not terms:
        terms = [t for t in tokens if len(t) > 1]
    if not terms:
        return None
    # Cap term count so a rambling spoken query stays a sane FTS5 expression.
    return " OR ".join(f'"{t}"' for t in terms[:12])


def _vector_candidates(conn: sqlite3.Connection, query: str, n: int) -> list[int]:
    """M46 — top-n chunk rowids by cosine similarity to the query. Returns []
    (→ the caller runs keyword-only) when embeddings are unavailable: a
    pre-M46 index with no `vectors` table, an empty table, or
    sentence-transformers not installed. Never raises."""
    try:
        rows = conn.execute("SELECT rowid, embedding FROM vectors").fetchall()
    except sqlite3.OperationalError:
        return []  # no `vectors` table — index predates M46 / not reindexed
    if not rows:
        return []
    qv = _embed([query])
    if qv is None:
        return []
    try:
        import numpy as np  # noqa: PLC0415
        mat = np.frombuffer(
            b"".join(emb for _, emb in rows), dtype=np.float32
        ).reshape(len(rows), -1)
        sims = mat @ qv[0]                       # both L2-normalized → cosine
        order = np.argsort(-sims)[:n]
        return [int(rows[i][0]) for i in order]
    except Exception as exc:  # noqa: BLE001 — degrade to keyword-only
        print(f"[knowledge] vector search failed ({exc})", file=sys.stderr)
        return []


def _rrf_fuse(keyword_ids: list[int], vector_ids: list[int]) -> list[int]:
    """Reciprocal Rank Fusion of two ranked rowid lists. Each list contributes
    1/(_RRF_K + rank) per item; the summed score ranks the merged set. A chunk
    both retrievers surface rises; a chunk only one finds still gets a fair
    shot. With one list empty this degenerates cleanly to the other's order —
    which is exactly the keyword-only fallback when embeddings are off."""
    scores: dict[int, float] = {}
    for ranked in (keyword_ids, vector_ids):
        for rank, rid in enumerate(ranked):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda r: scores[r], reverse=True)


def _search(query: str, limit: int) -> str:
    """M46 hybrid retrieval: FTS5 keyword search + local-embedding vector
    search, merged by Reciprocal Rank Fusion. Keyword search still carries
    exact terms (PlexService, ed25519, error strings) where embeddings go
    fuzzy; vector search carries the paraphrases an inverted index
    structurally misses ('microphone' for 'mic', 'home theater' for 'Plex').
    With embeddings unavailable the vector list is empty and this degrades
    exactly to the M45 keyword-only behaviour. Never raises."""
    db = _db_path()
    if not db.exists():
        return (
            "I have no knowledge base indexed yet, sir. Add notes to the "
            "knowledge folder and ask me to update my knowledge."
        )

    match = _build_match_query(query)  # None ⇒ no usable keyword terms

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db))
        # Keyword candidates — bm25-ranked (lower cost = better, so ORDER BY
        # ascending). Empty list if the query reduced to no usable terms.
        keyword_ids: list[int] = []
        if match is not None:
            try:
                keyword_ids = [
                    int(r[0]) for r in conn.execute(
                        "SELECT rowid FROM chunks WHERE chunks MATCH ? "
                        "ORDER BY bm25(chunks) LIMIT ?",
                        (match, _CANDIDATES),
                    ).fetchall()
                ]
            except sqlite3.OperationalError as exc:
                print(f"[knowledge] search op error: {exc}", file=sys.stderr)
                return (
                    "My knowledge index isn't ready, sir — ask me to update "
                    "my knowledge and try again."
                )
        # Vector candidates — [] if embeddings unavailable (keyword-only path).
        vector_ids = _vector_candidates(conn, query, _CANDIDATES)

        if not keyword_ids and not vector_ids:
            return (
                f"Nothing in your knowledge base matches '{query}', sir. "
                f"It may not be something you've taught me yet."
            )

        fused = _rrf_fuse(keyword_ids, vector_ids)[:limit]
        placeholders = ",".join("?" * len(fused))
        passages = {
            int(rid): (path, content)
            for rid, path, content in conn.execute(
                f"SELECT rowid, path, content FROM chunks "
                f"WHERE rowid IN ({placeholders})",
                fused,
            ).fetchall()
        }
    except sqlite3.Error as exc:
        print(f"[knowledge] search sqlite error: {exc}", file=sys.stderr)
        return "The knowledge search hit a database error, sir."
    finally:
        if conn is not None:
            conn.close()

    out = [f"Found {len(fused)} passage(s) in your knowledge base:"]
    for rid in fused:                       # keep fused (relevance) order
        if rid not in passages:
            continue
        path, content = passages[rid]
        excerpt = " ".join((content or "").split())  # collapse for voice
        if len(excerpt) > _EXCERPT_CHARS:
            excerpt = excerpt[:_EXCERPT_CHARS] + " …"
        out.append(f"- [{path}] {excerpt}")
    return "\n".join(out)


def execute_knowledge_tool(params: dict) -> str:
    """Run the tool. Always returns a string for Claude — never raises.
    Same shape as execute_tmdb_tool: validate args, dispatch, every failure
    path is a readable string Claude can paraphrase for voice."""
    query = (params.get("query") or "").strip()
    if not query:
        return "A search query is required for the knowledge base."

    try:
        limit = int(params.get("limit") or _DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    limit = max(1, min(limit, _MAX_LIMIT))

    return _search(query, limit)


# --- "Jarvis, remember this permanently: ..." voice-add path ---------------

def _slugify(text: str, max_len: int = 48) -> str:
    """First few words → a filesystem-safe slug. Mirrors the one-fact-per-
    file discipline of the Claude memory system: each taught fact is its own
    atomic, individually editable/deletable document."""
    words = _TOKEN_RE.findall(text.lower())
    slug = "-".join(words)[:max_len].strip("-")
    return slug or "note"


def remember_fact(fact: str) -> str:
    """Append one taught fact to the corpus as its own dated Markdown file,
    then reindex so it's immediately searchable. Returns a voice-ready
    confirmation. Never raises.

    One file per fact (vs. one growing notes.md): a single now-false fact can
    be deleted cleanly, git diffs stay readable, and reindex stays naturally
    incremental-friendly later — the same reasoning behind the Claude memory
    system's one-file-per-memory layout.
    """
    fact = (fact or "").strip()
    if not fact:
        return "There was nothing to remember, sir."

    corpus = _corpus_dir()
    try:
        corpus.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[knowledge] cannot create corpus dir: {exc}", file=sys.stderr)
        return "I couldn't reach the knowledge folder to save that, sir."

    now = datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    fname = f"{now.strftime('%Y-%m-%d')}-{_slugify(fact)}.md"
    path = corpus / fname
    # Collision guard if two facts share opening words within the same day.
    if path.exists():
        path = corpus / f"{now.strftime('%Y-%m-%d')}-{_slugify(fact)}-{stamp}.md"

    body = (
        f"# {fact.splitlines()[0][:80]}\n\n"
        f"_Taught by voice on {now.strftime('%Y-%m-%d %H:%M')}._\n\n"
        f"{fact}\n"
    )
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        print(f"[knowledge] cannot write fact file: {exc}", file=sys.stderr)
        return "I couldn't save that to the knowledge folder, sir."

    result = reindex()
    if result.ok:
        return "Noted and filed permanently, sir."
    # Saved but index didn't refresh — still durable on disk, say so honestly.
    return (
        "I've saved that to your knowledge folder, sir, though the index "
        "didn't refresh — ask me to update my knowledge."
    )


# --- Voice-intent matchers -------------------------------------------------
# Same pattern as src/face_auth.matches_enroll_intent: matched in main.py's
# listen loop BEFORE the turn reaches Claude, so the LLM doesn't try to
# satisfy a corpus-write intent with the wrong tool.

# "update/reindex/rebuild/refresh your knowledge[ base]". Deliberately
# requires the word 'knowledge' so a generic "update yourself" doesn't fire.
_REINDEX_INTENT_RE = re.compile(
    r"\b(update|reindex|re-?index|rebuild|refresh|sync)\b"
    r"[\w\s]*?\bknowledge(\s+base)?\b",
    re.IGNORECASE,
)

# "remember this permanently: X" / "permanently remember that X" / "remember
# X forever". The permanence adverb is the disambiguator from ordinary
# conversational "remember when we..." (that's episodic memory, a different
# store) and from face_auth's "remember my face". Group 'fact' is the payload.
_REMEMBER_INTENT_RE = re.compile(
    r"\b(?:remember|note|save|store)\b"
    r"(?:\s+(?:this|that|the\s+following))?"
    r"(?:[\s,:-]+)?"
    r"(?:permanently|forever|for\s+good|for\s+later)"
    r"(?:[\s,:.-]+|\s+that\s+|\s+this[:,]?\s+)"
    r"(?P<fact>.+\S)",
    re.IGNORECASE | re.DOTALL,
)
# Also accept the adverb FIRST: "permanently remember (that) X".
_REMEMBER_INTENT_RE_ALT = re.compile(
    r"\b(?:permanently|forever)\s+(?:remember|note|save|store)\b"
    r"(?:\s+(?:that|this))?[\s,:-]+"
    r"(?P<fact>.+\S)",
    re.IGNORECASE | re.DOTALL,
)


def matches_reindex_intent(transcript: str) -> bool:
    """True if the user asked to rebuild the knowledge index."""
    return bool(_REINDEX_INTENT_RE.search(transcript or ""))


def extract_remember_fact(transcript: str) -> str | None:
    """If the transcript is a 'remember this permanently: ...' command,
    return the fact to store; else None.

    v1 limitation (documented): the fact must be in the same utterance. A
    bare 'remember this permanently' with no payload returns None and falls
    through to Claude (which will simply ask what to remember) rather than
    entering a multi-turn capture state — that stateful flow is a clean
    follow-on, not v1 scope.
    """
    t = (transcript or "").strip()
    for rx in (_REMEMBER_INTENT_RE, _REMEMBER_INTENT_RE_ALT):
        m = rx.search(t)
        if m:
            fact = m.group("fact").strip(" \t\r\n.,:;-")
            if len(fact) >= 3:
                return fact
    return None
