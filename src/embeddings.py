"""Shared local sentence-embedding model (extracted M78.1).

A single lazily-loaded SentenceTransformer instance shared by every consumer
that needs local embeddings — currently src/knowledge.py (M46 hybrid knowledge
search) and src/conversation_recall.py (M78.1 semantic conversation recall).

WHY a shared module (the extraction rationale): the model is ~80 MB of weights
plus the torch runtime, so two independent loaders would hold two copies in RAM
whenever both tools are used in a session. This is a SHARED-EXPENSIVE-RESOURCE
extraction, not premature abstraction of logic — the cost of NOT sharing is
concrete and immediate at the second consumer (the same judgement that extracted
src/http_util.py). One module-level singleton, loaded on first use.

Local by REQUIREMENT, not convenience: an API embedding endpoint would ship
private text — the user's curated corpus AND their conversation transcripts — to
a third party, violating privacy-by-design. all-MiniLM-L6-v2: 384-dim, ~80 MB,
CPU-fast on the Ryzen.

Fail-soft contract (the optional-component contract): if sentence-transformers
isn't installed or the model can't load, get_embedder() returns None and embed()
returns None — every caller then degrades cleanly to keyword-only search.
Embeddings must never break the tools that use them.
"""

from __future__ import annotations

import sys
import threading


_EMBED_MODEL = "all-MiniLM-L6-v2"

# Sentinel: None = not yet tried, False = load failed (→ permanent keyword-only
# fallback for the rest of the process), else the loaded model object.
_embedder: object | None = None

# 2026-07-02 QA: the whole point of this module is ONE model instance, but the
# lazy check-then-load was unsynchronized — a tray "Reindex knowledge" racing a
# turn-thread recall_conversation on first use loaded the ~80 MB model twice.
_load_lock = threading.Lock()


def get_embedder():
    """The embedding model, lazily loaded and cached process-wide. Returns None
    if sentence-transformers isn't installed or the model can't load — callers
    fall back to keyword-only search."""
    global _embedder
    with _load_lock:
        if _embedder is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
                print(f"[embeddings] loading model ({_EMBED_MODEL})…", file=sys.stderr)
                _embedder = SentenceTransformer(_EMBED_MODEL)
                print("[embeddings] model ready", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 — degrade, never crash
                print(
                    f"[embeddings] unavailable ({type(exc).__name__}: {exc}) "
                    f"— keyword-only search",
                    file=sys.stderr,
                )
                _embedder = False
        return _embedder or None


def embed(texts: list[str]):
    """Encode texts → an (N, 384) float32 array, L2-normalized so cosine
    similarity reduces to a plain dot product. Returns None on any failure
    (caller falls back to keyword-only)."""
    model = get_embedder()
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
        print(f"[embeddings] encode failed ({exc})", file=sys.stderr)
        return None
