"""Cross-encoder reranker for precise relevance scoring.

Uses a lightweight cross-encoder model that scores (query, passage) pairs
much more accurately than bi-encoder cosine similarity, at the cost of
being slower (hence used only on the small candidate set after initial
retrieval).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from services.hybrid import RetrievalResult

# ---------------------------------------------------------------------------
# Lazy-loaded singleton cross-encoder
# ---------------------------------------------------------------------------

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the cross-encoder model on first use (thread-safe singleton)."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME)
        return _model


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """A retrieval result with an additional cross-encoder relevance score."""
    chunk_id: str
    text: str
    page: int
    chunk_index: int
    document_name: str
    is_reference: bool
    retrieval_score: float   # original retrieval / RRF score
    rerank_score: float      # cross-encoder relevance score (0-1 range)
    source: str


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 5,
) -> list[RerankResult]:
    """Score each (query, candidate) pair with the cross-encoder and return top-k.

    If fewer candidates than top_k, all are returned (still reranked).
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [[query, c.text] for c in candidates]
    scores = model.predict(pairs)

    import numpy as np

    scored = []
    for candidate, score in zip(candidates, scores):
        prob = float(1.0 / (1.0 + np.exp(-float(score))))
        scored.append(RerankResult(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            page=candidate.page,
            chunk_index=candidate.chunk_index,
            document_name=candidate.document_name,
            is_reference=candidate.is_reference,
            retrieval_score=candidate.score,
            rerank_score=round(prob, 4),
            source=candidate.source,
        ))

    scored.sort(key=lambda r: r.rerank_score, reverse=True)
    return scored[:top_k]

