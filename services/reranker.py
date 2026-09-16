"""Cross-encoder reranker for precise relevance scoring."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from services.hybrid import RetrievalResult

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME)
        return _model


@dataclass
class RerankResult:
    chunk_id: str
    text: str
    page: int
    chunk_index: int
    document_name: str
    is_reference: bool
    retrieval_score: float
    rerank_score: float
    source: str


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 5,
) -> list[RerankResult]:
    if not candidates:
        return []

    model = _get_model()
    pairs = [[query, c.text] for c in candidates]
    scores = model.predict(pairs)

    scored = []
    for candidate, score in zip(candidates, scores):
        scored.append(RerankResult(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            page=candidate.page,
            chunk_index=candidate.chunk_index,
            document_name=candidate.document_name,
            is_reference=candidate.is_reference,
            retrieval_score=candidate.score,
            rerank_score=float(score),
            source=candidate.source,
        ))

    scored.sort(key=lambda r: r.rerank_score, reverse=True)
    return scored[:top_k]
