"""Hybrid retrieval: Dense (Pinecone) + BM25 (in-memory) with Reciprocal Rank Fusion.

This module maintains an in-memory BM25 corpus per namespace that is populated
during document ingestion.  At query time both retrievers run in parallel and
their results are merged via RRF.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# In-memory BM25 corpus store  (namespace -> corpus)
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """A single chunk stored in the BM25 corpus."""
    chunk_id: str
    text: str
    page: int
    chunk_index: int
    document_name: str
    is_reference: bool


@dataclass
class BM25Corpus:
    """Holds tokenised chunks + BM25 index for one namespace."""
    records: list[ChunkRecord] = field(default_factory=list)
    bm25: BM25Okapi | None = None

    def build_index(self) -> None:
        tokenised = [_tokenize(r.text) for r in self.records]
        if tokenised:
            self.bm25 = BM25Okapi(tokenised)
        else:
            self.bm25 = None


_corpora: dict[str, BM25Corpus] = {}
_lock = threading.Lock()


def set_corpus(namespace: str, corpus: BM25Corpus) -> None:
    with _lock:
        _corpora[namespace] = corpus


def get_corpus(namespace: str) -> BM25Corpus | None:
    with _lock:
        return _corpora.get(namespace)


# ---------------------------------------------------------------------------
# Tokeniser (simple whitespace + lowering, good enough for BM25)
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _SPLIT_RE.split(text.lower()) if len(tok) >= 2]


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Unified result from any retrieval method."""
    chunk_id: str
    text: str
    page: int
    chunk_index: int
    document_name: str
    is_reference: bool
    score: float  # similarity / BM25 / RRF score
    source: str   # "dense" | "bm25" | "rrf"


def bm25_search(query: str, namespace: str, top_k: int = 20) -> list[RetrievalResult]:
    """Retrieve top-k chunks from the in-memory BM25 index."""
    corpus = get_corpus(namespace)
    if corpus is None or corpus.bm25 is None or not corpus.records:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = corpus.bm25.get_scores(tokens)
    # Get top-k indices sorted by score descending
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results: list[RetrievalResult] = []
    for rank_pos in ranked:
        rec = corpus.records[rank_pos]
        results.append(RetrievalResult(
            chunk_id=rec.chunk_id,
            text=rec.text,
            page=rec.page,
            chunk_index=rec.chunk_index,
            document_name=rec.document_name,
            is_reference=rec.is_reference,
            score=float(scores[rank_pos]),
            source="bm25",
        ))
    return results


# ---------------------------------------------------------------------------
# Dense search (Pinecone)
# ---------------------------------------------------------------------------

def dense_search(
    query_vector: list[float],
    index: Any,
    namespace: str,
    top_k: int = 20,
) -> list[RetrievalResult]:
    """Retrieve top-k chunks from Pinecone dense vector search."""
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    matches = result.get("matches", [])

    results: list[RetrievalResult] = []
    for match in matches:
        meta = match.get("metadata", {}) or {}
        text = meta.get("text", "")
        if not text:
            continue
        results.append(RetrievalResult(
            chunk_id=match.get("id", ""),
            text=text,
            page=int(meta.get("page", 0)),
            chunk_index=int(meta.get("chunk_index", 0)),
            document_name=str(meta.get("document_name", "")),
            is_reference=bool(meta.get("is_reference", False)),
            score=float(match.get("score", 0.0)),
            source="dense",
        ))
    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *result_lists: list[RetrievalResult],
    k: int = 60,
) -> list[RetrievalResult]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ  1 / (k + rank_i(d))
    where the sum is across all lists that contain d.
    """
    rrf_scores: dict[str, float] = {}
    best_result: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, r in enumerate(result_list, start=1):
            key = r.chunk_id or r.text[:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Keep the result with the higher original score for metadata
            if key not in best_result or r.score > best_result[key].score:
                best_result[key] = r

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    merged: list[RetrievalResult] = []
    for key in sorted_keys:
        r = best_result[key]
        merged.append(RetrievalResult(
            chunk_id=r.chunk_id,
            text=r.text,
            page=r.page,
            chunk_index=r.chunk_index,
            document_name=r.document_name,
            is_reference=r.is_reference,
            score=rrf_scores[key],
            source="rrf",
        ))
    return merged


# ---------------------------------------------------------------------------
# Full hybrid retrieve
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query_vector: list[float],
    query_text: str,
    index: Any,
    namespace: str,
    dense_top_k: int = 20,
    bm25_top_k: int = 20,
) -> list[RetrievalResult]:
    """Run dense + BM25 retrieval and fuse with RRF."""
    dense_results = dense_search(query_vector, index, namespace, top_k=dense_top_k)
    bm25_results = bm25_search(query_text, namespace, top_k=bm25_top_k)
    return reciprocal_rank_fusion(dense_results, bm25_results)
