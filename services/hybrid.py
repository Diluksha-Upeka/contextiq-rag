"""Hybrid retrieval: Dense (Pinecone) + BM25 (in-memory) with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi


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


_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _SPLIT_RE.split(text.lower()) if len(tok) >= 2]


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    page: int
    chunk_index: int
    document_name: str
    is_reference: bool
    score: float
    source: str


def bm25_search(query: str, namespace: str, top_k: int = 20) -> list[RetrievalResult]:
    """Retrieve top-k chunks from the in-memory BM25 index."""
    corpus = get_corpus(namespace)
    if corpus is None or corpus.bm25 is None or not corpus.records:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = corpus.bm25.get_scores(tokens)
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
