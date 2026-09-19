"""Unit tests for ContextIQ Engineered RAG Pipeline & Evaluation Metrics."""

import os
import pytest
from evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from services.hybrid import (
    RetrievalResult,
    reciprocal_rank_fusion,
    _tokenize,
)
from services.retrieval import detect_query_intent, normalize_source_text
from utils.pdf_loader import load_pdf_pages


def test_reciprocal_rank_fusion():
    """Verify RRF correctly boosts items present in both Dense and BM25 lists."""
    list1 = [
        RetrievalResult(chunk_id="chunk_A", text="Text A", page=1, chunk_index=0, document_name="doc.pdf", is_reference=False, score=0.9, source="dense"),
        RetrievalResult(chunk_id="chunk_B", text="Text B", page=2, chunk_index=1, document_name="doc.pdf", is_reference=False, score=0.8, source="dense"),
    ]
    list2 = [
        RetrievalResult(chunk_id="chunk_B", text="Text B", page=2, chunk_index=1, document_name="doc.pdf", is_reference=False, score=12.5, source="bm25"),
        RetrievalResult(chunk_id="chunk_C", text="Text C", page=3, chunk_index=2, document_name="doc.pdf", is_reference=False, score=10.0, source="bm25"),
    ]

    fused = reciprocal_rank_fusion(list1, list2, k=60)
    # chunk_B is at rank 2 in list1 and rank 1 in list2:
    # RRF score for B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~= 0.01613 + 0.01639 = 0.0325
    # RRF score for A: 1/(60+1) ~= 0.01639
    # RRF score for C: 1/(60+2) ~= 0.01613
    assert len(fused) == 3
    assert fused[0].chunk_id == "chunk_B"  # Present in both lists -> ranked #1!
    assert fused[1].chunk_id == "chunk_A"
    assert fused[2].chunk_id == "chunk_C"


def test_retrieval_metrics():
    """Verify standard Information Retrieval evaluation metrics."""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc2", "doc4", "doc6"}

    # Recall@5: 2 out of 3 relevant documents found = 2/3 ~= 0.6667
    assert recall_at_k(retrieved, relevant, k=5) == 0.6667

    # Precision@5: 2 out of 5 retrieved documents are relevant = 2/5 = 0.40
    assert precision_at_k(retrieved, relevant, k=5) == 0.40

    # MRR: First relevant hit is at rank 2 ("doc2") -> 1/2 = 0.50
    assert mean_reciprocal_rank(retrieved, relevant) == 0.50

    # nDCG@5: Ranking quality check
    ndcg = ndcg_at_k(retrieved, relevant, k=5)
    assert 0.0 < ndcg <= 1.0


def test_intent_detection():
    """Verify query intent classification."""
    assert detect_query_intent("summarize this paper in 5 points") == "summary"
    assert detect_query_intent("give me an overview of the architecture") == "summary"
    assert detect_query_intent("what are the challenges and trade-offs?") == "analysis"
    assert detect_query_intent("compare approach A and approach B") == "analysis"
    assert detect_query_intent("what is the learning rate?") == "qa"


def test_normalize_source_text():
    """Verify clean truncation of snippet text."""
    long_text = "This is a sentence. " * 30
    snippet = normalize_source_text(long_text, max_len=100)
    assert len(snippet) <= 104
    assert snippet.endswith(".") or snippet.endswith("...")


def test_pdf_page_loader():
    """Verify page-aware PDF loader on test.pdf if present."""
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "test.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        pages = load_pdf_pages(pdf_bytes)
        assert len(pages) > 0
        assert "page" in pages[0]
        assert "text" in pages[0]
        assert pages[0]["page"] == 1
