"""Retrieval and Generation Evaluation Metrics.

Implements standard Information Retrieval (IR) metrics:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)

And LLM-as-Judge generation metrics:
- Faithfulness (claim verification against context)
- Answer Relevance (question-answer alignment)
- Context Relevance (retrieval quality for query)
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """Calculate Recall@K: proportion of relevant items retrieved in top-k."""
    if not relevant:
        return 1.0
    top_k_items = retrieved[:k]
    hits = sum(1 for item in top_k_items if item in relevant)
    return round(hits / len(relevant), 4)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """Calculate Precision@K: proportion of top-k items that are relevant."""
    if k <= 0:
        return 0.0
    top_k_items = retrieved[:k]
    hits = sum(1 for item in top_k_items if item in relevant)
    return round(hits / k, 4)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the rank of the first relevant item."""
    if not relevant:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (nDCG@K) with binary relevance."""
    if not relevant or k <= 0:
        return 0.0

    top_k_items = retrieved[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k_items, start=1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG: all relevant items ranked first
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


# ---------------------------------------------------------------------------
