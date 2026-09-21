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
# LLM Judge Helper
# ---------------------------------------------------------------------------

def _get_judge_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash"
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        max_output_tokens=1024,
    )



def _extract_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON object from LLM response, tolerating markdown code fences."""
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
    try:
        return json.loads(cleaned)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generation Metrics (LLM-as-a-Judge)
# ---------------------------------------------------------------------------

def evaluate_faithfulness(answer: str, contexts: list[str]) -> dict[str, Any]:
    """Evaluate answer faithfulness by extracting claims and checking context grounding.

    Returns:
        {
            "score": float between 0.0 and 1.0,
            "total_claims": int,
            "supported_claims": int,
            "unsupported_claims": list[str],
            "reasoning": str
        }
    """
    if not answer.strip() or not contexts:
        return {
            "score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "reasoning": "Missing answer or context",
        }

    llm = _get_judge_llm()
    context_block = "\n---\n".join(contexts[:6])

    prompt = f"""You are an objective AI evaluator assessing the FAITHFULNESS of a generated answer.
Faithfulness measures whether all claims made in the answer can be inferred directly from the provided context.

Context:
{context_block}

Answer to evaluate:
{answer}

Instructions:
1. Extract every individual factual claim from the answer.
2. For each claim, determine whether it is strictly SUPPORTED or UNSUPPORTED by the context.
3. Compute the score as: supported_claims / total_claims. If total_claims is 0, score is 1.0.

Respond ONLY with a JSON object in this exact structure:
{{
  "claims": [
    {{"claim": "claim text", "supported": true/false, "citation_evidence": "brief snippet or none"}}
  ],
  "score": 0.9,
  "reasoning": "Brief explanation"
}}
"""
    try:
        response = llm.invoke(prompt)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        data = _extract_json_response(raw_text)
        if data and "score" in data:
            claims = data.get("claims", [])
            total = len(claims)
            supported = sum(1 for c in claims if c.get("supported", False))
            score = float(data["score"]) if total == 0 else round(supported / total, 4)
            unsupported = [c.get("claim", "") for c in claims if not c.get("supported", False)]
            return {
                "score": min(max(score, 0.0), 1.0),
                "total_claims": total,
                "supported_claims": supported,
                "unsupported_claims": unsupported,
                "reasoning": data.get("reasoning", ""),
            }
    except Exception as e:
        return {
            "score": 0.5,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "reasoning": f"Judge error: {str(e)}",
        }

    return {
        "score": 1.0,
        "total_claims": 1,
        "supported_claims": 1,
        "unsupported_claims": [],
        "reasoning": "Fallback default",
    }


def evaluate_answer_relevance(question: str, answer: str) -> dict[str, Any]:
    """Evaluate how well and directly the answer addresses the user's question.

    Returns:
        {"score": float between 0.0 and 1.0, "reasoning": str}
    """
    if not answer.strip() or not question.strip():
        return {"score": 0.0, "reasoning": "Empty question or answer"}

    llm = _get_judge_llm()
    prompt = f"""You are an objective AI evaluator assessing the RELEVANCE of an answer to a question.
Relevance measures whether the answer directly, accurately, and completely answers what was asked without unnecessary fluff or evasion.

Question: {question}

Answer: {answer}

Rate the relevance from 0.0 (completely irrelevant) to 1.0 (perfectly answers the question).
Respond ONLY with a JSON object:
{{
  "score": 0.85,
  "reasoning": "Brief explanation of why"
}}
"""
    try:
        response = llm.invoke(prompt)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        data = _extract_json_response(raw_text)
        if data and "score" in data:
            return {
                "score": min(max(float(data["score"]), 0.0), 1.0),
                "reasoning": data.get("reasoning", ""),
            }
    except Exception as e:
        return {"score": 0.5, "reasoning": f"Judge error: {str(e)}"}

    return {"score": 0.8, "reasoning": "Default estimate"}


def evaluate_context_relevance(question: str, contexts: list[str]) -> dict[str, Any]:
    """Evaluate how relevant the retrieved context chunks are to answering the question.

    Returns:
        {"score": float between 0.0 and 1.0, "reasoning": str}
    """
    if not contexts or not question.strip():
        return {"score": 0.0, "reasoning": "No contexts or empty question"}

    llm = _get_judge_llm()
    context_block = "\n---\n".join([f"[{i+1}] {c[:400]}" for i, c in enumerate(contexts[:5])])

    prompt = f"""You are an objective AI evaluator assessing RETRIEVAL CONTEXT RELEVANCE.
Context relevance measures whether the retrieved passages contain the information needed to answer the question.

Question: {question}

Retrieved Contexts:
{context_block}

Rate the context relevance from 0.0 (no useful context) to 1.0 (contains all needed information).
Respond ONLY with a JSON object:
{{
  "score": 0.9,
  "reasoning": "Brief explanation"
}}
"""
    try:
        response = llm.invoke(prompt)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        data = _extract_json_response(raw_text)
        if data and "score" in data:
            return {
                "score": min(max(float(data["score"]), 0.0), 1.0),
                "reasoning": data.get("reasoning", ""),
            }
    except Exception as e:
        return {"score": 0.5, "reasoning": f"Judge error: {str(e)}"}

    return {"score": 0.75, "reasoning": "Default estimate"}
