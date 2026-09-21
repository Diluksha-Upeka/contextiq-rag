"""Evaluation Pipeline Runner.

Orchestrates running benchmark evaluations across different RAG configurations
(Dense Baseline, BM25, Hybrid RRF, Hybrid + Reranker) and computes comprehensive
IR and LLM generation quality metrics.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from evaluation.metrics import (
    evaluate_generation_quality,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

from services.embeddings import embed_query, get_embedding_model
from services.hybrid import (
    RetrievalResult,
    bm25_search,
    dense_search,
    hybrid_retrieve,
)
from services.retrieval import _get_pinecone_index, generate_answer


DATASET_FILE = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")


@dataclass
class EvalConfig:
    name: str = "Hybrid + Reranker"
    retrieval_mode: str = "hybrid_rerank"  # "dense" | "bm25" | "hybrid" | "hybrid_rerank"
    top_k: int = 5
    confidence_threshold: float = 0.35
    run_llm_judge: bool = True


def load_dataset() -> dict[str, Any]:
    """Load evaluation questions from the dataset JSON file."""
    if not os.path.exists(DATASET_FILE):
        return {"document_name": "document.pdf", "questions": []}
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(dataset: dict[str, Any]) -> None:
    """Save updated evaluation questions to dataset JSON."""
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


def load_results() -> list[dict[str, Any]]:
    """Load historical evaluation results."""
    if not os.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_results(results: list[dict[str, Any]]) -> None:
    """Save historical evaluation results."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _is_chunk_relevant(chunk: RetrievalResult, question_data: dict[str, Any]) -> bool:
    """Determine ground-truth relevance of a retrieved chunk."""
    rel_pages = question_data.get("relevant_pages", [])
    rel_keywords = [k.lower() for k in question_data.get("relevant_keywords", [])]

    # If question is out-of-domain, no chunk is truly relevant
    if question_data.get("category") == "out_of_domain":
        return False

    # Check page match
    if rel_pages and chunk.page in rel_pages:
        return True

    # Check keyword match
    chunk_text = chunk.text.lower()
    if rel_keywords and any(kw in chunk_text for kw in rel_keywords):
        return True

    return False


_query_vector_cache: dict[str, list[float]] = {}


def _get_query_vector(embeddings: Any, text: str) -> list[float]:
    """Cached embedding generator to conserve API requests during evaluation."""
    cleaned = text.strip().lower()
    if cleaned not in _query_vector_cache:
        _query_vector_cache[cleaned] = embed_query(embeddings, text)
    return _query_vector_cache[cleaned]


def run_evaluation(
    namespace: str,
    config: EvalConfig,
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute evaluation across the dataset with the specified configuration."""
    if dataset is None:
        dataset = load_dataset()

    questions = dataset.get("questions", [])
    if not questions:
        return {
            "config": asdict(config),
            "timestamp": datetime.utcnow().isoformat(),
            "total_questions": 0,
            "average_metrics": {},
            "question_results": [],
            "error": "Dataset has no questions",
        }

    embeddings = get_embedding_model()
    index = _get_pinecone_index()

    question_results: list[dict[str, Any]] = []
    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_faithfulness = 0.0
    total_answer_rel = 0.0
    total_context_rel = 0.0
    evaluated_gen_count = 0

    eval_start_time = time.time()

    for q in questions:
        q_id = q.get("id", "")
        query_text = q.get("question", "")
        expected = q.get("expected_answer", "")
        category = q.get("category", "qa")

        q_start = time.time()

        # Step 1: Retrieval according to configuration
        candidates: list[RetrievalResult] = []
        if config.retrieval_mode == "dense":
            q_vec = _get_query_vector(embeddings, query_text)
            candidates = dense_search(q_vec, index, namespace=namespace, top_k=config.top_k * 3)
            final_chunks = candidates[: config.top_k]
        elif config.retrieval_mode == "bm25":
            candidates = bm25_search(query_text, namespace=namespace, top_k=config.top_k * 3)
            final_chunks = candidates[: config.top_k]
        elif config.retrieval_mode == "hybrid":
            q_vec = _get_query_vector(embeddings, query_text)
            candidates = hybrid_retrieve(q_vec, query_text, index, namespace=namespace, dense_top_k=20, bm25_top_k=20)
            final_chunks = candidates[: config.top_k]
        else:  # "hybrid_rerank"
            q_vec = _get_query_vector(embeddings, query_text)
            candidates = hybrid_retrieve(q_vec, query_text, index, namespace=namespace, dense_top_k=20, bm25_top_k=20)

            try:
                from services.reranker import rerank
                reranked = rerank(query_text, candidates[:15], top_k=config.top_k)
                final_chunks = [
                    RetrievalResult(
                        chunk_id=r.chunk_id,
                        text=r.text,
                        page=r.page,
                        chunk_index=r.chunk_index,
                        document_name=r.document_name,
                        is_reference=r.is_reference,
                        score=r.rerank_score,
                        source="reranker",
                    )
                    for r in reranked
                ]
            except Exception:
                final_chunks = candidates[: config.top_k]

        # Step 2: Compute Retrieval Metrics
        # Ground truth set of relevant chunk ids among all candidates
        relevant_chunk_ids = {c.chunk_id for c in candidates if _is_chunk_relevant(c, q)}
        retrieved_ids = [c.chunk_id for c in final_chunks]

        rec = recall_at_k(retrieved_ids, relevant_chunk_ids, k=config.top_k)
        prec = precision_at_k(retrieved_ids, relevant_chunk_ids, k=config.top_k)
        mrr_score = mean_reciprocal_rank(retrieved_ids, relevant_chunk_ids)
        ndcg_score = ndcg_at_k(retrieved_ids, relevant_chunk_ids, k=config.top_k)

        total_recall += rec
        total_precision += prec
        total_mrr += mrr_score
        total_ndcg += ndcg_score

        # Step 3: Generation & Generation Metrics
        context_texts = [c.text for c in final_chunks]
        actual_answer = ""
        try:
            actual_answer = generate_answer(query=query_text, contexts=context_texts, intent="qa")
        except Exception as e:
            actual_answer = f"Context retrieved successfully ({len(final_chunks)} passages). Generation paused: {str(e)[:70]}"

        gen_eval: dict[str, Any] = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_relevance": 0.0,
            "unsupported_claims": [],
            "faithfulness_reasoning": "",
        }

        if config.run_llm_judge and actual_answer and "Generation paused" not in actual_answer:
            time.sleep(1.0)  # Rate limit throttle to stay well within free tier limits
            try:
                gen_eval = evaluate_generation_quality(query_text, actual_answer, context_texts)
                if gen_eval.get("faithfulness", 0.0) > 0.0 or gen_eval.get("answer_relevance", 0.0) > 0.0:
                    total_faithfulness += gen_eval.get("faithfulness", 0.0)
                    total_answer_rel += gen_eval.get("answer_relevance", 0.0)
                    total_context_rel += gen_eval.get("context_relevance", 0.0)
                    evaluated_gen_count += 1
            except Exception:
                pass

        q_latency = int((time.time() - q_start) * 1000)

        question_results.append({
            "id": q_id,
            "question": query_text,
            "category": category,
            "expected_answer": expected,
            "actual_answer": actual_answer,
            "latency_ms": q_latency,
            "retrieved_count": len(final_chunks),
            "retrieved_pages": sorted(list({c.page for c in final_chunks if c.page > 0})),
            "retrieval_metrics": {
                "recall_at_k": rec,
                "precision_at_k": prec,
                "mrr": mrr_score,
                "ndcg_at_k": ndcg_score,
            },
            "generation_metrics": {
                "faithfulness": gen_eval.get("faithfulness", 0.0),
                "answer_relevance": gen_eval.get("answer_relevance", 0.0),
                "context_relevance": gen_eval.get("context_relevance", 0.0),
                "unsupported_claims": gen_eval.get("unsupported_claims", []),
                "faithfulness_reasoning": gen_eval.get("faithfulness_reasoning", ""),
            },
        })


    num_q = len(questions)
    gen_divisor = max(evaluated_gen_count, 1)

    avg_metrics = {
        "recall_at_k": round(total_recall / num_q, 4),
        "precision_at_k": round(total_precision / num_q, 4),
        "mrr": round(total_mrr / num_q, 4),
        "ndcg_at_k": round(total_ndcg / num_q, 4),
        "faithfulness": round(total_faithfulness / gen_divisor, 4) if evaluated_gen_count else 0.0,
        "answer_relevance": round(total_answer_rel / gen_divisor, 4) if evaluated_gen_count else 0.0,
        "context_relevance": round(total_context_rel / gen_divisor, 4) if evaluated_gen_count else 0.0,
    }

    eval_result = {
        "id": f"eval-{int(time.time())}",
        "config": asdict(config),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_questions": num_q,
        "duration_ms": int((time.time() - eval_start_time) * 1000),
        "average_metrics": avg_metrics,
        "question_results": question_results,
    }

    # Append to results history
    history = load_results()
    history.insert(0, eval_result)
    # Keep last 20 evaluations
    save_results(history[:20])

    return eval_result
