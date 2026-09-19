"""FastAPI router for ContextIQ Evaluation Lab.

Provides API endpoints for:
- Running evaluation benchmarks on active namespaces
- Retrieving and updating the golden evaluation dataset
- Viewing historical evaluation benchmark runs
- Comparing multiple RAG configurations side-by-side
"""

from __future__ import annotations

from typing import Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from evaluation.runner import (
    EvalConfig,
    load_dataset,
    load_results,
    run_evaluation,
    save_dataset,
)

router = APIRouter(prefix="/api/eval", tags=["Evaluation"])


class RunEvalRequest(BaseModel):
    namespace: str = Field(default="latest", description="Target Pinecone namespace")
    config_name: str = Field(default="Hybrid + Reranker", description="Friendly config name")
    retrieval_mode: str = Field(
        default="hybrid_rerank",
        description="One of: 'dense', 'bm25', 'hybrid', 'hybrid_rerank'",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    run_llm_judge: bool = Field(default=True, description="Whether to execute LLM-as-judge")


class QuestionSchema(BaseModel):
    id: str
    question: str
    expected_answer: str
    category: Optional[str] = "qa"
    relevant_pages: Optional[List[int]] = []
    relevant_keywords: Optional[List[str]] = []


class DatasetSchema(BaseModel):
    name: Optional[str] = "ContextIQ Evaluation Dataset"
    description: Optional[str] = ""
    document_name: Optional[str] = "document.pdf"
    questions: List[QuestionSchema]


class CompareRequest(BaseModel):
    namespace: str = Field(default="latest")
    top_k: int = Field(default=5, ge=1, le=20)
    run_llm_judge: bool = Field(default=False, description="Faster comparison without judge")


@router.get("/dataset")
async def get_dataset():
    """Retrieve the current golden evaluation dataset."""
    try:
        return load_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset")
async def update_dataset(dataset: DatasetSchema):
    """Update or overwrite the evaluation dataset."""
    try:
        save_dataset(dataset.model_dump())
        return {"status": "success", "message": f"Saved {len(dataset.questions)} questions"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_results():
    """Get historical evaluation results."""
    try:
        return load_results()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def run_single_eval(req: RunEvalRequest):
    """Run an evaluation benchmark with the specified configuration."""
    try:
        config = EvalConfig(
            name=req.config_name,
            retrieval_mode=req.retrieval_mode,
            top_k=req.top_k,
            confidence_threshold=req.confidence_threshold,
            run_llm_judge=req.run_llm_judge,
        )
        result = run_evaluation(namespace=req.namespace, config=config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_configurations(req: CompareRequest):
    """Run a multi-configuration benchmark comparing Baseline, Hybrid, and Hybrid+Reranker."""
    try:
        dataset = load_dataset()
        configs = [
            EvalConfig(
                name="1. Baseline (Dense Vector)",
                retrieval_mode="dense",
                top_k=req.top_k,
                run_llm_judge=req.run_llm_judge,
            ),
            EvalConfig(
                name="2. Hybrid (Dense + BM25)",
                retrieval_mode="hybrid",
                top_k=req.top_k,
                run_llm_judge=req.run_llm_judge,
            ),
            EvalConfig(
                name="3. Hybrid + Cross-Encoder Rerank",
                retrieval_mode="hybrid_rerank",
                top_k=req.top_k,
                run_llm_judge=req.run_llm_judge,
            ),
        ]

        runs = []
        for cfg in configs:
            res = run_evaluation(namespace=req.namespace, config=cfg, dataset=dataset)
            runs.append(res)

        return {
            "timestamp": runs[0]["timestamp"] if runs else "",
            "document_name": dataset.get("document_name", ""),
            "total_questions": len(dataset.get("questions", [])),
            "configurations": [
                {
                    "name": r["config"]["name"],
                    "mode": r["config"]["retrieval_mode"],
                    "duration_ms": r["duration_ms"],
                    "metrics": r["average_metrics"],
                }
                for r in runs
            ],
            "full_runs": runs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
