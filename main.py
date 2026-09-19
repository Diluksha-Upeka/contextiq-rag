from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import re
from dotenv import load_dotenv

from evaluation.api import router as eval_router
from services.embeddings import get_embedding_model
from services.ingest import ingest_pdf_bytes
from services.retrieval import (
    CONFIDENCE_THRESHOLD,
    detect_query_intent,
    generate_answer,
    normalize_source_text,
    retrieve_chunks,
    run_rag_pipeline,
    top_k_for_intent,
)

load_dotenv()

app = FastAPI(
    title="ContextIQ API",
    description="Engineered RAG System with Hybrid Retrieval, Reranking, Grounding & Evaluation Lab",
    version="2.0.0",
)

# Mount evaluation router
app.include_router(eval_router)


def _parse_retry_after_seconds(message: str) -> int | None:
    """Best-effort parsing for provider retry hints."""
    lower = message.lower()
    retry_in_match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", lower)
    retry_delay_match = re.search(
        r"retry_delay\s*\{[^}]*seconds:\s*([0-9]+)",
        lower,
        flags=re.DOTALL,
    )
    if retry_in_match:
        return int(float(retry_in_match.group(1))) + 1
    if retry_delay_match:
        return int(retry_delay_match.group(1)) + 1
    return None


def _is_quota_or_rate_error(message: str) -> bool:
    """Detect common quota/rate-limit provider errors."""
    lower = message.lower()
    quota_markers = [
        "quota",
        "resource_exhausted",
        "insufficient_quota",
        "rate limit",
        "too many requests",
        "exceeded your current quota",
    ]
    return "429" in message or any(marker in lower for marker in quota_markers)


def _raise_friendly_quota_error(exc: Exception) -> None:
    """Raise a user-friendly HTTP 429 for exhausted API key quota."""
    message = str(exc)
    if not _is_quota_or_rate_error(message):
        return

    retry_after = _parse_retry_after_seconds(message)
    detail = "Your AI API key has reached its usage limit."
    if retry_after is not None:
        detail += f" Please wait about {retry_after}s and try again."
    else:
        detail += " Please try again in a little while."
    detail += " If this keeps happening, add billing or switch to a key with available quota."
    raise HTTPException(status_code=429, detail=detail)


# Allow requests from local frontend dev servers (Vite/Next) and production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://contextiq-rag.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "features": {
            "hybrid_search": True,
            "cross_encoder_rerank": True,
            "grounding_validation": True,
            "evaluation_lab": True,
        },
    }


class QueryRequest(BaseModel):
    query: str
    namespace: str = "latest"
    top_k: Optional[int] = None


class Source(BaseModel):
    id: int
    text: str
    full_text: Optional[str] = None
    page: Optional[int] = 1
    chunk_index: Optional[int] = 0
    document_name: Optional[str] = "document.pdf"
    relevance_score: Optional[float] = 0.0
    chunk_id: Optional[str] = None
    source_type: Optional[str] = "dense"


class RetrievalTraceModel(BaseModel):
    dense_candidates: int
    bm25_candidates: int
    rrf_merged: int
    reranked_top: int
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    grounding_ms: int
    total_ms: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: Optional[float] = 1.0
    is_grounded: Optional[bool] = True
    intent: str
    is_unanswerable: Optional[bool] = False
    retrieval_trace: Optional[RetrievalTraceModel] = None


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = (file.filename or "document.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        content = await file.read()
        namespace = "latest"
        ingest_info = ingest_pdf_bytes(
            content,
            namespace=namespace,
            replace_namespace=True,
            document_name=filename,
        )
        return {
            "message": "Successfully indexed PDF",
            "namespace": namespace,
            "pages": ingest_info.get("pages", 1),
            "chunks": ingest_info.get("chunks", 1),
            "document_name": filename,
        }
    except Exception as e:
        _raise_friendly_quota_error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    try:
        pipeline_output = run_rag_pipeline(
            query=request.query,
            namespace=request.namespace,
            top_k=request.top_k,
        )
        return QueryResponse(
            answer=pipeline_output["answer"],
            sources=[
                Source(
                    id=s["id"],
                    text=s["text"],
                    full_text=s.get("full_text"),
                    page=s.get("page", 1),
                    chunk_index=s.get("chunk_index", 0),
                    document_name=s.get("document_name", "document.pdf"),
                    relevance_score=s.get("relevance_score", 0.0),
                    chunk_id=s.get("chunk_id"),
                    source_type=s.get("source_type", "dense"),
                )
                for s in pipeline_output["sources"]
            ],
            confidence=pipeline_output.get("confidence", 1.0),
            is_grounded=pipeline_output.get("is_grounded", True),
            intent=pipeline_output.get("intent", "qa"),
            is_unanswerable=pipeline_output.get("is_unanswerable", False),
            retrieval_trace=RetrievalTraceModel(**pipeline_output["retrieval_trace"])
            if pipeline_output.get("retrieval_trace")
            else None,
        )
    except Exception as e:
        _raise_friendly_quota_error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)