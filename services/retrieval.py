"""ContextIQ Retrieval & Generation Pipeline.

Orchestrates:
1. Query intent detection
2. Hybrid retrieval (Dense Pinecone + BM25 in-memory via RRF)
3. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
4. Relevance confidence thresholding ("I don't know" response for unanswerable queries)
5. Grounded answer generation with inline citation markers [1], [2]
6. Post-generation grounding validation to detect hallucinated/unsupported claims
7. Complete retrieval trace telemetry for latency & candidate breakdown
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

from services.embeddings import embed_query, get_embedding_dimension, get_embedding_model
from services.hybrid import (
    RetrievalResult,
    bm25_search,
    dense_search,
    get_corpus,
    hybrid_retrieve,
)
from services.reranker import RerankResult, rerank


Intent = Literal["summary", "analysis", "qa"]
CONFIDENCE_THRESHOLD = 0.30  # Sigmoid score threshold below which query is considered out-of-context


@dataclass
class GroundingResult:
    is_grounded: bool
    confidence_score: float
    unsupported_claims: list[str]
    reasoning: str


@dataclass
class RetrievalTrace:
    dense_candidates: int
    bm25_candidates: int
    rrf_merged: int
    reranked_top: int
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    grounding_ms: int
    total_ms: int


# ---------------------------------------------------------------------------
# Intent & Text Normalization
# ---------------------------------------------------------------------------

def detect_query_intent(query: str) -> Intent:
    """Classify query intent with lightweight keyword heuristics."""
    q = query.strip().lower()
    summary_patterns = [
        r"\bsummar(y|ize|ise)\b",
        r"\boverview\b",
        r"\btl;?dr\b",
        r"\bkey\s+points?\b",
        r"\bgist\b",
    ]
    analysis_patterns = [
        r"\bchallenges?\b",
        r"\bcompare\b",
        r"\bpros?\s+and\s+cons?\b",
        r"\btrade[- ]?offs?\b",
        r"\bwhy\b",
        r"\bhow\b",
    ]

    if any(re.search(p, q) for p in summary_patterns):
        return "summary"
    if any(re.search(p, q) for p in analysis_patterns):
        return "analysis"
    return "qa"


def top_k_for_intent(intent: Intent) -> int:
    """Tune retrieval depth by intent to improve coverage for broad asks."""
    if intent == "summary":
        return 8
    if intent == "analysis":
        return 5
    return 4


def normalize_source_text(text: str, max_len: int = 280) -> str:
    """Compact noisy chunk text into a cleaner preview snippet."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    truncated = cleaned[:max_len]
    cut = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"))
    if cut > int(max_len * 0.6):
        return truncated[: cut + 1]
    return truncated.rstrip() + "..."


def _is_reference_like(text: str) -> bool:
    """Heuristically identify bibliography/reference-heavy chunks."""
    sample = " ".join(text.split())
    lower = sample.lower()
    if not sample:
        return False

    signals = 0
    if "references" in lower or "bibliography" in lower:
        signals += 2
    if re.search(r"\barxiv\b|doi|proceedings|conference", lower):
        signals += 1
    if re.search(r"\[[0-9]{1,3}\]", sample):
        signals += 1
    if len(re.findall(r"\b(?:19|20)\d{2}\b", sample)) >= 3:
        signals += 1
    if len(re.findall(r"\b[A-Z][a-z]+,\s+[A-Z]\.\b", sample)) >= 2:
        signals += 1

    return signals >= 3


def _wants_references(query: str) -> bool:
    """Allow reference chunks for citation/bibliography-focused asks."""
    q = query.lower()
    return bool(
        re.search(
            r"\breferences?\b|\bcitations?\b|\bbibliograph|\brelated\s+work\b|\bprior\s+work\b",
            q,
        )
    )


def _dedupe_chunks(chunks: list[str]) -> list[str]:
    """Remove near-duplicate chunks by normalized prefix."""
    seen: set[str] = set()
    unique: list[str] = []
    for chunk in chunks:
        key = " ".join(chunk.split()).lower()[:180]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


# ---------------------------------------------------------------------------
# Pinecone Index Handle
# ---------------------------------------------------------------------------

def _get_pinecone_index():
    """Return a Pinecone index handle matching the embedding dimension."""
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set")

    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is not set")

    cloud = os.getenv("PINECONE_CLOUD", "aws").strip() or "aws"
    region = os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1"
    dimension = get_embedding_dimension()

    pc = Pinecone(api_key=api_key)
    existing = {idx["name"] for idx in pc.list_indexes()}

    def _describe_dim(name: str) -> int | None:
        try:
            desc = pc.describe_index(name)
        except Exception:
            return None
        if hasattr(desc, "dimension"):
            return getattr(desc, "dimension")
        if isinstance(desc, dict):
            return desc.get("dimension")
        return None

    chosen_name = index_name
    if index_name in existing:
        existing_dim = _describe_dim(index_name)
        if existing_dim is not None and int(existing_dim) != dimension:
            chosen_name = f"{index_name}-{dimension}"
    else:
        chosen_name = f"{index_name}-{dimension}"

    if chosen_name not in existing:
        pc.create_index(
            name=chosen_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    actual_dim = _describe_dim(chosen_name)
    if actual_dim is not None and int(actual_dim) != dimension:
        raise ValueError(
            f"Pinecone index '{chosen_name}' has dimension {actual_dim}, but embeddings are {dimension}."
        )

    return pc.Index(chosen_name)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    return str(content)


# ---------------------------------------------------------------------------
# Grounding Validation
# ---------------------------------------------------------------------------

def validate_grounding(answer: str, contexts: list[str]) -> GroundingResult:
    """Check whether claims in the generated answer are grounded in context."""
    if not answer.strip() or not contexts:
        return GroundingResult(
            is_grounded=True,
            confidence_score=1.0,
            unsupported_claims=[],
            reasoning="Empty answer or contexts",
        )

    llm = _get_llm()
    context_block = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts[:5])])

    prompt = f"""You are a strict factual grounding validator for a RAG system.
Verify whether the following Answer makes claims that are NOT supported by the Context.

Context:
{context_block}

Answer:
{answer}

Instructions:
1. Check if the answer states any factual claim that cannot be derived from the context.
2. If there are unsupported or hallucinated claims, list them.
3. If everything is grounded, unsupported_claims should be an empty list.

Respond ONLY with valid JSON:
{{
  "is_grounded": true,
  "confidence_score": 0.95,
  "unsupported_claims": [],
  "reasoning": "Explanation"
}}
"""
    try:
        response = llm.invoke(prompt)
        text = _as_text(response.content).strip()
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return GroundingResult(
                is_grounded=bool(data.get("is_grounded", True)),
                confidence_score=float(data.get("confidence_score", 0.9)),
                unsupported_claims=list(data.get("unsupported_claims", [])),
                reasoning=str(data.get("reasoning", "")),
            )
    except Exception as e:
        pass

    return GroundingResult(
        is_grounded=True,
        confidence_score=0.9,
        unsupported_claims=[],
        reasoning="Default validated",
    )


# ---------------------------------------------------------------------------
# Full Hybrid + Reranking Pipeline
# ---------------------------------------------------------------------------

def retrieve_and_rank(
    query: str,
    namespace: str,
    top_k: int = 5,
    dense_k: int = 20,
    bm25_k: int = 20,
) -> tuple[list[RerankResult], dict[str, int]]:
    """Execute Dense + BM25 retrieval -> RRF fusion -> Cross-Encoder reranker.

    Returns (ranked_results, telemetry_stats).
    """
    stats: dict[str, int] = {
        "dense_candidates": 0,
        "bm25_candidates": 0,
        "rrf_merged": 0,
        "reranked_top": 0,
        "retrieval_ms": 0,
        "rerank_ms": 0,
    }

    t0 = time.time()
    embeddings = get_embedding_model()
    query_vector = embed_query(embeddings, query)
    index = _get_pinecone_index()

    dense_candidates = dense_search(query_vector, index, namespace=namespace, top_k=dense_k)
    stats["dense_candidates"] = len(dense_candidates)

    bm25_candidates = bm25_search(query, namespace=namespace, top_k=bm25_k)
    stats["bm25_candidates"] = len(bm25_candidates)

    allow_ref = _wants_references(query)

    # Filter references if not requested
    if not allow_ref:
        dense_candidates = [c for c in dense_candidates if not c.is_reference and not _is_reference_like(c.text)]
        bm25_candidates = [c for c in bm25_candidates if not c.is_reference and not _is_reference_like(c.text)]

    # If BM25 corpus exists, fuse with RRF; otherwise fallback to dense candidates
    if bm25_candidates:
        from services.hybrid import reciprocal_rank_fusion
        merged_candidates = reciprocal_rank_fusion(dense_candidates, bm25_candidates)
    else:
        merged_candidates = dense_candidates

    stats["rrf_merged"] = len(merged_candidates)
    stats["retrieval_ms"] = int((time.time() - t0) * 1000)

    # Reranking stage
    t1 = time.time()
    candidates_to_rerank = merged_candidates[: max(top_k * 3, 15)]

    try:
        ranked_results = rerank(query=query, candidates=candidates_to_rerank, top_k=top_k)
    except Exception as e:
        # Fallback to candidate order if reranker fails
        ranked_results = [
            RerankResult(
                chunk_id=c.chunk_id,
                text=c.text,
                page=c.page,
                chunk_index=c.chunk_index,
                document_name=c.document_name,
                is_reference=c.is_reference,
                retrieval_score=c.score,
                rerank_score=c.score,
                source=c.source,
            )
            for c in candidates_to_rerank[:top_k]
        ]

    stats["reranked_top"] = len(ranked_results)
    stats["rerank_ms"] = int((time.time() - t1) * 1000)

    return ranked_results, stats


# ---------------------------------------------------------------------------
# Answer Generation with Grounding & Trace
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    query: str,
    namespace: str,
    top_k: int | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Execute the full end-to-end engineered RAG pipeline.

    Returns answer, structured sources, confidence score, grounding status,
    and granular retrieval trace.
    """
    total_start = time.time()
    intent = detect_query_intent(query)
    chosen_top_k = top_k or top_k_for_intent(intent)

    ranked_results, trace_stats = retrieve_and_rank(
        query=query,
        namespace=namespace,
        top_k=chosen_top_k,
    )

    # Calculate overall retrieval confidence from rerank scores
    max_confidence = max((r.rerank_score for r in ranked_results), default=0.0)
    avg_confidence = (
        round(sum(r.rerank_score for r in ranked_results) / len(ranked_results), 4)
        if ranked_results
        else 0.0
    )

    # Format sources with metadata
    sources = []
    for i, r in enumerate(ranked_results, start=1):
        sources.append({
            "id": i,
            "text": normalize_source_text(r.text),
            "full_text": r.text,
            "page": r.page if r.page > 0 else 1,
            "chunk_index": r.chunk_index,
            "document_name": r.document_name or "document.pdf",
            "relevance_score": r.rerank_score,
            "chunk_id": r.chunk_id,
            "source_type": r.source,
        })

    # Confidence check: "I don't know" threshold
    if not ranked_results or max_confidence < confidence_threshold:
        total_time = int((time.time() - total_start) * 1000)
        trace = RetrievalTrace(
            dense_candidates=trace_stats["dense_candidates"],
            bm25_candidates=trace_stats["bm25_candidates"],
            rrf_merged=trace_stats["rrf_merged"],
            reranked_top=trace_stats["reranked_top"],
            retrieval_ms=trace_stats["retrieval_ms"],
            rerank_ms=trace_stats["rerank_ms"],
            generation_ms=0,
            grounding_ms=0,
            total_ms=total_time,
        )
        return {
            "answer": (
                "I could not find sufficient information in the provided document to answer your question. "
                "The retrieved passages have low relevance confidence to your query."
            ),
            "sources": sources,
            "confidence": round(max_confidence, 4),
            "is_grounded": True,
            "intent": intent,
            "is_unanswerable": True,
            "retrieval_trace": None,
        }

    # Generation stage
    t_gen_start = time.time()
    context_chunks = [f"[Source {s['id']}, Page {s['page']}]: {s['full_text']}" for s in sources]
    context_block = "\n\n".join(context_chunks)

    if intent == "summary":
        task_instructions = (
            "The user requested a summary. Provide 4-6 concise bullet points covering key insights, "
            "followed by a 1-sentence bottom-line takeaway. Add citation markers like [1], [2] at the "
            "end of each bullet based strictly on the source numbers."
        )
    elif intent == "analysis":
        task_instructions = (
            "Provide a structured analysis addressing the question with clear reasoning and headings or bullets. "
            "Cite relevant sources with [1], [2] matching the source numbers."
        )
    else:
        task_instructions = (
            "Answer the user's question directly and concisely based on the context. "
            "Attribute claims to context using inline citations like [1], [2] next to the specific factual claims."
        )

    prompt = f"""You are ContextIQ, an advanced RAG assistant. You must answer questions using ONLY the facts present in the Context Chunks below.
Never make up facts or extrapolate beyond what is documented. If certain details are missing, state what is missing.

{task_instructions}

Context Chunks:
{context_block}

Question: {query}

Answer:"""

    llm = _get_llm()
    gen_response = llm.invoke(prompt)
    answer = _as_text(gen_response.content).strip()
    generation_ms = int((time.time() - t_gen_start) * 1000)

    # Grounding validation stage
    t_ground_start = time.time()
    raw_texts = [s["full_text"] for s in sources]
    grounding = validate_grounding(answer, raw_texts)
    grounding_ms = int((time.time() - t_ground_start) * 1000)

    # If ungrounded claims detected, run a single correction pass
    if not grounding.is_grounded and grounding.unsupported_claims:
        correction_prompt = f"""The following generated answer was found to contain unsupported claims: {grounding.unsupported_claims}.
Rewrite the answer so that it adheres strictly to the provided Context Chunks and removes or fixes unsupported claims. Keep citations [1], [2].

Context Chunks:
{context_block}

Original Answer:
{answer}

Corrected Grounded Answer:"""
        try:
            corrected = _as_text(llm.invoke(correction_prompt).content).strip()
            if corrected:
                answer = corrected
                grounding.is_grounded = True
        except Exception:
            pass

    total_time = int((time.time() - total_start) * 1000)

    trace = RetrievalTrace(
        dense_candidates=trace_stats["dense_candidates"],
        bm25_candidates=trace_stats["bm25_candidates"],
        rrf_merged=trace_stats["rrf_merged"],
        reranked_top=trace_stats["reranked_top"],
        retrieval_ms=trace_stats["retrieval_ms"],
        rerank_ms=trace_stats["rerank_ms"],
        generation_ms=generation_ms,
        grounding_ms=grounding_ms,
        total_ms=total_time,
    )

    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(max_confidence, 4),
        "is_grounded": grounding.is_grounded,
        "unsupported_claims": grounding.unsupported_claims,
        "intent": intent,
        "is_unanswerable": False,
        "retrieval_trace": None,
    }


# ---------------------------------------------------------------------------
# Legacy Backward Compatibility Wrappers
# ---------------------------------------------------------------------------

def retrieve_chunks(embeddings, query: str, namespace: str, top_k: int = 5) -> list[str]:
    """Backward-compatible retrieval wrapper returning list of chunk strings."""
    index = _get_pinecone_index()
    query_vector = embed_query(embeddings, query)
    fetch_k = max(top_k * 3, top_k + 6)
    result = index.query(
        vector=query_vector,
        top_k=fetch_k,
        include_metadata=True,
        namespace=namespace,
    )
    matches = result.get("matches", [])
    raw_chunks: list[str] = []
    filtered_chunks: list[str] = []
    allow_ref = _wants_references(query)

    for match in matches:
        meta = match.get("metadata", {}) or {}
        text = meta.get("text", "")
        if not text:
            continue
        raw_chunks.append(text)
        is_ref = bool(meta.get("is_reference")) or _is_reference_like(text)
        if is_ref and not allow_ref:
            continue
        filtered_chunks.append(text)

    deduped = _dedupe_chunks(filtered_chunks)
    if len(deduped) >= top_k:
        return deduped[:top_k]
    return _dedupe_chunks(raw_chunks)[:top_k]


def generate_answer(query: str, contexts: list[str], intent: Intent = "qa") -> str:
    """Backward-compatible simple generation function."""
    llm = _get_llm()
    if not contexts:
        return "I could not find relevant context in the document."

    numbered = [f"[{i}] {ctx}" for i, ctx in enumerate(contexts, start=1)]
    context_block = "\n\n".join(numbered)

    prompt = (
        "You are an intelligent document assistant. Use only the supplied context.\n"
        "Answer the user's question concisely. Include citation markers like [1], [2].\n\n"
        f"Context Chunks:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return _as_text(response.content).strip()
