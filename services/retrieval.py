import os
import re
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

from services.embeddings import embed_query, get_embedding_dimension


Intent = Literal["summary", "analysis", "qa"]


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
    return 3


def normalize_source_text(text: str, max_len: int = 280) -> str:
    """Compact noisy chunk text into a cleaner preview snippet."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    truncated = cleaned[:max_len]
    # Prefer ending at punctuation to avoid awkward cutoff mid-sentence.
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
            f"Pinecone index '{chosen_name}' has dimension {actual_dim}, but embeddings are {dimension}. "
            "Use a different PINECONE_INDEX_NAME or delete/recreate the index with the correct dimension."
        )

    return pc.Index(chosen_name)


def retrieve_chunks(
    embeddings, query: str, namespace: str, top_k: int = 5
) -> list[str]:
    """Retrieve top-k chunks relevant to the query from Pinecone."""
    index = _get_pinecone_index()
    query_vector = embed_query(embeddings, query)

    # Over-fetch, then filter/dedupe to improve context quality.
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
    allow_references = _wants_references(query)

    for match in matches:
        meta = match.get("metadata", {}) or {}
        text = meta.get("text", "")
        if not text:
            continue
        raw_chunks.append(text)

        is_reference = bool(meta.get("is_reference")) or _is_reference_like(text)
        if is_reference and not allow_references:
            continue
        filtered_chunks.append(text)

    deduped = _dedupe_chunks(filtered_chunks)
    if len(deduped) >= top_k:
        return deduped[:top_k]

    # Fallback to raw deduped matches to avoid returning too little context.
    return _dedupe_chunks(raw_chunks)[:top_k]


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model with deterministic settings."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")

    max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=max_output_tokens,
    )


def _extract_finish_reason(response) -> str:
    """Best-effort extraction of provider finish reason for truncation checks."""
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        reason = meta.get("finish_reason") or meta.get("finishReason")
        if isinstance(reason, str):
            return reason.upper()
    return ""


def _as_text(content) -> str:
    """Normalize provider content into plain text."""
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


def _looks_incomplete(text: str, finish_reason: str = "") -> bool:
    """Heuristically detect truncated model output."""
    if not text:
        return False

    normalized = text.rstrip()
    if not normalized:
        return False

    if finish_reason in {"MAX_TOKENS", "LENGTH"}:
        return True

    if len(normalized) < 200:
        return False

    if normalized.endswith((".", "!", "?", '"', "'")):
        return False
    if re.search(r"\[[0-9]+\]\s*$", normalized):
        return False

    if normalized.endswith((":", ",", "-", "(", "/")):
        return True

    return True


def generate_answer(query: str, contexts: list[str], intent: Intent = "qa") -> str:
    """Generate a grounded answer using retrieved context only."""
    llm = _get_llm()
    if not contexts:
        return "I could not find relevant context in the document."

    numbered_context = [f"[{i}] {ctx}" for i, ctx in enumerate(contexts, start=1)]
    context_block = "\n\n".join(numbered_context)

    if intent == "summary":
        task_instructions = (
            "The user requested a summary. Prefer 5 concise bullet points (4-6 if needed), "
            "keep each bullet short, then add a 1-sentence takeaway. Include citation markers like [1], [2] at the end of each bullet "
            "using ONLY the provided context IDs."
        )
    else:
        task_instructions = (
            "Answer the user's question based on the provided context. Be concise but complete. "
            "Include citation markers like [1], [2] next to important claims. "
            "If context is insufficient, say what is missing."
        )

    prompt = (
        "You are an intelligent document assistant. Use only the supplied context.\n"
        f"{task_instructions}\n\n"
        f"Context Chunks:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    answer = _as_text(response.content).strip()
    finish_reason = _extract_finish_reason(response)

    # One continuation pass avoids mid-sentence cutoffs when the model stops on length.
    if _looks_incomplete(answer, finish_reason=finish_reason):
        continuation_prompt = (
            "You were generating an answer from the same context and stopped early. "
            "Continue from exactly where the partial answer ended. Do not repeat prior text. "
            "Keep the same style and citation format, and finish the answer cleanly.\n\n"
            f"Context Chunks:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Partial answer:\n{answer}\n\n"
            "Continuation only:"
        )
        continuation = _as_text(llm.invoke(continuation_prompt).content).strip()
        if continuation:
            answer = f"{answer}\n{continuation}".strip()

    return answer
