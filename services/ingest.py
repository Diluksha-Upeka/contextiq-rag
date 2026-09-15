import os
import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from services.embeddings import embed_texts, get_embedding_dimension, get_embedding_model
from services.hybrid import BM25Corpus, ChunkRecord, set_corpus
from utils.pdf_loader import load_pdf_pages


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

    years = re.findall(r"\b(19|20)\d{2}\b", sample)
    if len(years) >= 3:
        signals += 1

    # Many short "Last, F." style names are common in references.
    initials = re.findall(r"\b[A-Z][a-z]+,\s+[A-Z]\.\b", sample)
    if len(initials) >= 2:
        signals += 1

    return signals >= 3


def _get_pinecone_index(expected_dimension: int):
    """Ensure a Pinecone index exists for the embedding dimension and return it."""
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set")

    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is not set")

    cloud = os.getenv("PINECONE_CLOUD", "aws").strip() or "aws"
    region = os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1"
    dimension = int(expected_dimension)

    pc = Pinecone(api_key=api_key)

    # Pinecone SDK list response shape varies by version; normalize to a set of names.
    list_response = pc.list_indexes()
    existing: set[str] = set()
    if hasattr(list_response, "names") and callable(getattr(list_response, "names")):
        existing = set(list_response.names())
    elif isinstance(list_response, dict):
        indexes = list_response.get("indexes", [])
        for idx in indexes:
            if isinstance(idx, dict) and idx.get("name"):
                existing.add(str(idx["name"]))
    else:
        try:
            for idx in list_response:
                if isinstance(idx, dict) and idx.get("name"):
                    existing.add(str(idx["name"]))
                elif hasattr(idx, "name"):
                    existing.add(str(getattr(idx, "name")))
        except TypeError:
            # Non-iterable response; keep empty and continue.
            pass

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
        # Prefer creating a dimension-suffixed index so changing embedding models
        # doesn't silently break existing data.
        chosen_name = f"{index_name}-{dimension}"

    if chosen_name not in existing:
        pc.create_index(
            name=chosen_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    # Sanity-check: if the index exists, make sure its dimension matches.
    actual_dim = _describe_dim(chosen_name)
    if actual_dim is not None and int(actual_dim) != dimension:
        raise ValueError(
            f"Pinecone index '{chosen_name}' has dimension {actual_dim}, but embeddings are {dimension}. "
            "Use a different PINECONE_INDEX_NAME or delete/recreate the index with the correct dimension."
        )

    return pc.Index(chosen_name)


def _assign_page_to_chunk(chunk_text: str, page_boundaries: list[dict]) -> int:
    """Find which page a chunk most likely belongs to by text overlap."""
    best_page = 1
    best_overlap = 0
    chunk_lower = chunk_text.lower()[:200]  # Use prefix for fast matching

    for pb in page_boundaries:
        page_lower = pb["text"].lower()
        # Simple heuristic: check how much of the chunk prefix appears in the page
        overlap = 0
        words = chunk_lower.split()[:20]
        for word in words:
            if word in page_lower:
                overlap += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = pb["page"]
    return best_page


def ingest_pdf_bytes(
    pdf_bytes: bytes,
    namespace: str,
    replace_namespace: bool = False,
    document_name: str = "document.pdf",
) -> dict:
    """Extract, chunk, embed, and upsert PDF text into Pinecone.

    If replace_namespace is True, the target namespace is cleared first.
    Also builds an in-memory BM25 index for hybrid retrieval.

    Returns metadata about the ingested document.
    """
    pages = load_pdf_pages(pdf_bytes)
    raw_text = "\n\n".join(p["text"] for p in pages)
    if not raw_text.strip():
        raise ValueError("No extractable text found in PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(raw_text)

    # Assign page numbers to each chunk
    chunk_pages = [_assign_page_to_chunk(chunk, pages) for chunk in chunks]

    embeddings = get_embedding_model()
    dimension = get_embedding_dimension()
    index = _get_pinecone_index(expected_dimension=dimension)

    if replace_namespace:
        # Keep only the latest document's vectors in this namespace.
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            # Pinecone returns 404 if the namespace doesn't exist yet.
            msg = str(e)
            if "Namespace not found" not in msg and "(404)" not in msg:
                raise

    vectors = embed_texts(embeddings, chunks)
    payload = []
    bm25_records: list[ChunkRecord] = []

    for i, (chunk, vector, page) in enumerate(zip(chunks, vectors, chunk_pages)):
        chunk_id = f"{namespace}-{uuid.uuid4().hex}-{i}"
        is_ref = _is_reference_like(chunk)

        payload.append(
            (
                chunk_id,
                vector,
                {
                    "text": chunk,
                    "page": page,
                    "chunk_index": i,
                    "document_name": document_name,
                    "is_reference": is_ref,
                },
            )
        )

        bm25_records.append(ChunkRecord(
            chunk_id=chunk_id,
            text=chunk,
            page=page,
            chunk_index=i,
            document_name=document_name,
            is_reference=is_ref,
        ))

    index.upsert(vectors=payload, namespace=namespace)

    # Build BM25 index for hybrid retrieval
    corpus = BM25Corpus(records=bm25_records)
    corpus.build_index()
    set_corpus(namespace, corpus)

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "document_name": document_name,
        "namespace": namespace,
    }
