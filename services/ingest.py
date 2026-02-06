import os
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from services.embeddings import embed_texts, get_embedding_dimension, get_embedding_model
from utils.pdf_loader import load_pdf_text


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


def ingest_pdf_bytes(pdf_bytes: bytes, namespace: str, replace_namespace: bool = False) -> None:
    """Extract, chunk, embed, and upsert PDF text into Pinecone.

    If replace_namespace is True, the target namespace is cleared first.
    """
    raw_text = load_pdf_text(pdf_bytes)
    if not raw_text.strip():
        raise ValueError("No extractable text found in PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(raw_text)

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
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload.append(
            (
                f"{namespace}-{uuid.uuid4().hex}-{i}",
                vector,
                {"text": chunk},
            )
        )
    index.upsert(vectors=payload, namespace=namespace)
