import os

from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec

from services.embeddings import embed_query, get_embedding_dimension


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
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    matches = result.get("matches", [])
    return [m.get("metadata", {}).get("text", "") for m in matches if m.get("metadata")]


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model with deterministic settings."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=512,
    )


def generate_answer(query: str, contexts: list[str]) -> str:
    """Generate a grounded answer using retrieved context only."""
    llm = _get_llm()
    if not contexts:
        return "I could not find relevant context in the document."

    context_block = "\n\n".join(contexts)
    prompt = (
        "You are a PDF assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return response.content
