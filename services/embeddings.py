import os
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Create a Gemini embeddings client."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    """Return the embedding vector dimension for the configured model.

    Note: this makes a single embedding request and caches the result for the
    lifetime of the process.
    """
    embeddings = get_embedding_model()
    vector = embeddings.embed_query("dimension probe")
    return len(vector)


def embed_texts(embeddings: GoogleGenerativeAIEmbeddings, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into vectors."""
    return embeddings.embed_documents(texts)


def embed_query(embeddings: GoogleGenerativeAIEmbeddings, text: str) -> list[float]:
    """Embed a single query string."""
    return embeddings.embed_query(text)
