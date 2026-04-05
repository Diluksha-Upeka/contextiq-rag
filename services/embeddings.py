import os
import re
import time
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
    """Embed a list of texts into vectors with light retry for transient 429s."""
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            message = str(e)
            lower = message.lower()
            is_rate_limited = "429" in message or "quota" in lower or "rate" in lower
            if not is_rate_limited or attempt == max_attempts:
                raise

            # Respect provider hint when present: "retry in 44.49s" or "retry_delay { seconds: 44 }"
            wait_seconds = 4 * attempt
            retry_in_match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", lower)
            retry_delay_match = re.search(r"retry_delay\s*\{[^}]*seconds:\s*([0-9]+)", lower, flags=re.DOTALL)
            if retry_in_match:
                wait_seconds = max(wait_seconds, int(float(retry_in_match.group(1))) + 1)
            elif retry_delay_match:
                wait_seconds = max(wait_seconds, int(retry_delay_match.group(1)) + 1)

            time.sleep(min(wait_seconds, 60))

    # Unreachable, loop either returns or raises.
    raise RuntimeError("Embedding failed unexpectedly")


def embed_query(embeddings: GoogleGenerativeAIEmbeddings, text: str) -> list[float]:
    """Embed a single query string."""
    return embeddings.embed_query(text)
