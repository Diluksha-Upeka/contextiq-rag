import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from services.embeddings import get_embedding_model
from services.ingest import ingest_pdf_bytes
from services.retrieval import generate_answer, retrieve_chunks

load_dotenv()


def _require_env(var_name: str) -> str:
    """Return env var or empty string if missing."""
    return os.getenv(var_name, "").strip()


def _check_required_env() -> bool:
    """Validate required environment variables for the app."""
    required = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [name for name in required if not _require_env(name)]
    if missing:
        st.error(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". See .env.example for setup."
        )
        return False
    return True


def main() -> None:
    st.set_page_config(page_title="ContextIQ", page_icon="📄", layout="wide")
    st.title("ContextIQ")
    st.caption("Sourced answers from your PDFs using Gemini + Pinecone")

    latest_namespace = "latest"

    if not _check_required_env():
        st.stop()

    if "namespace" not in st.session_state:
        st.session_state.namespace = None

    with st.sidebar:
        st.subheader("Upload")
        uploaded = st.file_uploader("PDF file", type=["pdf"])
        process_btn = st.button("Process PDF", type="primary", disabled=uploaded is None)

    if process_btn and uploaded is not None:
        namespace = latest_namespace
        with st.spinner("Extracting, chunking, embedding, and indexing..."):
            ingest_pdf_bytes(uploaded.getvalue(), namespace=namespace, replace_namespace=True)
        st.session_state.namespace = namespace
        st.success("PDF indexed. Ask a question in the main panel.")

    st.subheader("Ask")
    query = st.text_input("Your question", placeholder="What is this document about?")

    if st.button("Get answer", disabled=not query):
        if not st.session_state.namespace:
            st.warning("Upload and process a PDF first.")
        else:
            with st.spinner("Searching and generating response..."):
                embeddings = get_embedding_model()
                contexts = retrieve_chunks(
                    embeddings=embeddings,
                    query=query,
                    namespace=st.session_state.namespace,
                    top_k=5,
                )
                answer = generate_answer(query=query, contexts=contexts)

            st.markdown("### Answer")
            st.write(answer)

            if contexts:
                st.markdown("### Sources")
                for i, chunk in enumerate(contexts, start=1):
                    st.markdown(f"**Chunk {i}:** {chunk}")


if __name__ == "__main__":
    main()
