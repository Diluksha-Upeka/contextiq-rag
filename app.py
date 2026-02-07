import os

import streamlit as st
from dotenv import load_dotenv

from services.embeddings import get_embedding_model
from services.ingest import ingest_pdf_bytes
from services.retrieval import generate_answer, retrieve_chunks

load_dotenv()


MAX_QUERIES = 3  # Limit queries in the demo to prevent abuse and control costs


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

    # --- Premium minimal styling (Apple-like) ---
    st.markdown(
        """
        <style>
        :root {
            --ci-text: #1d1d1f;
            --ci-subtle: #86868b;
            --ci-border: rgba(60, 60, 67, 0.18);
            --ci-surface: rgba(255, 255, 255, 0.78);
            --ci-surface-strong: rgba(255, 255, 255, 0.92);
            --ci-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            --ci-accent: #0a84ff;
            --ci-accent-2: #0071e3;
        }

        /* Global typography */
        html, body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--ci-text);
        }

        /* Force readable text colors (Streamlit theme can override defaults) */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] * {
            color: var(--ci-text);
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {
            color: var(--ci-text);
        }

        /* Subtle secondary text */
        .stCaption, .stCaption *,
        small, .stMarkdown small {
            color: var(--ci-subtle) !important;
        }

        /* Markdown containers */
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] * {
            color: var(--ci-text);
        }

        /* App background */
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(1200px 600px at 30% 0%, rgba(10, 132, 255, 0.08), transparent 60%),
                        radial-gradient(900px 500px at 90% 10%, rgba(0, 113, 227, 0.06), transparent 55%),
                        #f5f5f7;
        }

        /* Titles - crisp and clean */
        h1 {
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--ci-text);
        }
        h2, h3 {
            font-weight: 500;
            letter-spacing: -0.01em;
            color: var(--ci-text);
        }

        /* App container spacing */
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 980px;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(251, 251, 253, 0.92);
            border-right: 1px solid var(--ci-border);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-weight: 500;
        }

        /* Card-like containers (st.container(border=True)) */
        div[data-testid="stContainer"] {
            background: var(--ci-surface);
            border: 1px solid var(--ci-border);
            border-radius: 18px;
            box-shadow: var(--ci-shadow);
            padding: 1.25rem;
            backdrop-filter: blur(14px);
        }

        /* Input fields */
        .stTextInput input {
            border-radius: 14px;
            border: 1px solid rgba(0, 0, 0, 0.00);
            background: rgba(255, 255, 255, 0.88);
            padding: 14px 16px;
            font-size: 1rem;
            box-shadow: 0 10px 26px rgba(0,0,0,0.06);
            transition: box-shadow 0.2s ease, border-color 0.2s ease, background 0.2s ease;
            color: var(--ci-text);
        }
        .stTextInput input:hover {
            border-color: rgba(60, 60, 67, 0.22);
            background: rgba(255, 255, 255, 0.92);
        }
        .stTextInput input::placeholder {
            color: rgba(134, 134, 139, 0.95);
        }
        .stTextInput input:focus {
            border-color: rgba(10, 132, 255, 0.55);
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 0 0 5px rgba(10, 132, 255, 0.14), 0 14px 34px rgba(0,0,0,0.10);
        }

        /* Buttons - premium + consistent */
        div.stButton > button {
            border-radius: 14px;
            font-weight: 600;
            padding: 0.7rem 1.2rem;
            border: 1px solid var(--ci-border);
            background: var(--ci-surface-strong);
            color: var(--ci-text);
            transition: transform 0.12s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            box-shadow: 0 10px 24px rgba(0,0,0,0.06);
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(10, 132, 255, 0.35);
            box-shadow: 0 16px 30px rgba(0,0,0,0.09);
        }
        div.stButton > button:active {
            transform: translateY(0px);
        }

        /* Primary CTA buttons */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(180deg, rgba(10, 132, 255, 0.98), rgba(0, 113, 227, 0.98));
            color: #ffffff;
            border-color: rgba(10, 132, 255, 0.25);
            box-shadow: 0 14px 30px rgba(0, 113, 227, 0.22);
        }
        div.stButton > button[kind="primary"]:hover {
            border-color: rgba(10, 132, 255, 0.35);
            box-shadow: 0 18px 38px rgba(0, 113, 227, 0.28);
        }

        /* Sidebar buttons should be full-width */
        [data-testid="stSidebar"] div.stButton > button {
            width: 100%;
        }

        /* File Uploader - make it look like a drop zone */
        [data-testid="stFileUploader"] section {
            border-radius: 18px;
            padding: 1.6rem;
            border: 1.8px dashed rgba(60, 60, 67, 0.25);
            background: rgba(255,255,255,0.7);
            text-align: center;
        }

        /* File uploader button ("Browse files") */
        [data-testid="stFileUploader"] button {
            background: var(--ci-accent-2) !important;
            color: #ffffff !important;
            border: 1px solid rgba(0, 113, 227, 0.25) !important;
            border-radius: 14px !important;
            box-shadow: 0 12px 26px rgba(0, 113, 227, 0.18) !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background: var(--ci-accent) !important;
            border-color: rgba(10, 132, 255, 0.35) !important;
        }
        
        /* Custom class for source cards */
        .source-card {
            background: var(--ci-surface-strong);
            border: 1px solid var(--ci-border);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
            line-height: 1.65;
            color: #2c2c2e;
            box-shadow: 0 12px 30px rgba(0,0,0,0.06);
            transition: transform 0.12s ease, box-shadow 0.2s ease;
        }
        .source-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 38px rgba(0,0,0,0.10);
        }

        .answer-card {
            background: var(--ci-surface-strong);
            border: 1px solid rgba(10, 132, 255, 0.25);
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 14px 34px rgba(10, 132, 255, 0.08);
        }

        /* Sidebar collapse/expand control (arrow) */
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] * {
            color: var(--ci-text) !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="collapsedControl"] button {
            background: rgba(255, 255, 255, 0.85) !important;
            border: 1px solid var(--ci-border) !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.10) !important;
        }
        [data-testid="collapsedControl"] svg {
            fill: currentColor !important;
        }

        /* Prevent code blocks from inheriting forced text color oddly */
        pre, code, pre *, code * {
            color: inherit;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header section with more whitespace
    st.markdown("<div style='margin-bottom: 2rem;'>", unsafe_allow_html=True)
    st.title("ContextIQ")
    st.markdown(
        "<p style='font-size: 1.05rem; color: #86868b; margin-top: -0.8rem;'>Sourced answers from your PDFs - fast, grounded, and private to your workspace.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    latest_namespace = "latest"

    if not _check_required_env():
        st.stop()

    if "queries" not in st.session_state:
        st.session_state.queries = 0

    if "namespace" not in st.session_state:
        st.session_state.namespace = None

    with st.sidebar:
        with st.container(border=True):
            st.subheader("Upload")
            uploaded = st.file_uploader("PDF file", type=["pdf"])
            process_btn = st.button(
                "Process PDF",
                type="primary",
                disabled=uploaded is None,
                use_container_width=True,
            )

    if process_btn and uploaded is not None:
        if st.session_state.queries >= MAX_QUERIES:
            st.error("Demo limit reached. Please refresh to try again.")
            st.stop()

        st.session_state.queries += 1
        namespace = latest_namespace
        with st.spinner("Extracting, chunking, embedding, and indexing..."):
            ingest_pdf_bytes(uploaded.getvalue(), namespace=namespace, replace_namespace=True)
        st.session_state.namespace = namespace
        st.success("PDF indexed. Ask a question in the main panel.")

    with st.container(border=True):
        st.subheader("Ask")
        st.caption("Ask a question about the PDF you just uploaded.")
        q_col, btn_col = st.columns([6, 1])
        with q_col:
            query = st.text_input(
                "Question",
                placeholder="Ask about the PDF…",
                label_visibility="collapsed",
            )
        with btn_col:
            ask_btn = st.button(
                "Ask",
                type="primary",
                disabled=not query,
                use_container_width=True,
            )

    if ask_btn:
        if st.session_state.queries >= MAX_QUERIES:
            st.error("Demo limit reached. Please refresh to try again.")
            st.stop()

        st.session_state.queries += 1
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

            with st.container(border=True):
                st.markdown("### Answer")
                st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
                st.write(answer)
                st.markdown("</div>", unsafe_allow_html=True)

                if contexts:
                    st.markdown("<br><h3 style='font-weight: 500; font-size: 1.15rem; color: #1d1d1f;'>Sources</h3>", unsafe_allow_html=True)
                    for i, chunk in enumerate(contexts, start=1):
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #86868b; margin-bottom: 0.45rem;">Source {i}</div>
                                {chunk}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
