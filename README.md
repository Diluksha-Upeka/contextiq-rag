# ContextIQ

ContextIQ is a professional "Chat with PDF" Retrieval-Augmented Generation (RAG) app.
Upload a PDF, index it in Pinecone using Gemini embeddings, and ask questions with answers grounded in retrieved context.

## Architecture Overview

```mermaid
graph TD
    A[User] -->|Uploads PDF| B(Streamlit UI)
    B -->|Extracts & splits text| C[LangChain Text Splitter]
    C -->|Generates embeddings| D[Google Gemini Embeddings]
    D -->|Stores vectors| E[(Pinecone Vector DB)]
    
    A -->|Asks Question| B
    B -->|Embeds query| F[Google Gemini Embeddings]
    F -->|Retrieves relevant chunks| E
    E -->|Returns context chunks| G[Google Gemini LLM]
    G -->|Generates grounded answer| B
    B -->|Displays answer| A
```

1. Upload a PDF in the Streamlit UI
2. Extract text and split into chunks (`RecursiveCharacterTextSplitter`)
3. Generate embeddings with Gemini
4. Store vectors in Pinecone
5. Embed the query and retrieve relevant chunks
6. Generate an answer strictly from the retrieved context

## Tech Stack

- Streamlit (UI)
- Google Gemini (LLM + embeddings)
- Pinecone (vector database)
- LangChain (chunking)
- PyPDF (PDF parsing)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create an environment file:

```bash
cp .env.example .env
```

3. Fill in the required values in `.env`:

- `GOOGLE_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

4. Run the app:

```bash
streamlit run app.py
```

## Notes

- The app keeps only the latest uploaded PDF (vectors are stored in a single Pinecone namespace and overwritten on each upload).
- If no relevant context is found, the app will say it does not know.
