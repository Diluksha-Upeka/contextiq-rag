from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import re
from dotenv import load_dotenv

from services.embeddings import get_embedding_model
from services.ingest import ingest_pdf_bytes
from services.retrieval import (
    detect_query_intent,
    generate_answer,
    normalize_source_text,
    retrieve_chunks,
    top_k_for_intent,
)

load_dotenv()

app = FastAPI(title="ContextIQ API")

# Allow requests from local frontend dev servers (Vite/Next).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://contextiq-rag.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

class QueryRequest(BaseModel):
    query: str
    namespace: str

class Source(BaseModel):
    id: int
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    intent: str

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        content = await file.read()
        # Use a hardcoded namespace for the demo, or generate a UUID if handling multi-users
        namespace = "latest"
        ingest_pdf_bytes(content, namespace=namespace, replace_namespace=True)
        return {"message": "Successfully indexed PDF", "namespace": namespace}
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if "429" in msg or "quota" in lower or "rate" in lower:
            retry_after = None
            retry_in_match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", lower)
            retry_delay_match = re.search(r"retry_delay\s*\{[^}]*seconds:\s*([0-9]+)", lower, flags=re.DOTALL)
            if retry_in_match:
                retry_after = int(float(retry_in_match.group(1))) + 1
            elif retry_delay_match:
                retry_after = int(retry_delay_match.group(1)) + 1

            detail = "Embedding provider rate limit/quota exceeded."
            if retry_after is not None:
                detail += f" Retry after about {retry_after}s."
            detail += " If this persists, use a new API key with available quota or upgrade billing."
            raise HTTPException(status_code=429, detail=detail)

        raise HTTPException(status_code=500, detail=msg)

@app.post("/api/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    try:
        embeddings = get_embedding_model()
        intent = detect_query_intent(request.query)
        contexts = retrieve_chunks(
            embeddings=embeddings,
            query=request.query,
            namespace=request.namespace,
            top_k=top_k_for_intent(intent),
        )
        answer = generate_answer(query=request.query, contexts=contexts, intent=intent)
        
        sources = [{"id": i, "text": normalize_source_text(chunk)} for i, chunk in enumerate(contexts, start=1)]
        return QueryResponse(answer=answer, sources=sources, intent=intent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)