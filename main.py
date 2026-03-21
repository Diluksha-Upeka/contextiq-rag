from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

from services.embeddings import get_embedding_model
from services.ingest import ingest_pdf_bytes
from services.retrieval import generate_answer, retrieve_chunks

load_dotenv()

app = FastAPI(title="ContextIQ API")

# Allow requests from our Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    namespace: str

class Source(BaseModel):
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        content = await file.read()
        # Use a hardcoded namespace for the demo, or generate a UUID if handling multi-users
        namespace = "latest"
        ingest_pdf_bytes(content, namespace=namespace, replace_namespace=True)
        return {"message": "Successfully indexed PDF", "namespace": namespace}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    try:
        embeddings = get_embedding_model()
        contexts = retrieve_chunks(
            embeddings=embeddings,
            query=request.query,
            namespace=request.namespace,
            top_k=2,
        )
        answer = generate_answer(query=request.query, contexts=contexts)
        
        sources = [{"text": chunk} for chunk in contexts]
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)