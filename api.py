import os
import json
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from sentence_transformers import SentenceTransformer

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TOP_K = 5

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Mini-RAG Safety Docs API", version="1.0")

# -------------------------------
# Load FAISS index and ID map
# -------------------------------
print("[INFO] Loading FAISS index and ID map...")
index = faiss.read_index(FAISS_INDEX_FILE)
with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
    id_map = json.load(f)

# -------------------------------
# Load embedding model
# -------------------------------
print(f"[INFO] Loading embedding model ({EMBEDDING_MODEL})...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# -------------------------------
# Pydantic request models
# -------------------------------
class QuestionRequest(BaseModel):
    question: str

class QuestionsRequest(BaseModel):
    questions: List[str]

# -------------------------------
# Helper Functions
# -------------------------------
def embed_query(query: str) -> np.ndarray:
    """Embed query using the same model as FAISS index."""
    vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

def hybrid_search(query: str, top_k: int = TOP_K):
    """Perform hybrid reranker search (FAISS + embedding similarity)."""
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        metadata = id_map.get(str(idx), {})
        results.append({
            "chunk_id": metadata.get("chunk_id"),
            "title": metadata.get("title"),
            "source_id": metadata.get("source_id"),
            "page_num": metadata.get("page_num"),
            "distance": float(dist)
        })
    return results

# -------------------------------
# API Endpoints
# -------------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Single question endpoint."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    print(f"[INFO] Received question: {request.question}")
    top_chunks = hybrid_search(request.question, top_k=TOP_K)
    
    return {"question": request.question, "top_chunks": top_chunks}

@app.post("/ask_batch")
def ask_batch(request: QuestionsRequest):
    """Batch questions endpoint."""
    if not request.questions or not all(q.strip() for q in request.questions):
        raise HTTPException(status_code=400, detail="Questions list cannot be empty or contain empty strings.")
    
    results = []
    for q in request.questions:
        print(f"[INFO] Received question: {q}")
        top_chunks = hybrid_search(q, top_k=TOP_K)
        results.append({"question": q, "top_chunks": top_chunks})
    
    return {"results": results}

# -------------------------------
# Run info
# -------------------------------
print("[INFO] API is ready. Use `uvicorn api:app --reload` to run the server.")
