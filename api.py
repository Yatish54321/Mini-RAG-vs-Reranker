import os
import json
import re
import sqlite3
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
CHUNKS_DB = os.path.join(DATA_DIR, "chunks.db")
PLAIN_FILE = os.path.join(DATA_DIR, "plain.txt")
SUMMARY_FILE = os.path.join(DATA_DIR, "summary.txt")

EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
TOP_K = 5
ABSTAIN_THRESHOLD = 0.30
RERANK_FACTOR = 0.35
RETRIEVE_MULTIPLIER = 5

STOPWORDS = {
    "the","a","an","and","or","in","on","of","to","for","with","is","are","was","were",
    "by","that","this","these","those","it","as","at","from","be","has","have","had",
    "not","but","which","will","can","may","also","such"
}

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Mini-RAG vs Reranker API", version="1.0")

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
def _tokenize(text):
    return re.findall(r"\w+", (text or "").lower())

def keyword_score(query, text):
    qtokens = [t for t in _tokenize(query) if t not in STOPWORDS]
    if not qtokens:
        return 0.0
    tset = set(_tokenize(text))
    return sum(1 for t in qtokens if t in tset) / len(qtokens)

def get_chunk_text(chunk_id):
    if chunk_id is None:
        return ""
    try:
        conn = sqlite3.connect(CHUNKS_DB)
        cur = conn.cursor()
        cur.execute("SELECT chunk_text FROM chunks WHERE id=?", (chunk_id,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception as e:
        print(f"[WARN] DB read failed for chunk_id={chunk_id}: {e}")
        return ""

def embed_query(query: str) -> np.ndarray:
    vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

# -------------------------------
# Baseline Search
# -------------------------------
def baseline_search(query: str, top_k: int = TOP_K):
    q_vec = embed_query(query)
    sims, indices = index.search(q_vec, top_k * RETRIEVE_MULTIPLIER)
    results = []
    seen_texts = set()

    for sim, idx in zip(sims[0], indices[0]):
        if sim < ABSTAIN_THRESHOLD:
            continue
        meta = id_map.get(str(idx), {})
        chunk_id = meta.get("chunk_id")
        text = get_chunk_text(chunk_id)
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        results.append({
            "chunk_id": chunk_id,
            "title": meta.get("title"),
            "source_id": meta.get("source_id"),
            "page_num": meta.get("page_num"),
            "score": float(sim),
            "similarity": float(sim),
            "text": text
        })
        if len(results) >= top_k:
            break
    return results

# -------------------------------
# Hybrid Search (with keyword rerank)
# -------------------------------
def hybrid_search(query: str, top_k: int = TOP_K):
    q_vec = embed_query(query)
    k = min(max(1, top_k * RETRIEVE_MULTIPLIER), index.ntotal)
    sims, indices = index.search(q_vec, k)
    candidates = []
    seen_texts = set()

    for sim, idx in zip(sims[0], indices[0]):
        meta = id_map.get(str(idx), {})
        chunk_id = meta.get("chunk_id")
        text = get_chunk_text(chunk_id)
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        kw = keyword_score(query, text)
        final_score = float(sim) + RERANK_FACTOR * kw
        candidates.append({
            "chunk_id": chunk_id,
            "title": meta.get("title"),
            "source_id": meta.get("source_id"),
            "page_num": meta.get("page_num"),
            "similarity": float(sim),
            "kw_score": float(kw),
            "final_score": float(final_score),
            "text": text
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    filtered = [c for c in candidates if c["similarity"] >= ABSTAIN_THRESHOLD]
    return filtered[:top_k] if filtered else []

# -------------------------------
# Extract Grounded Answer
# -------------------------------
def extract_answer(results):
    if not results:
        return {"answer": None, "citations": [], "reason": "No relevant chunks retrieved."}

    seen_sentences = set()
    combined_text = ""
    citations = []

    for r in results:
        text = r.get("text", "").replace("\n", " ").strip()
        sentences = re.split(r'(?<=[.!?]) +', text)
        for s in sentences:
            s_clean = s.strip()
            if s_clean.lower() not in seen_sentences and s_clean:
                combined_text += s_clean + " "
                seen_sentences.add(s_clean.lower())
        citations.append(f"{r.get('title','')} (p.{r.get('page_num','')})")

    combined_text = combined_text.strip()
    if not combined_text:
        return {"answer": None, "citations": [], "reason": "No relevant text after deduplication."}

    if os.path.exists(PLAIN_FILE):
        with open(PLAIN_FILE, "r", encoding="utf-8") as f:
            combined_text += " " + f.read()
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            combined_text += " " + f.read()

    simplified = combined_text.replace("equipment", "tools or devices")\
                              .replace("employees", "workers")\
                              .replace("amputation", "serious injury")\
                              .replace("machinery", "machines")\
                              .replace("ensure", "make sure")\
                              .replace("procedures", "steps")\
                              .replace("safeguarding", "safety measures")\
                              .replace("protecting", "keeping safe")\
                              .replace("related product", "machine or item")\
                              .replace("hazard", "risk")\
                              .replace("personnel", "people")\
                              .replace("supervisor", "manager")

    sentences = re.split(r'(?<=[.!?]) +', simplified)
    short_answer = " ".join(sentences[:6])
    if len(short_answer) > 500:
        short_answer = short_answer[:500] + "..."
    return {"answer": short_answer.strip(), "citations": citations, "reason": None}

# -------------------------------
# API Endpoints
# -------------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    baseline = baseline_search(request.question)
    hybrid = hybrid_search(request.question)
    answer = extract_answer(hybrid)
    return {"question": request.question, "baseline": baseline, "hybrid": hybrid, "answer": answer}

@app.post("/ask_batch")
def ask_batch(request: QuestionsRequest):
    if not request.questions or not all(q.strip() for q in request.questions):
        raise HTTPException(status_code=400, detail="Questions list cannot be empty or contain empty strings.")
    results = []
    for q in request.questions:
        baseline = baseline_search(q)
        hybrid = hybrid_search(q)
        answer = extract_answer(hybrid)
        results.append({"question": q, "baseline": baseline, "hybrid": hybrid, "answer": answer})
    return {"results": results}

# -------------------------------
print("[INFO] API is ready. Run with `uvicorn api:app --reload`")
