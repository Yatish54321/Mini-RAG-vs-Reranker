import os
import json
import faiss
import numpy as np
import re
import sqlite3
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
CHUNKS_DB = os.path.join(DATA_DIR, "chunks.db")

EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
RETRIEVE_MULTIPLIER = 5
TOP_K_DEFAULT = 5
SIMILARITY_THRESHOLD = 0.30  

STOPWORDS = {
    "the","a","an","and","or","in","on","of","to","for","with","is","are","was","were",
    "by","that","this","these","those","it","as","at","from","be","has","have","had",
    "not","but","which","will","can","may","also","such"
}

# -------- Load FAISS + id_map --------
if not os.path.exists(FAISS_INDEX_FILE):
    raise SystemExit(f"[ERROR] FAISS index not found at {FAISS_INDEX_FILE}")

if not os.path.exists(ID_MAP_FILE):
    raise SystemExit(f"[ERROR] id_map not found at {ID_MAP_FILE}")

index = faiss.read_index(FAISS_INDEX_FILE)
with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
    id_map = json.load(f)

# -------- Lazy load embedding model --------
_embed_model = None
def _get_model():
    global _embed_model
    if _embed_model is None:
        print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model

# -------- Helper Functions --------
def _tokenize(text):
    return re.findall(r"\w+", (text or "").lower())

def keyword_score(query, text):
    qtokens = [t for t in _tokenize(query) if t not in STOPWORDS]
    if not qtokens:
        return 0.0
    tset = set(_tokenize(text))
    count = sum(1 for q in qtokens if q in tset)
    return count / len(qtokens)

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

def embed_query(query):
    model = _get_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

# -------- Hybrid Reranker --------
def hybrid_search(query, top_k=TOP_K_DEFAULT, rerank_factor=0.35, retrieval_multiplier=RETRIEVE_MULTIPLIER):
    qvec = embed_query(query)
    k = min(max(1, top_k * retrieval_multiplier), index.ntotal)
    distances, indices = index.search(qvec, k)

    candidates = []
    for sim, idx in zip(distances[0], indices[0]):
        meta = id_map.get(str(int(idx)), {})
        chunk_id = meta.get("chunk_id")
        text = get_chunk_text(chunk_id)
        kw = keyword_score(query, text)
        final = float(sim) + float(rerank_factor) * float(kw)
        candidates.append({
            "chunk_id": chunk_id,
            "title": meta.get("title"),
            "source_id": meta.get("source_id"),
            "page_num": meta.get("page_num"),
            "similarity": float(sim),
            "kw_score": float(kw),
            "final_score": float(final),
            "chunk_text": text
        })

    # Sort descending by final_score
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Filter by similarity threshold
    filtered = [c for c in candidates if c["similarity"] >= SIMILARITY_THRESHOLD]

    return filtered[:top_k] if filtered else []

# -------- CLI --------
if __name__ == "__main__":
    print("[INFO] reranker_hybrid ready. Only reranker results will be shown.")
    print("Type query or 'exit'.")

    while True:
        q = input("\nEnter your query: ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break

        results = hybrid_search(q, top_k=TOP_K_DEFAULT)
        if not results:
            print("\n[NO RELEVANT RESULTS FOUND]")
        else:
            print("\n[HYBRID RERANK RESULTS]")
            for i, h in enumerate(results, 1):
                print(f"[{i}] Title: {h['title']} | Page: {h['page_num']} | "
                      f"Similarity: {h['similarity']:.4f} | Keyword Score: {h['kw_score']:.3f} | "
                      f"Final Score: {h['final_score']:.4f}")
