import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import re

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TOP_K = 5  # Top chunks to return
RERANK_FACTOR = 0.3  # How much weight to give to keyword match (0-1)

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
# Helper Functions
# -------------------------------
def embed_query(query):
    """Embed a query text"""
    vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

def keyword_score(query, text):
    """Simple keyword-based scoring"""
    query_words = re.findall(r'\w+', query.lower())
    text_words = re.findall(r'\w+', text.lower())
    if not query_words:
        return 0.0
    count = sum([1 for w in query_words if w in text_words])
    return count / len(query_words)

def hybrid_search(query, top_k=TOP_K):
    """FAISS search + keyword reranker"""
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k * 3)  # retrieve more for reranking

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

    # Apply keyword reranking
    for r in results:
        # Higher keyword match -> lower effective distance (better rank)
        r["score"] = r["distance"] - RERANK_FACTOR * keyword_score(query, get_chunk_text(r["chunk_id"]))

    # Sort by final score
    results.sort(key=lambda x: x["score"])
    return results[:top_k]

def get_chunk_text(chunk_id):
    """Fetch chunk text from SQLite DB"""
    import sqlite3
    db_file = os.path.join(DATA_DIR, "chunks.db")
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT chunk_text FROM chunks WHERE id=?", (chunk_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else ""

# -------------------------------
# Main Demo
# -------------------------------
if __name__ == "__main__":
    print("[INFO] Ready to hybrid search. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit"]:
            break

        print("[INFO] Running hybrid reranker search...")
        top_chunks = hybrid_search(query, top_k=TOP_K)
        for i, chunk in enumerate(top_chunks, 1):
            print(f"\n[{i}] Title: {chunk['title']}, Page: {chunk['page_num']}, Score: {chunk['score']:.4f}")
