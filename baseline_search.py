import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")

TOP_K = 5  
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  

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
    """Embed a query text using the same model as index"""
    vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

def search(query, top_k=TOP_K):
    """Search FAISS index and return top-k chunks"""
    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        metadata = id_map.get(str(idx)) or id_map.get(idx, {})
        results.append({
            "chunk_id": metadata.get("chunk_id"),
            "title": metadata.get("title"),
            "source_id": metadata.get("source_id"),
            "page_num": metadata.get("page_num"),
            "distance": float(dist)
        })
    return results

# -------------------------------
# Main Demo
# -------------------------------
if __name__ == "__main__":
    print("[INFO] Ready to search. Type your question or 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit"]:
            print("[INFO] Exiting search.")
            break

        print("[INFO] Searching top chunks...")
        top_chunks = search(query, top_k=TOP_K)
        if not top_chunks:
            print("[INFO] No results found.")
            continue

        for i, chunk in enumerate(top_chunks, 1):
            print(f"\n[{i}] Title: {chunk['title']}, Page: {chunk['page_num']}, Distance: {chunk['distance']:.4f}")
