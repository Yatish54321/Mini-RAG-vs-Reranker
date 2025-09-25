import os
import json
import sqlite3
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "chunks.db")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
SUMMARY_FILE = os.path.join(DATA_DIR, "summary.txt")
EMBEDDINGS_CACHE = os.path.join(DATA_DIR, "embeddings.npy")

EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2" 

# -------------------------------
# Device setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
cpu_cores = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
os.environ["MKL_NUM_THREADS"] = str(cpu_cores)

# Batch size auto-set
if device == "cuda":
    BATCH_SIZE = 1024
else:
    BATCH_SIZE = 256

print(f"[INFO] Using device: {device}, CPU cores: {cpu_cores}, Batch size: {BATCH_SIZE}")

# -------------------------------
# Load embedding model
# -------------------------------
print("[INFO] Loading sentence-transformers model...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)

# -------------------------------
# Helper functions
# -------------------------------
def fetch_chunks_from_db(db_file):
    """Fetch all chunks from SQLite DB"""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT id, chunk_text, title, source_id, page_num FROM chunks")
    rows = cur.fetchall()
    conn.close()
    return rows

def embed_texts(texts):
    """Generate embeddings with manual normalization"""
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def load_summary_chunk(summary_file):
    """Load summary chunk with dummy metadata"""
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return [(-1, text, "Summary", "Summary", 0)]
    return []

# -------------------------------
# Main process
# -------------------------------
if __name__ == "__main__":
    chunks = fetch_chunks_from_db(DB_FILE)
    print(f"[INFO] Total chunks fetched: {len(chunks)}")

    # Include summary
    summary_chunk = load_summary_chunk(SUMMARY_FILE)
    if summary_chunk:
        print("[INFO] Summary chunk included.")
        chunks.extend(summary_chunk)

    # Check for existing embeddings cache
    if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(ID_MAP_FILE) and os.path.exists(FAISS_INDEX_FILE):
        print("[INFO] Embeddings cache and FAISS index exist. Loading directly...")
        embeddings = np.load(EMBEDDINGS_CACHE)
        with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"[INFO] Loaded {embeddings.shape[0]} embeddings from cache.")
    else:
        print("[INFO] Generating embeddings (batched)...")
        all_embeddings = []
        id_map = {}

        for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [c[1] for c in batch]
            batch_embeddings = embed_texts(texts)
            all_embeddings.append(batch_embeddings)

            for j, c in enumerate(batch):
                idx = i + j
                id_map[idx] = {
                    "chunk_id": c[0],
                    "title": c[2],
                    "source_id": c[3],
                    "page_num": c[4]
                }

        embeddings = np.vstack(all_embeddings)
        print(f"[INFO] Embeddings generated: {embeddings.shape}")

        # Save embeddings cache
        np.save(EMBEDDINGS_CACHE, embeddings)
        print(f"[INFO] Embeddings saved to cache: {EMBEDDINGS_CACHE}")

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_FILE)
        print(f"[INFO] FAISS index saved: {FAISS_INDEX_FILE}")

        # Save ID map
        with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=2)
        print(f"[INFO] ID map saved: {ID_MAP_FILE}")

    print("[INFO] Embedding process completed successfully!")
