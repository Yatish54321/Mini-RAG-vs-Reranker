import os
import json
import sqlite3
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "chunks.db")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")

BATCH_SIZE = 256  # adjust for memory
EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"  # fast, small transformer

# -------------------------------
# Initialize Model
# -------------------------------
print("[INFO] Loading sentence-transformers model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# -------------------------------
# Helper Functions
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
    """Generate embeddings for a list of texts"""
    return model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("[INFO] Loading chunks from DB...")
    chunks = fetch_chunks_from_db(DB_FILE)
    print(f"[INFO] Total chunks fetched: {len(chunks)}")

    # Prepare FAISS index
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized embeddings
    id_map = {}  # vector idx -> metadata

    all_embeddings = []

    print("[INFO] Generating embeddings in batches...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c[1] for c in batch]  # chunk_text
        embeddings = embed_texts(texts)
        all_embeddings.append(embeddings)
        for j, c in enumerate(batch):
            idx = i + j
            id_map[idx] = {
                "chunk_id": c[0],
                "title": c[2],
                "source_id": c[3],
                "page_num": c[4]
            }

    all_embeddings = np.vstack(all_embeddings)

    print(f"[INFO] Adding {all_embeddings.shape[0]} embeddings to FAISS index...")
    index.add(all_embeddings)

    print(f"[INFO] Saving FAISS index to {FAISS_INDEX_FILE} ...")
    faiss.write_index(index, FAISS_INDEX_FILE)

    print(f"[INFO] Saving ID map to {ID_MAP_FILE} ...")
    with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print("[INFO] Embedding process completed successfully!")
