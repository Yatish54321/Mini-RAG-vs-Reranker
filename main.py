import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TOP_K = 5

# -------------------------------
# Load FAISS index & ID Map
# -------------------------------
@st.cache_resource
def load_index_and_map():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return index, id_map

index, id_map = load_index_and_map()

# -------------------------------
# Load embedding model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

embed_model = load_model()

# -------------------------------
# Embedding function
# -------------------------------
def embed_query(query: str) -> np.ndarray:
    vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

# -------------------------------
# Baseline Search
# -------------------------------
def baseline_search(query: str, top_k: int = TOP_K):
    q_vec = embed_query(query)
    distances, indices = index.search(q_vec, top_k)
    
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
# Hybrid Reranker (Simple re-score)
# -------------------------------
def hybrid_search(query: str, top_k: int = TOP_K):
    q_vec = embed_query(query)
    distances, indices = index.search(q_vec, top_k * 2)  # get more for reranking
    
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

    results = sorted(results, key=lambda x: x["distance"])
    return results[:top_k]

# -------------------------------
# Display results in Streamlit
# -------------------------------
def show_results(question: str, baseline, hybrid):
    st.subheader(f"❓ Question: {question}")

    data = []
    for i in range(TOP_K):
        b = baseline[i] if i < len(baseline) else {}
        h = hybrid[i] if i < len(hybrid) else {}

        data.append({
            "Rank": i + 1,
            "Baseline": f"{b.get('title','')} (p.{b.get('page_num','')})",
            "Hybrid": f"{h.get('title','')} (p.{h.get('page_num','')})"
        })

    df = pd.DataFrame(data)
    st.table(df)

# -------------------------------
# Streamlit UI
# -------------------------------
def run_ui():
    st.set_page_config(page_title="Mini-RAG vs Reranker Sprint", layout="wide")

    st.title("💪 Mini-RAG vs Reranker Sprint")
    st.markdown("Explore **Baseline** vs **Hybrid** retrieval performance interactively.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    mode = st.sidebar.radio("Choose Mode", ["Single Question", "Batch Questions"])

    if mode == "Single Question":
        q = st.text_input("Enter your question:")
        if st.button("Search"):
            if q.strip():
                baseline = baseline_search(q)
                hybrid = hybrid_search(q)
                show_results(q, baseline, hybrid)
            else:
                st.warning("Please enter a valid question.")

    elif mode == "Batch Questions":
        q_text = st.text_area("Enter multiple questions (separated by new lines):")
        if st.button("Run Batch Search"):
            questions = [line.strip() for line in q_text.split("\n") if line.strip()]
            if not questions:
                st.warning("Please enter at least one question.")
            for q in questions:
                with st.expander(f"🔍 Results for: {q}"):
                    baseline = baseline_search(q)
                    hybrid = hybrid_search(q)
                    show_results(q, baseline, hybrid)

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    run_ui()
