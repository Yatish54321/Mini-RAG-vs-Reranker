import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
import re
import sqlite3

# ------------------------------- CONFIG -------------------------------
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

# ------------------------------- Load FAISS + ID Map -------------------------------
@st.cache_resource
def load_index_and_map():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return index, id_map

index, id_map = load_index_and_map()

# ------------------------------- Load Embedding Model -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

embed_model = load_model()

# ------------------------------- Helper Functions -------------------------------
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

def embed_query(query: str) -> np.ndarray:
    vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)

# ------------------------------- Baseline Search -------------------------------
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

# ------------------------------- Hybrid Reranker -------------------------------
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
        final_score = float(sim) + RERANK_FACTOR * float(kw)
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

# ------------------------------- Grounded & Simplified Answer -------------------------------
def extract_answer(results: list, query: str):
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

# ------------------------------- Display Results -------------------------------
def show_results(question: str, baseline, hybrid):
    st.subheader(f"❓ Question: {question}")

    baseline_ans = extract_answer(baseline, question)
    hybrid_ans = extract_answer(hybrid, question)

    st.markdown("### 📝 Answers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline**")
        if baseline_ans["answer"]:
            st.success(baseline_ans["answer"])
            if baseline_ans["citations"]:
                st.caption(f"Citation: {baseline_ans['citations'][0]}")
        else:
            st.warning(baseline_ans["reason"])
    with col2:
        st.markdown("**Hybrid**")
        if hybrid_ans["answer"]:
            st.success(hybrid_ans["answer"])
            if hybrid_ans["citations"]:
                st.caption(f"Citation: hybrid citations: {', '.join(hybrid_ans['citations'])}")
        else:
            st.warning(hybrid_ans["reason"])

    st.markdown("### 📑 Retrieved Chunks (Top-5)")
    data = []
    for i in range(TOP_K):
        b = baseline[i] if i < len(baseline) else {}
        h = hybrid[i] if i < len(hybrid) else {}
        data.append({
            "Rank": i + 1,
            "Baseline": f"{b.get('title','')} (p.{b.get('page_num','')}) | score={b.get('score','')}" if b else "",
            "Hybrid": f"{h.get('title','')} (p.{h.get('page_num','')}) | score={h.get('final_score','')}" if h else ""
        })

    df = pd.DataFrame(data)

    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'),
                                     ('background-color', '#4B8BBE'),
                                     ('color', 'white'),
                                     ('padding', '8px')]},
        {'selector': 'td', 'props': [('text-align', 'center'),
                                     ('padding', '8px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
        {'selector': 'tr:hover', 'props': [('background-color', '#e0f0ff')]}
    ]).hide(axis="index")  

    st.dataframe(styled_df, use_container_width=True)

# ------------------------------- Streamlit UI -------------------------------
def run_ui():
    st.set_page_config(page_title="Mini-RAG vs Reranker Sprint", layout="wide")

    # ---------------- Header ----------------
    st.markdown("""
        <div style="background: linear-gradient(90deg, #4B8BBE, #306998);
                    padding:25px; border-radius:12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom:20px;">
            <h1 style="color:white;text-align:center; margin-bottom:5px;">💪 Mini-RAG vs Reranker Sprint</h1>
            <p style="color:white;text-align:center; font-size:16px; margin-top:0;">
                Compare Baseline vs Hybrid retrieval and expected grounded answers
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("Select the mode below ⚡")
    mode = st.sidebar.radio("Mode", ["Single Question", "Batch Questions"])

    # ---------------- Single Question Mode ----------------
    if mode == "Single Question":
        st.subheader("🔹 Single Question Mode")
        q = st.text_input("Enter your question here:")
        if st.button("Search"):
            if q.strip():
                with st.spinner("Searching documents... 🔍"):
                    baseline = baseline_search(q)
                    hybrid = hybrid_search(q)
                    show_results(q, baseline, hybrid)
            else:
                st.warning("⚠️ Please enter a valid question.")

    # ---------------- Batch Questions Mode ----------------
    elif mode == "Batch Questions":
        st.subheader("🔹 Batch Questions Mode")
        q_text = st.text_area("Enter multiple questions (one per line):")
        if st.button("Run Batch Search"):
            questions = [line.strip() for line in q_text.split("\n") if line.strip()]
            if not questions:
                st.warning("⚠️ Please enter at least one question.")
            else:
                st.info(f"Running batch search for {len(questions)} questions...")
                progress_bar = st.progress(0)
                for i, q in enumerate(questions):
                    with st.expander(f"🔍 Results for: {q}", expanded=True):
                        baseline = baseline_search(q)
                        hybrid = hybrid_search(q)
                        show_results(q, baseline, hybrid)
                    progress_bar.progress((i + 1) / len(questions))
                st.success("✅ Batch search completed!")

# ------------------------------- Entry Point -------------------------------
if __name__ == "__main__":
    run_ui()
