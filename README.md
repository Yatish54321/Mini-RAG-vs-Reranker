
````markdown
# Mini-RAG vs Reranker Sprint 🏃‍♂️

This project is a mini retrieval-augmented generation (RAG) system designed to query and retrieve safety document information efficiently. It combines FAISS-based embedding search with Sentence Transformers for semantic similarity, along with a Streamlit UI and a FastAPI backend. The system supports both single-question and batch queries, provides a grounded answer with citations, and can abstain when no relevant information is found.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
````

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure the following data files are present in the `data/` directory:

   * `faiss_index.index` (FAISS index of embeddings)
   * `id_map.json` (mapping of FAISS indices to chunk metadata)
   * `chunks.db` (SQLite database with document chunks)
   * Optional: `plain.txt` and `summary.txt` for simplifying answers

## How to Run

### Streamlit UI

```bash
streamlit run main.py
```

* Enter a single question or multiple questions in the sidebar.
* The results table shows **Baseline vs Hybrid** retrieved chunks with their titles, page numbers, and scores.
* Grounded answers with citations appear above the table.

### FastAPI Backend

```bash
uvicorn api:app --reload
```

* API endpoints:

  * `/ask` – for single question queries
  * `/ask_batch` – for multiple questions in one request

### Results Table

The table compares **Baseline** (simple FAISS retrieval) vs **Hybrid** (reranked by keyword relevance) for each top-K chunk. Columns include rank, title, page number, and score. This helps visualize how reranking improves retrieval quality.

## What I Learned

Building this project deepened my understanding of hybrid retrieval systems and how semantic embeddings can be combined with keyword-based scoring for more precise document retrieval. Integrating both a UI and an API taught me practical workflow management, caching strategies, and performance tuning for embedding-heavy applications. Additionally, handling abstention logic and text deduplication highlighted the importance of filtering and cleaning results before presenting them to users.

## Gratitude

I am sincerely grateful for the opportunity to work on this project, which allowed me to apply modern NLP techniques to a real-world problem and improve my skills in building end-to-end retrieval systems.

## Example cURL Requests

### Easy Question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "What is machine guarding?"}'
```

### Tricky Question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Who is aligarh?"}'
```

* The tricky question demonstrates the system’s abstain logic when the query has no relevant match in the corpus.

```


```
