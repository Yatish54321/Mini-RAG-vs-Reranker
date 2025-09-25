````markdown
# Mini-RAG vs Reranker Sprint 🏃‍♂️

**Mini-RAG vs Reranker** is a lightweight, CPU-friendly document retrieval system built for **industrial safety PDFs**.  
It combines **FAISS vector search**, **SentenceTransformer embeddings**, and a **hybrid reranker** to provide precise, grounded answers.

---

## 🌟 Features

- **PDF Ingestion**: Converts PDFs into chunked text stored in SQLite DB.  
- **Embeddings**: Uses CPU-friendly embeddings with `paraphrase-MiniLM-L3-v2`.  
- **FAISS Vector Search**: Efficient similarity search across document chunks.  
- **Hybrid Reranker**: Re-ranks baseline results using keyword scoring for higher relevance.  
- **Interfaces**: 
  - CLI demo via Python
  - Streamlit Web UI (optional)
  - FastAPI backend (optional)
- **Single & Batch Questions**: Query individually or in batch.
- **Logging**: All query results recorded in `examples/run_results.csv`.

---

## ⚡ Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/Yatish54321/Mini-RAG-vs-Reranker.git
cd Mini-RAG-vs-Reranker
````

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirement.txt
```

4. **Prepare data**

* Place PDFs in `data/industrial-safety-pdfs/`.
* Run ingestion and embeddings:

```bash
python ingest.py       # Convert PDFs → DB
python embeddings.py   # Generate embeddings & FAISS index
```

---

## 🚀 How to Run

### CLI / Streamlit

```bash
python main.py
```

* Choose **single** or **batch** mode.
* Enter questions to see **baseline** vs **hybrid** results side by side.

### FastAPI Backend

```bash
uvicorn api:app --reload
```

* POST to `/ask` endpoint with JSON:

```json
{
  "question": "Who is responsible for safety protocols on site?"
}
```

* Returns top matching chunks from your documents.

---

## 📑 Results Table

The results display includes:

| Rank | Baseline       | Hybrid         |
| ---- | -------------- | -------------- |
| 1    | Title (p.Page) | Title (p.Page) |
| 2    | Title (p.Page) | Title (p.Page) |
| …    | …              | …              |

* **Baseline**: Direct FAISS similarity search.
* **Hybrid**: Re-ranked results using both similarity + keyword score.

---

## 💡 What I Learned

Working on this project helped me understand **end-to-end retrieval pipelines**, from PDF ingestion to embedding-based vector search.
I learned how to combine **semantic search with heuristic reranking** to get more relevant and human-readable results, as well as best practices for **batch processing, logging, and API deployment**.

---

## 🛠️ Dependencies

* Python 3.13
* `sentence-transformers`
* `faiss-cpu`
* `numpy`
* `pandas`
* `rich`
* `fastapi`
* `uvicorn`
* `streamlit`

*(All listed in `requirement.txt`)*

---

## 📝 Example cURL Requests

### Easy Question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Who is responsible for safety on site?"}'
```

### Tricky Question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Explain the process of machine safeguarding and personnel protection in complex multi-level industrial setups."}'
```

* Returns top document chunks matching the query, including page numbers and titles.

---

## 👏 Gratitude

I sincerely thank the company for providing this opportunity.
This project allowed me to **apply NLP and vector retrieval techniques** to real-world industrial documents and gain hands-on experience in building **scalable, CPU-efficient search systems**.

```

```
