```markdown
# Mini-RAG vs Reranker: Industrial Safety Document Search

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Mini-RAG vs Reranker** is a CPU-friendly, fast, and accurate document search system designed for **industrial safety PDFs**.  
It combines **FAISS vector search**, **sentence-transformer embeddings**, and **hybrid reranker logic** to provide highly relevant results.

---

## 🌟 Features

- **PDF Ingestion**: Convert PDFs into chunked text stored in SQLite DB.  
- **Embeddings**: CPU-friendly embeddings using `paraphrase-MiniLM-L3-v2`.  
- **FAISS Search**: Fast vector similarity search across document chunks.  
- **Hybrid Reranker**: Re-ranks retrieved results for higher relevance.  
- **Interfaces**: 
  - CLI demo via Rich  
  - Optional FastAPI backend  
  - Streamlit Web UI (optional)  
- **Single & Batch Questions**: Run queries individually or in batch.  
- **Examples**: `examples/questions.json` included for testing.  

---

## 📂 Project Structure

```

Mini-RAG-vs-Reranker/
│── **pycache**/
│── data/
│   ├── industrial-safety-pdfs/
│   ├── chunks.db
│   ├── faiss\_index.index
│   ├── id\_map.json
│   └── sources.json
│
│── examples/
│   ├── questions.json
│   └── run\_results.csv
│
│── api.py
│── baseline\_search.py
│── embeddings.py
│── ingest.py
│── main.py
│── reranker\_hybrid.py
│── requirement.txt
│── README.md

````

> Large data files (FAISS index, PDFs) are **not tracked in Git**. Add them to `.gitignore`.

---

## ⚡ Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/Yatish54321/Mini-RAG-vs-Reranker.git
cd Mini-RAG-vs-Reranker
````

2. **Create a virtual environment (recommended)**

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

## 🚀 Usage

### CLI Demo

```bash
python main.py
```

* Choose **single** or **batch** mode.
* Enter questions to see **baseline** vs **hybrid** results side by side.

### FastAPI Backend (Optional)

```bash
uvicorn api:app --reload
```

* POST to `/ask` endpoint with JSON:

```json
{
  "question": "Who is responsible for safety protocols on site?"
}
```

* Returns top chunks from the documents.

---

## 📝 Example Questions

Located in `examples/questions.json`.
Batch-test multiple questions in CLI using `||` as separator.

---

## 📈 Logging

`examples/run_results.csv` records all queries and retrieved results for evaluation.

---

## 💡 Notes

* CPU-only, **no paid API** required.
* Embedding model: `sentence-transformers/paraphrase-MiniLM-L3-v2`.
* FAISS index ensures **fast vector search** even on CPU.

---

## 🛠️ Dependencies

* Python 3.13
* `sentence-transformers`
* `faiss-cpu`
* `numpy`
* `rich`
* `fastapi`
* `uvicorn`
* `pandas`

*(All listed in `requirement.txt`)*

---

## 📌 License

MIT License. See `LICENSE` for details.

---

## 👏 Credits

* Developed for **industrial safety document retrieval assessment**.
* Implements **RAG workflow** with hybrid reranker using FAISS and SentenceTransformers.

```

---
