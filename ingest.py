import os
import json
import sqlite3
from datetime import datetime
from multiprocessing import Pool, cpu_count
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# -------------------------------
# NLTK setup
# -------------------------------
for resource in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}") if "punkt" in resource else nltk.data.find(f"corpora/{resource}")
    except LookupError:
        print(f"[INFO] Downloading NLTK resource: {resource}")
        nltk.download(resource)

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
PDF_FOLDER = os.path.join(DATA_DIR, "industrial-safety-pdfs")
SOURCES_FILE = os.path.join(DATA_DIR, "sources.json")
DB_FILE = os.path.join(DATA_DIR, "chunks.db")

PLAIN_TEXT_FILE = os.path.join(DATA_DIR, "plain_text.txt")
SUMMARY_FILE = os.path.join(DATA_DIR, "summary.txt")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
NUM_WORKERS = max(1, cpu_count() - 1)

# -------------------------------
# Helper Functions
# -------------------------------
def load_sources(sources_path):
    """Load sources.json (list of dicts)"""
    with open(sources_path, "r", encoding="utf-8") as f:
        sources = json.load(f)
    return sources

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        if chunk:
            chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

def create_database(db_path):
    """Create SQLite DB with chunks + FTS5 table"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        source_id TEXT,
        title TEXT,
        page_num INTEGER,
        chunk_text TEXT,
        chunk_len INTEGER,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        chunk_text,
        content='chunks',
        content_rowid='id'
    )
    """)

    conn.commit()
    return conn

def process_pdf(args):
    """Process single PDF and return list of chunks to insert"""
    pdf_file, source = args
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    title = source.get("title", "Unknown Title")
    source_id = pdf_file
    chunks_data = []

    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if not text:
                continue
            chunks = split_text_into_chunks(text)
            for chunk_text in chunks:
                chunk_len = len(chunk_text.split())
                created_at = datetime.now().isoformat()

            
                if "electrical" in chunk_text.lower():
                    print(f"[DEBUG] '{title}' p.{page_num} → {chunk_text[:200]}...")

                chunks_data.append(
                    (source_id, title, page_num, chunk_text, chunk_len, created_at)
                )
        doc.close()
    except Exception as e:
        print(f"[ERROR] Failed to process {pdf_path}: {e}")

    return chunks_data

def summarize_text(text, num_sentences=10):
    """Simple frequency-based extractive summarization"""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())

    freq = {}
    for w in words:
        if w.isalnum() and w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sent in sentences:
        sentence_scores[sent] = sum(freq.get(word.lower(),0) for word in word_tokenize(sent))

    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return ' '.join(top_sentences)

def ingest_pdfs(pdf_folder, sources, conn):
    """Process PDFs in parallel, insert chunks into DB"""
    cur = conn.cursor()
    total_chunks = 0
    all_text = []

    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")])

    if len(pdf_files) != len(sources):
        print(f"[WARNING] Number of PDFs ({len(pdf_files)}) != number of sources ({len(sources)})")

    # Prepare arguments for parallel processing
    args_list = []
    for idx, source in enumerate(sources):
        try:
            pdf_file = pdf_files[idx]
        except IndexError:
            print(f"[WARNING] No PDF file found for source {source.get('title')}, skipping.")
            continue
        args_list.append((pdf_file, source))

    # Process PDFs in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_pdf, args_list)

    # Insert chunks into DB and accumulate text
    for chunks_data in results:
        for data in chunks_data:
            cur.execute("""
            INSERT INTO chunks (source_id, title, page_num, chunk_text, chunk_len, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, data)
            chunk_id = cur.lastrowid
            cur.execute("""
            INSERT INTO chunks_fts (rowid, chunk_text)
            VALUES (?, ?)
            """, (chunk_id, data[3]))
            total_chunks += 1
            all_text.append(data[3])

    conn.commit()
    print(f"[INFO] Ingestion completed. Total chunks inserted: {total_chunks}")

    # Save all text to plain_text.txt
    with open(PLAIN_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    print(f"[INFO] Plain text saved at {PLAIN_TEXT_FILE}")

    # Generate extractive summary
    summary = summarize_text(" ".join(all_text), num_sentences=10)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[INFO] Summary saved at {SUMMARY_FILE}")

    # Insert summary as a special chunk
    created_at = datetime.now().isoformat()
    cur.execute("""
    INSERT INTO chunks (source_id, title, page_num, chunk_text, chunk_len, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, ("summary", "Extractive Summary", 0, summary, len(summary.split()), created_at))
    chunk_id = cur.lastrowid
    cur.execute("""
    INSERT INTO chunks_fts (rowid, chunk_text)
    VALUES (?, ?)
    """, (chunk_id, summary))
    conn.commit()
    print(f"[INFO] Summary chunk inserted into DB with id {chunk_id}")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("[INFO] Starting PDF ingestion...")

    if not os.path.exists(SOURCES_FILE):
        print(f"[ERROR] sources.json not found in {DATA_DIR}. Exiting.")
        exit(1)

    sources = load_sources(SOURCES_FILE)
    conn = create_database(DB_FILE)

    ingest_pdfs(PDF_FOLDER, sources, conn)

    conn.close()
    print("[INFO] Done. SQLite DB created at:", DB_FILE)
