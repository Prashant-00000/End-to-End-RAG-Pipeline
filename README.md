# 🤖 RAG System

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with BM25 + vector search hybrid retrieval, Groq LLM inference, and Supabase as the backend — deployable on Streamlit Cloud.

---

## 🚀 Features

- **Hybrid Retrieval** — Combines BM25 sparse search and dense vector embeddings for best-of-both-worlds retrieval
- **Reranking** — Cross-encoder reranker to refine retrieved chunks before generation
- **Groq LLM** — Ultra-fast inference via Groq API
- **Supabase Backend** — Stores embeddings, documents, and metadata in the cloud
- **PDF Ingestion** — Ingest and chunk PDF documents into the knowledge base
- **Streamlit UI** — Clean chat interface for querying the RAG pipeline

---

## 🗂️ Project Structure

```
RAG-SYSTEM/
├── app/
│   ├── __init__.py
│   ├── bm25_store.py        # BM25 index management
│   ├── chunking.py          # Document chunking strategies
│   ├── cloud_storage.py     # Cloud storage integration
│   ├── db.py                # Database helpers
│   ├── embedding.py         # Embedding model wrapper
│   ├── groq_client.py       # Groq LLM client
│   ├── ingestion.py         # PDF ingestion pipeline
│   ├── pipeline.py          # End-to-end RAG pipeline
│   ├── reranker.py          # Cross-encoder reranker
│   ├── supabase_client.py   # Supabase client setup
│   ├── test_api.py          # API tests
│   └── vector_store.py      # Vector store operations
├── data/                    # Local PDFs for ingestion
├── indexes/                 # Local BM25 index cache
├── main.py                  # App entry point
├── requirements.txt
├── .env                     # Local secrets (never commit)
└── .gitignore
```

---

## ⚙️ Setup (Local)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_or_service_key
```

### 5. Ingest documents

Place your PDFs in the `data/` folder, then run:

```bash
python -c "from app.ingestion import ingest; ingest()"
```

### 6. Run the app

```bash
streamlit run main.py
# or
python main.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push your code to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file to `main.py` (or `streamlit_app.py` if renamed)
4. Add your secrets under **Settings → Secrets**:

```toml
GROQ_API_KEY = "your-key"
SUPABASE_URL = "your-url"
SUPABASE_KEY = "your-key"
```

5. Click **Deploy** 🚀

> **Note:** Streamlit Cloud is stateless. PDFs in `data/` and local BM25 indexes in `indexes/` won't persist. Use Supabase Storage for document uploads and rebuild the BM25 index on startup.

---

## 🧠 How It Works

```
User Query
    │
    ▼
BM25 Sparse Search  +  Vector Similarity Search
    │                          │
    └──────────┬───────────────┘
               ▼
          Reranker (Cross-Encoder)
               │
               ▼
         Top-K Chunks
               │
               ▼
      Groq LLM (Generation)
               │
               ▼
          Final Answer
```

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| LLM | [Groq](https://groq.com) |
| Embeddings | Sentence Transformers |
| Vector Store | Supabase (pgvector) |
| Sparse Search | BM25 (rank-bm25) |
| Reranker | Cross-Encoder |
| Frontend | Streamlit |
| Database | Supabase |

---

## 📄 License

MIT License — feel free to use and adapt.


## 🌐 Live Demo

👉 [Try it here](https://end-to-end-rag-pipelinegit-m2tjnwfmpv4xqlvpnjwt9t.streamlit.app/)
