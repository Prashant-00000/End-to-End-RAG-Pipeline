# 🤖 RAG System - Complete Project Information

## 📋 Project Overview
This is an **End-to-End RAG (Retrieval Augmented Generation) Pipeline** - a sophisticated document retrieval and question-answering system that:
- Ingests PDF documents
- Creates semantic and keyword-based indexes
- Uses hybrid retrieval (dense + sparse search)
- Reranks results for quality
- Generates answers with AI using Groq LLM API
- Persists sessions to Supabase

---

## 🏗️ Architecture & Core Components

### 1. **Data Ingestion** (`app/ingestion.py`)
- **Function**: `load_pdfs()` 
- Extracts text from PDF files with:
  - Per-page metadata (source, page number, total pages)
  - Optional OCR fallback for scanned documents
  - Minimum character filtering (skips blank pages)
- **Input Files**: 
  - `data/A_Brief_Introduction_To_AI.pdf`
  - `data/2024-wttc-introduction-to-ai.pdf`
  - `data/Gen ai.pdf`
  - `data/Employee-Handbook.pdf`

### 2. **Text Chunking** (`app/chunking.py`)
Three strategies:
- **Semantic Chunking**: Uses embeddings to group semantically similar sentences
  - Configurable similarity threshold (default: 0.45)
  - Better context preservation
- **Fixed Chunking**: Simple sliding window (chunk_size=300, overlap=50)
  - Fast, reliable fallback
- **Hybrid Chunking**: Combines both approaches
  - Semantic grouping within fixed-size boundaries

### 3. **Embedding Model** (`app/embedding.py`)
- **Model**: `BAAI/bge-small-en` (1024-dimensional embeddings)
- **Framework**: Sentence Transformers
- **Features**:
  - Automatic GPU detection (CUDA if available, else CPU)
  - L2-normalization for cosine similarity
  - Query-specific prefix: "Represent this sentence for searching relevant passages: "
  - Batch processing with configurable batch size
  - Progress callbacks for monitoring

### 4. **Vector Store** (`app/vector_store.py`)
- **Backend**: FAISS (Facebook AI Similarity Search)
- **Index Type**: `IndexFlatIP` (inner product = cosine similarity on normalized vectors)
- **Features**:
  - Fast semantic search via embeddings
  - Metadata tracking (source, page_number, etc.)
  - JSON persistence (`indexes/vector_store.json`)
  - FAISS binary format (`indexes/vector_store.faiss`)
  - Optional metadata filtering during search

### 5. **BM25 Keyword Search** (`app/bm25_store.py`)
- **Algorithm**: BM25Okapi (probabilistic relevance framework)
- **Features**:
  - Stopword filtering (tuned for domain)
  - Punctuation handling (preserves hyphens for "state-of-the-art" etc.)
  - Keyword-based retrieval (sparse search)
  - Complements dense vector search
  - Persisted to `indexes/bm25_store.bm25.json`

### 6. **Retrieval Pipeline** (`app/pipeline.py`)
**Hybrid Retrieval Strategy**:
1. **Dense Retrieval**: Vector similarity search (top 20 results)
2. **Sparse Retrieval**: BM25 keyword search (top 20 results)
3. **Result Fusion**: Reciprocal Rank Fusion combines both
4. **Reranking**: Cross-encoder reranks fused results
5. **Final Output**: Top 3 most relevant chunks

**Pipeline Configurations**:
```python
PipelineConfig:
  - top_k_retrieve: 20 (initial retrieval)
  - top_k_final: 3 (final results)
  - score_threshold: -5.0 (minimum relevance score)

SummaryConfig (for broader summaries):
  - top_k_retrieve: 60
  - top_k_final: 10
  - score_threshold: -100.0 (accepts lower scores)
```

### 7. **Document Reranker** (`app/reranker.py`)
- **Model**: Cross-Encoder neural reranker
- **Options**:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast) ← current
  - `cross-encoder/ms-marco-MiniLM-L-12-v2` (balanced)
  - `BAAI/bge-reranker-large` (best quality, 3× slower)
- **Purpose**: Neural re-scoring of candidate chunks
- **Output**: `RankedResult` objects with relevance scores

### 8. **LLM Integration** (`app/groq_client.py`)
- **Provider**: Groq API (fast inference)
- **Model**: `llama-3.3-70b-versatile`
- **Features**:
  - Query rewriting for better retrieval
  - Intent detection (summary vs. specific search)
  - Streaming response generation
  - Error handling (connection, auth, rate limit)
  - Configuration via `.env` file:
    ```
    GROQ_API_KEY=your_key_here
    GROQ_TIMEOUT_SECS=120
    GROQ_MAX_RETRIES=5
    ```

### 9. **Database / Session Management** (`app/db.py`)
- **Backend**: Supabase (PostgreSQL)
- **Tables**: `sessions` table with:
  - `name`: Session identifier
  - `history`: Retrieval history
  - `chat_history`: Conversation messages
- **Operations**:
  - `get_session()` - Retrieve or create
  - `update_session()` - Persist chat state
  - `load_all_sessions()` - List all saved chats
  - `delete_session()` - Remove session
  - `clear_all_sessions()` - Wipe all sessions

### 10. **Cloud Storage** (`app/cloud_storage.py`)
- Syncs indexes to cloud storage (Supabase)
- Downloads pre-built indexes to avoid rebuilding
- Enables distributed deployment

---

## 🎨 User Interface & Entry Points

### **UI Frontend** (`ui.py`)
- **Framework**: Streamlit
- **Features**:
  - Interactive chat interface (🤖 RAG System)
  - Document upload
  - Session management (save/load/delete conversations)
  - Index status (built vs. cloud)
  - Real-time streaming responses
- **Workflow**:
  1. Check if FAISS indexes exist
  2. If not, try downloading from cloud
  3. If not, build indexes from scratch
  4. Display chat interface with session history

### **Main Entry Point** (`main.py`)
- Core functions:
  - `build_indexes()` - Creates vector + BM25 indexes from PDFs
  - `load_indexes()` - Loads pre-built indexes from disk
  - Shared state: `query_pipeline`, `UPLOAD_TASKS`, etc.

---

## 📦 Dependencies

### Core Libraries
```
pypdf                    # PDF extraction
sentence-transformers   # Embedding model (BGE)
faiss-cpu              # Vector similarity search
rank-bm25              # BM25 keyword search
numpy                  # Numerical operations
```

### API & Backend
```
fastapi                # REST API (optional)
uvicorn               # ASGI server (optional)
python-dotenv         # Environment variable management
groq                  # LLM API client
```

### UI
```
streamlit             # Web interface for RAG
```

---

## 📂 File Structure

```
rag-system/
├── main.py                      # Core entry point + index building
├── ui.py                        # Streamlit UI interface
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── app/                         # Application modules
│   ├── __init__.py
│   ├── ingestion.py            # PDF loading & parsing
│   ├── chunking.py             # Text chunking strategies
│   ├── embedding.py            # BGE embedding model
│   ├── vector_store.py         # FAISS vector store
│   ├── bm25_store.py           # BM25 keyword index
│   ├── pipeline.py             # Hybrid retrieval pipeline
│   ├── reranker.py             # Cross-encoder reranking
│   ├── groq_client.py          # Groq LLM integration
│   ├── db.py                   # Supabase session management
│   ├── supabase_client.py      # Supabase connection
│   ├── cloud_storage.py        # Cloud sync
│   ├── test_api.py             # API testing
│   └── (other utilities)
│
├── data/                        # Source documents
│   ├── A_Brief_Introduction_To_AI.pdf
│   ├── 2024-wttc-introduction-to-ai.pdf
│   ├── Gen ai.pdf
│   └── Employee-Handbook.pdf
│
├── indexes/                     # Pre-built indexes (persisted)
│   ├── vector_store.faiss       # FAISS binary index
│   ├── vector_store.json        # Vector metadata
│   ├── bm25_store.bm25.json     # BM25 index
│   └── bm25_store.json          # BM25 metadata
│
└── (scratch files & configs)
    ├── scratch_history.json
    ├── scratch_test.py
    ├── sessions.json.migrated
    └── sessions.json.migrated
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)
```bash
# LLM Configuration
GROQ_API_KEY=<your_api_key>
GROQ_TIMEOUT_SECS=120
GROQ_MAX_RETRIES=5

# Supabase Configuration (for sessions)
SUPABASE_URL=<your_supabase_url>
SUPABASE_KEY=<your_supabase_key>

# Cloud Storage (optional)
CLOUD_STORAGE_BUCKET=<your_bucket>
```

### Pipeline Parameters
```python
# app/pipeline.py
PipelineConfig:
  top_k_retrieve: int = 20      # Dense + sparse retrieval
  top_k_final: int = 3          # Final results after reranking
  score_threshold: float = -5.0 # Minimum relevance cutoff
```

---

## 🚀 Workflow: Query → Answer

```
User Query
    ↓
[1] Intent Detection (Groq)
    ├─→ "summary" → SummaryConfig (60 chunks, threshold=-100)
    └─→ "search" → PipelineConfig (20 chunks, threshold=-5)
    ↓
[2] Hybrid Retrieval
    ├─→ Dense: Vector search (BGE embeddings)
    ├─→ Sparse: BM25 keyword search
    └─→ Fusion: Reciprocal rank fusion
    ↓
[3] Reranking
    └─→ Cross-encoder scores all candidates
    ↓
[4] Filtering
    └─→ Top-K by score threshold
    ↓
[5] Context Building
    └─→ Format chunks + metadata for LLM
    ↓
[6] Generation (Groq llama-3.3-70b)
    └─→ Stream response to user
    ↓
[7] Session Save (Supabase)
    └─→ Persist query + answer + history
```

---

## 🎯 Key Features

✅ **Hybrid Search**
- Combines dense (semantic) + sparse (keyword) retrieval
- Better coverage for different query types

✅ **Smart Chunking**
- Semantic chunking preserves context better than fixed chunks
- Fallback to fixed chunking if semantic fails

✅ **Reranking**
- Cross-encoder neural reranking improves relevance
- Configurable models for speed/quality tradeoff

✅ **Session Management**
- Save/load conversation history via Supabase
- Query tracking for analytics

✅ **Cloud-Ready**
- Indexes downloadable from cloud storage
- No need to rebuild on fresh deployment

✅ **Fast LLM**
- Groq API provides low-latency inference
- Streaming responses for better UX

✅ **Production-Ready**
- Error handling (API failures, rate limits)
- Metadata tracking (source, page numbers)
- Configuration via environment variables

---

## 📊 Data Flow Diagram

```
PDF Files (data/)
    ↓
[Load PDFs]  ← ingestion.py
    ↓
Raw Text + Metadata
    ↓
[Semantic Chunking]  ← chunking.py
    ↓
Text Chunks
    ↓
    ├─→ [BGE Embedding]  ← embedding.py
    │        ↓
    │   Embeddings
    │        ↓
    │   [FAISS Index]  ← vector_store.py
    │        ↓
    │   indexes/vector_store.faiss
    │
    └─→ [BM25 Tokenize]  ← bm25_store.py
             ↓
        BM25 Index
             ↓
        indexes/bm25_store.bm25.json

═══════════════════════════════════════════════════════

User Query (ui.py)
    ↓
[Detect Intent]  ← groq_client.py
    ↓
[Retrieve] (pipeline.py)
    ├─→ Vector Search (FAISS)
    └─→ BM25 Search
    ↓
    [Reciprocal Rank Fusion]
    ↓
    [Rerank]  ← reranker.py
    ↓
    [Generate Answer]  ← groq_client.py
    ↓
    [Save Session]  ← db.py (Supabase)
    ↓
User Response (Streamlit UI)
```

---

## 🔧 Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env`:
```bash
GROQ_API_KEY=your_api_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

### 3. Add PDF Data
Place PDF files in `data/` folder

### 4. Build Indexes (first time)
```bash
python main.py  # Builds from scratch or downloads from cloud
```

### 5. Run UI
```bash
streamlit run ui.py
```

Opens at: `http://localhost:8501`

---

## 📈 Performance Characteristics

| Component | Model | Speed | Quality |
|-----------|-------|-------|---------|
| Embedding | BGE-small (1024D) | ~1000 docs/min | Good, fast |
| Vector Search | FAISS IndexFlatIP | <10ms for 1k docs | Exact |
| BM25 Search | BM25Okapi | <5ms | Good for keywords |
| Reranking | MiniLM-L-12 | ~50ms for 20 docs | Balanced |
| LLM | Groq llama-3.3-70b | ~1-2s per response | Excellent |

---

## 🐛 Debugging Tips

- **Check index exists**: `ls indexes/` should show `.faiss` and `.json` files
- **Verify embeddings**: Ensure BGE model downloaded (~400MB)
- **Test Groq API**: Check `GROQ_API_KEY` in `.env`
- **Supabase connection**: Verify `SUPABASE_URL` and `SUPABASE_KEY`
- **Logs**: Streamlit shows pipeline steps in console

---

## 📚 Additional Resources

- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- BM25: https://en.wikipedia.org/wiki/Okapi_BM25
- Cross-Encoders: https://www.sbert.net/docs/pretrained_cross-encoders.html
- Groq API: https://console.groq.com
- Supabase: https://supabase.com

---

**Last Updated**: April 2026  
**Project Type**: Full-Stack RAG Application  
**Status**: Production-Ready
