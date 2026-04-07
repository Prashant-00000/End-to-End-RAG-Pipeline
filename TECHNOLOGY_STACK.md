# 🛠️ Complete Technology Stack - RAG System

## 📚 Python Core Libraries

### Data Processing & Text
| Technology | Version | Purpose | Used In |
|------------|---------|---------|---------|
| **pypdf** | Latest | PDF parsing and text extraction | `app/ingestion.py` |
| **numpy** | Latest | Numerical arrays and matrix operations | `app/embedding.py`, `app/vector_store.py` |
| **re** (Built-in) | Python 3.x | Regular expressions for text parsing | `app/chunking.py`, `app/bm25_store.py` |
| **json** (Built-in) | Python 3.x | JSON serialization for indexes | `app/vector_store.py`, `app/bm25_store.py` |
| **pathlib** (Built-in) | Python 3.x | Path handling cross-platform | `main.py`, `ui.py` |
| **dataclasses** (Built-in) | Python 3.7+ | Type-safe data containers | Multiple modules |
| **unicodedata** (Built-in) | Python 3.x | Unicode text normalization | `app/ingestion.py` |

---

## 🤖 Machine Learning & NLP

### Embedding Models
| Technology | Model | Dimension | Purpose |
|------------|-------|-----------|---------|
| **Sentence Transformers** | BAAI/bge-small-en | 1024-dim | Dense vector embeddings for semantic search |
| **BGE-Large-EN** | Optional upgrade | 1024-dim | Alternative higher-quality embedding |

**Features:**
- Automatic GPU/CPU detection (CUDA support)
- L2-normalization for cosine similarity
- Query prefix: "Represent this sentence for searching relevant passages: "
- Batch processing with configurable batch size
- ~1000 documents/minute throughput

### Reranking Models
| Technology | Model | Speed | Quality | Purpose |
|------------|-------|-------|---------|---------|
| **Cross-Encoder (MiniLM-L-6)** | ms-marco-MiniLM-L-6-v2 | ⚡ Fast | ⭐⭐ | Current: fast reranking |
| **Cross-Encoder (MiniLM-L-12)** | ms-marco-MiniLM-L-12-v2 | ⚡⚡ Medium | ⭐⭐⭐ | Balanced speed/quality |
| **Cross-Encoder (BGE-Large)** | bge-reranker-large | ⚡⚡⚡ Slow | ⭐⭐⭐⭐ | Best quality, 3× slower |

**Framework:** PyTorch-based cross-encoders via Sentence Transformers

---

## 🔍 Search & Retrieval

### Vector Search Engine
| Technology | Purpose | Details |
|------------|---------|---------|
| **FAISS** | Vector similarity search | Facebook's library for efficient similarity search |
| **Index Type** | IndexFlatIP | Inner product for cosine similarity on L2-normalized vectors |
| **Performance** | Sub-10ms latency | Fast exact search for <100k vectors |
| **Persistence** | Binary format (.faiss) | Efficient serialization and disk storage |

**Key Features:**
- L2-normalized vector support
- Optional approximate search (IVF, HNSW for larger datasets)
- GPU acceleration (FAISS-GPU available)

### Keyword Search Engine
| Technology | Algorithm | Purpose |
|------------|-----------|---------|
| **rank-bm25** | BM25Okapi | Probabilistic relevance scoring for keyword search |

**Features:**
- Stopword filtering (24 common English stopwords)
- Punctuation handling (preserves hyphens for compound terms)
- IDF-based term weighting
- ~5ms latency for typical queries
- JSON-serializable for persistence

### Hybrid Retrieval
| Component | Method | Purpose |
|-----------|--------|---------|
| **Reciprocal Rank Fusion (RRF)** | Custom implementation | Combines dense + sparse results |
| **Dense Retrieval** | Vector embeddings (BGE) | Semantic understanding |
| **Sparse Retrieval** | BM25 keywords | Exact term matching |

---

## 🧠 Large Language Models (LLM)

### API Provider
| Technology | Endpoint | Model | Specs |
|------------|----------|-------|-------|
| **Groq API** | Production API | llama-3.3-70b-versatile | 70 billion parameters |

**Capabilities:**
- Ultra-fast inference (1-2s per response)
- Streaming response support
- Error handling (connection, auth, rate limit)
- Timeout: 120 seconds (configurable)
- Max retries: 5 (configurable)

**Functions Using LLM:**
- `detect_intent()` - Classify "summary" vs "search" queries
- `rewrite_query()` - Improve query for better retrieval
- `generate_stream()` - Generate answers with context

---

## 💾 Database & Storage

### Backend Database
| Technology | Type | Purpose |
|------------|------|---------|
| **Supabase** | PostgreSQL | Session and conversation persistence |

**Tables:**
- `sessions` - Stores chat history, search history, session metadata

**Operations:**
- CRUD for session management
- Session ordering and filtering
- Batch operations for bulk delete

### Local Storage
| Format | Purpose | Contains |
|--------|---------|----------|
| **FAISS Binary** (.faiss) | Vector index persistence | Dense vectors, 1024-dim |
| **JSON** (.json) | Metadata storage | Vector text, source, page numbers |
| **BM25 JSON** (.bm25.json) | Keyword index persistence | BM25 model and tokenized documents |

---

## 🎨 Web Framework & UI

### Frontend Framework
| Technology | Purpose | Features |
|-----------|---------|----------|
| **Streamlit** | Web UI | Rapid interactive dashboard |

**UI Components:**
- Chat interface with message history
- File upload for PDFs
- Session management (dropdown selector)
- Real-time streaming responses
- Status indicators and spinners
- Sidebar for settings

**Architecture:**
- Single-page app (SPA)
- Server-side caching with `@st.cache_resource`
- Session state management (`st.session_state`)
- WebSocket support for streaming

### Optional API Framework
| Technology | Purpose | Status |
|-----------|---------|--------|
| **FastAPI** | REST API framework | Optional (not actively used) |
| **Uvicorn** | ASGI server | Optional (not actively used) |

---

## 🔐 Environment & Configuration

### Configuration Management
| Technology | Purpose | Usage |
|-----------|---------|-------|
| **python-dotenv** | Environment variable loading | Load `.env` file at startup |

**Environment Variables:**
```
GROQ_API_KEY          # Groq API authentication
GROQ_TIMEOUT_SECS     # LLM timeout setting
GROQ_MAX_RETRIES      # LLM retry attempts
SUPABASE_URL          # Database endpoint
SUPABASE_KEY          # Database authentication
CLOUD_STORAGE_BUCKET  # Optional cloud sync
```

---

## 📦 Package Management

| Technology | Purpose |
|-----------|---------|
| **pip** | Python package manager |
| **requirements.txt** | Dependency specification (frozen versions) |
| **virtualenv** (implicit) | Isolated Python environment (recommended) |

---

## 🔧 Development & Testing

### Testing Frameworks
| Technology | File | Purpose |
|-----------|------|---------|
| **pytest** (optional) | `app/test_api.py` | API endpoint testing |

### Debugging & Logging
| Technology | Purpose |
|-----------|---------|
| **print()** (built-in) | Console logging for debugging |
| **Exception handling** | Try-catch for resilience |

---

## 🌐 Cloud & Deployment

### Cloud Platforms
| Service | Purpose | Details |
|---------|---------|---------|
| **Supabase Cloud** | PostgreSQL hosting | Session persistence |
| **Groq Cloud API** | LLM inference | Fast inference as a service |
| **Optional Cloud Storage** | Index backup | Download pre-built indexes |

---

## 🔌 API & Integration

### External APIs
| API | Provider | Purpose | Method |
|-----|----------|---------|--------|
| **Groq Chat Completions** | Groq | Text generation | REST API + streaming |
| **Supabase PostgreSQL** | Supabase | Data persistence | REST API / Python SDK |

### Protocol & Formats
| Protocol | Purpose |
|----------|---------|
| **HTTP/REST** | API communication |
| **WebSocket** | Streaming responses |
| **JSON** | Data serialization |

---

## 📊 Data Structures & Algorithms

### Custom Data Classes
```python
PageDoc              # PDF page with text + metadata
SearchResult         # Vector store search result
BM25Result          # BM25 search result
RankedResult        # Reranked document result
EmbeddedChunk       # Chunk with embedding + metadata
PipelineConfig      # Pipeline hyperparameters
```

### Algorithms
| Algorithm | Implementation | Purpose |
|-----------|----------------|---------|
| **Cosine Similarity** | FAISS IndexFlatIP | Vector similarity scoring |
| **BM25 Okapi** | rank-bm25 library | Term-based relevance |
| **Reciprocal Rank Fusion** | Custom Python function | Merge dual retrieval results |
| **Cross-Encoder Reranking** | Sentence Transformers | Neural relevance scoring |
| **Semantic Chunking** | Custom with embeddings | Context-aware text splitting |

---

## 🎯 Performance & Optimization

### Resource Management
| Component | Resource | Optimization |
|-----------|----------|--------------|
| **GPU/CUDA** | Graphics memory | Auto-detect via PyTorch |
| **Model Loading** | Memory | Lazy-load and singleton pattern |
| **Batch Processing** | CPU throughput | Configurable batch size (64) |
| **Caching** | Query results | Streamlit @st.cache_resource |

### Latency Profile
```
Query Embedding:      ~50ms  (BGE model)
Vector Search:        ~5ms   (FAISS)
BM25 Search:          ~3ms   (In-memory)
Reranking (20 docs):  ~50ms  (Cross-encoder)
LLM Generation:       1-2s   (Groq API)
────────────────────────────
Total Query → Answer: ~2-3s
```

---

## 📋 Dependency Tree

```
rag-system/
├── pypdf                    # Text extraction
├── sentence-transformers    # BGE embeddings + Cross-encoders
│   └── torch (PyTorch)      # Deep learning framework
│   └── transformers         # Hugging Face models
├── faiss-cpu                # Vector search
├── rank-bm25                # Keyword search
├── numpy                    # Numerical operations
├── streamlit                # Web UI
│   └── streamlit-server     # ASGI server
├── groq                     # LLM API client
├── python-dotenv            # Environment config
├── supabase                 # PostgreSQL client
├── fastapi (optional)       # REST framework
├── uvicorn (optional)       # ASGI server
└── (system dependencies)
    └── CUDA Toolkit (optional, for GPU)
```

---

## 🔐 Security & Best Practices

### Security Libraries
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Authentication** | API Keys (.env) | Secure credential management |
| **Database** | Supabase Auth | Row-level security (potential) |
| **LLM Access** | Groq API key | Rate limiting, quota management |

### Code Quality
| Tool | Status |
|------|--------|
| **Type hints** | Used in dataclasses and functions |
| **Error handling** | Try-catch blocks in critical sections |
| **Logging** | Print statements with emoji prefixes |
| **Validation** | Path validation, minimum text length checks |

---

## 📌 Tech Stack Summary Table

| Category | Tech | Purpose |
|----------|------|---------|
| **Language** | Python 3.8+ | Core language |
| **PDF Processing** | pypdf | Extract text from PDFs |
| **Embeddings** | Sentence Transformers (BGE) | Dense vectors for semantics |
| **Vector Search** | FAISS | Fast similarity search |
| **Keyword Search** | rank-bm25 | Fast keyword retrieval |
| **Reranking** | Cross-Encoder | Neural relevance scoring |
| **LLM** | Groq (llama-3.3-70b) | Text generation |
| **UI** | Streamlit | Web interface |
| **Database** | Supabase (PostgreSQL) | Session storage |
| **Config** | python-dotenv | Environment management |
| **Numeric Ops** | NumPy | Array operations |

---

## 🚀 Optional/Future Tech

| Technology | Purpose | Status |
|-----------|---------|--------|
| **FastAPI** | REST API endpoints | In requirements.txt, not used |
| **Uvicorn** | ASGI server | In requirements.txt, not used |
| **PyTorch GPU** | GPU acceleration | Optional, auto-detected |
| **FAISS-GPU** | GPU vector search | Optional upgrade |
| **Docker** | Containerization | Not included |
| **Kubernetes** | Orchestration | Not included |

---

## 🔗 Technology Interactions

```
User Query (Streamlit)
        ↓
    Groq LLM (Intent Detection)
        ↓
    Query Preprocessing
        ↓
    ├─→ FAISS (Vector Search)←─ BGE Embeddings
    │         ↓
    │    Vector Results
    │
    └─→ BM25 (Keyword Search)
            ↓
        Keyword Results
        ↓
    Reciprocal Rank Fusion
        ↓
    Cross-Encoder (Reranking)
        ↓
    Top K Results
        ↓
    Groq LLM (Generation)
        ↓
    Supabase (Session Save)
        ↓
    Streamlit (Display)
```

---

**Total Unique Technologies: 25+**  
**Lines of Code: ~2000+**  
**Active Production Dependencies: 12**  
**Optional Dependencies: 2**
