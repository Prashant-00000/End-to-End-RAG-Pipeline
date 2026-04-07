# 🤖 End-to-End RAG Pipeline

A production-ready Retrieval Augmented Generation system with hybrid search, neural reranking, and Groq LLM integration.

## 🎯 Features

✅ **Hybrid Search** - Dense (semantic) + Sparse (keyword) retrieval  
✅ **Smart Chunking** - Semantic-aware text splitting  
✅ **Neural Reranking** - Cross-encoder for relevance scoring  
✅ **Streaming LLM** - Groq API with ultra-fast inference  
✅ **Session Management** - Supabase for persistent conversations  
✅ **Cloud-Ready** - Deploy to Streamlit Cloud in minutes  

---

## 🚀 Quick Start

### Local Development

1. **Clone & Install**
```bash
git clone https://github.com/YOUR_USERNAME/rag-system.git
cd rag-system
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
# Create .env file
GROQ_API_KEY=your_api_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

3. **Run UI**
```bash
streamlit run ui.py
```

Opens at: `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud

### **Complete Deployment Guide**
📖 See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions

### **5-Minute Quick Deploy**
📖 See [QUICK_DEPLOY.md](QUICK_DEPLOY.md) for rapid setup

**Your live app:** `https://YOUR_USERNAME-rag-system.streamlit.app/`

---

## 📚 Project Documentation

| Document | Purpose |
|----------|---------|
| [PROJECT_INFO.md](PROJECT_INFO.md) | Complete architecture & components |
| [TECHNOLOGY_STACK.md](TECHNOLOGY_STACK.md) | All 25+ technologies explained |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Step-by-step cloud deployment |
| [QUICK_DEPLOY.md](QUICK_DEPLOY.md) | 5-minute quick start |

---

## 🏗️ Architecture Overview

```
PDF Input
    ↓
[Ingestion] → Parse + Extract Text
    ↓
[Chunking] → Semantic + Fixed strategies
    ↓
    ├─→ [BGE Embedding] → FAISS Index
    └─→ [BM25 Tokenize] → BM25 Index
    ↓
[Hybrid Search] → Dense + Sparse retrieval
    ↓
[Reranking] → Neural relevance scoring
    ↓
[Groq LLM] → Generate answer
    ↓
[Supabase] → Save session
    ↓
User Response (Streamlit UI)
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI Framework** | Streamlit |
| **Embeddings** | Sentence Transformers (BGE) |
| **Vector Search** | FAISS |
| **Keyword Search** | BM25 |
| **Reranking** | Cross-Encoder |
| **LLM** | Groq (Llama 3.3-70b) |
| **Database** | Supabase (PostgreSQL) |
| **PDF Processing** | PyPDF |

See [TECHNOLOGY_STACK.md](TECHNOLOGY_STACK.md) for complete breakdown.

---

## 📋 Requirements

### API Keys (Free Tier)
- 🔑 **Groq API** - Get at https://console.groq.com
- 🔑 **Supabase** - Sign up at https://supabase.com
- 🔑 **GitHub** - For Streamlit Cloud deployment

### Python Dependencies
```
Python 3.8+
pypdf, numpy, torch/pytorch
sentence-transformers, faiss-cpu
rank-bm25
streamlit, groq, supabase, python-dotenv
```

Full list in [requirements.txt](requirements.txt)

---

## 🎨 Web Interface

**Features:**
- 💬 Interactive chat with streaming responses
- 📄 PDF upload and processing
- 💾 Save/load conversation sessions
- 🔍 Real-time search results display
- ⚡ Session history persistence

---

## ⚙️ Configuration

### Environment Variables
```bash
# .env file
GROQ_API_KEY=gsk_your_api_key
GROQ_TIMEOUT_SECS=120
GROQ_MAX_RETRIES=5

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

CLOUD_STORAGE_BUCKET=optional-bucket-name
```

### Pipeline Parameters
Edit `app/pipeline.py`:
```python
PipelineConfig:
  top_k_retrieve: int = 20      # Initial retrieval count
  top_k_final: int = 3          # Final results after reranking
  score_threshold: float = -5.0 # Minimum relevance score
```

---

## 📊 Performance

| Operation | Time |
|-----------|------|
| Model loading | 2-5s (first run) |
| Query embedding | 2s |
| Vector search | 0.1s |
| Reranking (20 docs) | 0.5s |
| LLM response | 1-3s |
| **Total per query** | ~10-20s |

---

## 🔒 Security

✅ API keys stored in `.env` (local) or Streamlit Secrets (cloud)  
✅ `.gitignore` excludes sensitive files  
✅ HTTPS enabled on Streamlit Cloud  
✅ Supabase provides row-level security (RLS)  

**Never commit:**
- `.env` files
- API keys
- `secrets.toml`
- PDFs with sensitive data

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "GROQ_API_KEY not found" | Add to `.env` or Secrets |
| "Supabase connection failed" | Verify URL and key |
| "Model download timeout" | Check internet, retry |
| "Out of memory" | Reduce batch size in `embedding.py` |
| "Slow startup" | Models cache after first run |

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for more troubleshooting.

---

## 📚 Learn More

### Technical Deep Dives
- [Project Architecture](PROJECT_INFO.md) - Components & workflow
- [Technology Stack](TECHNOLOGY_STACK.md) - 25+ technologies explained
- [Deployment Steps](DEPLOYMENT_GUIDE.md) - Cloud deployment details

### External Resources
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-large-en)
- [Cross-Encoders](https://www.sbert.net/docs/pretrained_cross-encoders.html)
- [Groq API](https://console.groq.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [Supabase Docs](https://supabase.com/docs)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit (`git commit -m 'Add amazing feature'`)
5. Push (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📄 License

This project is open source. See LICENSE file for details.

---

## 🎉 Get Started Now!

### Local Development
```bash
streamlit run ui.py
```

### Cloud Deployment
Follow [QUICK_DEPLOY.md](QUICK_DEPLOY.md) for 5-minute setup on Streamlit Cloud.

---

**Status:** ✅ Production-Ready  
**Last Updated:** April 2026  
**Python Version:** 3.8+  
**Maintenance:** Active
