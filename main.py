# Suppress warnings early - must be first
import warnings
import os
import sys

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*No module named.*torchvision.*')
warnings.filterwarnings('ignore', message='.*No module named.*timm.*')
warnings.filterwarnings('ignore', message='.*Accessing `__path__`.*')
warnings.filterwarnings('ignore', message='.*Tried to instantiate class.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import app package (which also has warning suppression)
import app

from pathlib import Path
from app.ingestion import load_pdfs
from app.chunking import semantic_chunking
from app.embedding import embed_documents, embed_query
from app.vector_store import VectorStore
from app.bm25_store import BM25Store
from app.reranker import rerank, RankedResult
from app.pipeline import run_pipeline, PipelineConfig, SummaryConfig
import threading

# ── Paths ──────────────────────────────────────────────────────────────────────

INDEX_DIR = Path("indexes")
DATA_DIR = Path("data")

# Module-level exports (for UI)
UPLOAD_TASKS = {}

# ── Index building ─────────────────────────────────────────────────────────────

def build_indexes() -> tuple[VectorStore, BM25Store]:
    """Build vector store and BM25 index from PDFs."""
    print("🔨 Building indexes from scratch...")
    INDEX_DIR.mkdir(exist_ok=True)

    pdf_files = [str(p) for p in DATA_DIR.glob("*.pdf")]
    
    # If no PDFs found, return empty indexes (for Streamlit Cloud)
    if not pdf_files:
        print("⚠️ No PDFs found in data/ folder")
        print("📌 App will work with BM25 search only (add PDFs to data/ to enable full RAG)")
        
        # Return empty indexes
        vs = VectorStore(384)  # BGE small has 384 dimensions
        bm25 = BM25Store([])
        
        vs.save(INDEX_DIR / "vector_store.faiss")
        bm25.save(INDEX_DIR / "bm25_store.bm25.json")
        
        return vs, bm25

    page_docs = load_pdfs(pdf_files)
    if not page_docs:
        print("⚠️ No documents loaded from PDFs")
        vs = VectorStore(384)
        bm25 = BM25Store([])
        vs.save(INDEX_DIR / "vector_store.faiss")
        bm25.save(INDEX_DIR / "bm25_store.bm25.json")
        return vs, bm25
    
    text = " ".join([doc.text for doc in page_docs])

    text = text.replace("\n", " ")
    text = " ".join(text.split())
    print(f"📄 Loaded {len(text)} characters")

    chunks = semantic_chunking(text, threshold=0.45)
    print(f"📦 Initial chunks: {len(chunks)}")

    clean_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 50:
            continue
        if "ACKNOWLEDGEMENTS" in chunk.upper():
            continue
        if "ANNEX" in chunk.upper():
            continue
        if "ENDNOTES" in chunk.upper():
            continue
        if "........" in chunk:
            continue
        clean_chunks.append(chunk)

    print(f"✨ {len(clean_chunks)} clean chunks")

    if not clean_chunks:
        print("⚠️ No clean chunks found! Using fixed chunking instead...")
        from app.chunking import fixed_chunking
        chunks = fixed_chunking(text, chunk_size=500, overlap=50)
        clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 50]
        print(f"✨ {len(clean_chunks)} clean chunks (from fixed chunking)\n")

    if not clean_chunks:
        print("⚠️ Still no chunks after fallback!")
        vs = VectorStore(384)
        bm25 = BM25Store([])
        vs.save(INDEX_DIR / "vector_store.faiss")
        bm25.save(INDEX_DIR / "bm25_store.bm25.json")
        return vs, bm25

    print(f"Chunks to embed: {len(clean_chunks)}")
    
    embedded_chunks = embed_documents(clean_chunks)
    embeddings = [ec.embedding for ec in embedded_chunks]
    
    print(f"Embeddings created: {len(embeddings)}")
    
    if not embeddings or len(embeddings[0]) == 0:
        print("⚠️ Failed to create embeddings, using empty indexes")
        vs = VectorStore(384)
        bm25 = BM25Store([])
    else:
        vs = VectorStore(len(embeddings[0]))
        vs.add(embeddings, clean_chunks)
        bm25 = BM25Store(clean_chunks)

    vs.save(INDEX_DIR / "vector_store.faiss")
    bm25.save(INDEX_DIR / "bm25_store.bm25.json")

    # Sync to Supabase cloud storage
    try:
        from app.cloud_storage import upload_indexes
        upload_indexes(INDEX_DIR)
    except Exception as e:
        print(f"ℹ️ Cloud upload skipped: {e}")

    print("💾 Indexes saved to disk")
    return vs, bm25


def load_indexes() -> tuple[VectorStore, BM25Store]:
    """Load pre-built vector store and BM25 index from disk."""
    print("📂 Loading indexes from disk...")
    try:
        vs = VectorStore.load(INDEX_DIR / "vector_store.faiss")
        bm25 = BM25Store.load(INDEX_DIR / "bm25_store.bm25.json")
        print("✅ Indexes loaded")
        return vs, bm25
    except FileNotFoundError:
        print("ℹ️ Index files not found, rebuilding...")
        return build_indexes()
    except Exception as e:
        print(f"⚠️ Error loading indexes: {e}")
        print("ℹ️ Rebuilding indexes...")
        return build_indexes()


# ── Query pipeline ─────────────────────────────────────────────────────────────

def query_pipeline(
    query: str,
    vs: VectorStore,
    bm25: BM25Store,
    *,
    is_summary_mode: bool = False,
    metadata_filter: dict | None = None,
) -> list[RankedResult]:
    """Run the full RAG retrieval pipeline."""
    config = SummaryConfig() if is_summary_mode else PipelineConfig()
    return run_pipeline(query, vs, bm25, config=config, metadata_filter=metadata_filter)


# ── Document management ────────────────────────────────────────────────────────

def _add_document_worker(file_path: Path, vs: VectorStore, bm25: BM25Store) -> None:
    """Background worker: ingest a new PDF and update indexes."""
    name = file_path.name
    try:
        UPLOAD_TASKS[name] = {"status": "Loading PDF...", "progress": 0.1}

        page_docs = load_pdfs([str(file_path)])
        text = " ".join([doc.text for doc in page_docs])
        text = text.replace("\n", " ")
        text = " ".join(text.split())

        UPLOAD_TASKS[name] = {"status": "Chunking...", "progress": 0.3}
        chunks = semantic_chunking(text, threshold=0.45)
        clean_chunks = [
            c.strip() for c in chunks
            if len(c.strip()) >= 50
            and "........" not in c
        ]

        if not clean_chunks:
            from app.chunking import fixed_chunking
            chunks = fixed_chunking(text, chunk_size=500, overlap=50)
            clean_chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

        UPLOAD_TASKS[name] = {"status": "Embedding...", "progress": 0.6}
        embedded_chunks = embed_documents(clean_chunks)
        embeddings = [ec.embedding for ec in embedded_chunks]
        metadata = [{"source": name, "page_number": 1} for _ in clean_chunks]

        vs.add(embeddings, clean_chunks, metadata)
        bm25.add(clean_chunks, metadata)

        UPLOAD_TASKS[name] = {"status": "Saving...", "progress": 0.9}
        INDEX_DIR.mkdir(exist_ok=True)
        vs.save(INDEX_DIR / "vector_store.faiss")
        bm25.save(INDEX_DIR / "bm25_store.bm25.json")

        try:
            from app.cloud_storage import upload_indexes
            upload_indexes(INDEX_DIR)
        except Exception:
            pass

        UPLOAD_TASKS[name] = {"status": "Done", "progress": 1.0}

    except Exception as e:
        UPLOAD_TASKS[name] = {"status": f"Error: {e}", "progress": 0.0}


def async_add_document(file_path: Path, vs: VectorStore, bm25: BM25Store) -> None:
    """Kick off background ingestion of a new PDF (non-blocking)."""
    name = file_path.name
    UPLOAD_TASKS[name] = {"status": "Queued...", "progress": 0.0}
    thread = threading.Thread(
        target=_add_document_worker,
        args=(file_path, vs, bm25),
        daemon=True,
    )
    thread.start()


def remove_document_fast(file_path: Path, vs: VectorStore, bm25: BM25Store) -> None:
    """Remove all chunks belonging to a PDF from the indexes."""
    name = file_path.name
    try:
        # Filter out chunks belonging to this document
        keep_indices = [
            i for i, meta in enumerate(vs.metadata)
            if meta.get("source") != name
        ]

        kept_texts = [vs.texts[i] for i in keep_indices]
        kept_meta  = [vs.metadata[i] for i in keep_indices]

        # Rebuild vector store without the removed document
        if kept_texts:
            embedded_chunks = embed_documents(kept_texts)
            embeddings = [ec.embedding for ec in embedded_chunks]
            new_vs = VectorStore(vs.dim)
            new_vs.add(embeddings, kept_texts, kept_meta)
        else:
            new_vs = VectorStore(vs.dim)

        # Copy back into the passed vs object
        vs.index    = new_vs.index
        vs.texts    = new_vs.texts
        vs.metadata = new_vs.metadata

        # Rebuild BM25 without the removed document
        kept_bm25_texts = [
            t for t, m in zip(bm25.texts, bm25.metadata)
            if m.get("source") != name
        ]
        kept_bm25_meta = [
            m for m in bm25.metadata
            if m.get("source") != name
        ]
        bm25.__init__(kept_bm25_texts, kept_bm25_meta if kept_bm25_meta else None)

        # Save updated indexes
        INDEX_DIR.mkdir(exist_ok=True)
        vs.save(INDEX_DIR / "vector_store.faiss")
        bm25.save(INDEX_DIR / "bm25_store.bm25.json")

        try:
            from app.cloud_storage import upload_indexes
            upload_indexes(INDEX_DIR)
        except Exception:
            pass

        print(f"🗑 Removed {name} from indexes")

    except Exception as e:
        print(f"⚠️ Failed to remove {name}: {e}")