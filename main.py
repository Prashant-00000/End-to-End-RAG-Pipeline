from __future__ import annotations

import sys
import threading
from pathlib import Path

from app.ingestion   import load_pdfs, docs_to_texts_and_metadata
from app.cloud_storage import upload_indexes
from app.chunking    import hybrid_chunking
from app.embedding   import embed_documents, EmbeddedChunk
from app.vector_store import VectorStore
from app.bm25_store  import BM25Store
from app.reranker    import rerank, reciprocal_rank_fusion, RankedResult
from app.groq_client import generate
from app.pipeline import PipelineConfig, SummaryConfig, run_pipeline

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Windows consoles often use cp1252; avoid UnicodeEncodeError on log emoji.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("data")
INDEX_DIR       = Path("indexes")
EMBEDDING_DIM   = 384         # bge-small-en output dimension
CHUNK_MAX_WORDS = 400
SEMANTIC_THRESH = 0.45
TOP_K_RETRIEVE  = 20          # candidates fed into reranker
TOP_K_FINAL     = 5           # results passed to LLM (broader context for overview questions)
SCORE_THRESHOLD = -5.0        # cross-encoder scores can be negative; keep more for overview queries


# ── Background Task Registry ───────────────────────────────────────────────────
UPLOAD_TASKS: dict[str, dict] = {}
INDEX_LOCK = threading.Lock()


# ── Index building ─────────────────────────────────────────────────────────────

def build_indexes() -> tuple[VectorStore, BM25Store]:
    """
    Full ingestion pipeline: PDF → chunks → embeddings → persisted indexes.
    Only needs to run once. Subsequent runs use load_indexes().
    """
    print("\n" + "="*60)
    print("BUILDING INDEXES")
    print("="*60)

    # 1. Load PDFs
    pdf_paths = [str(p) for p in DATA_DIR.glob("*.pdf")]
    if not pdf_paths:
        print("⚠️ No PDFs found in the data directory. Creating empty indexes.")
        vs_empty = VectorStore(dim=EMBEDDING_DIM)
        bm25_empty = BM25Store()
        INDEX_DIR.mkdir(exist_ok=True)
        vs_empty.save(INDEX_DIR / "vector_store")
        bm25_empty.save(INDEX_DIR / "bm25_store")
        return vs_empty, bm25_empty
        
    docs = load_pdfs(pdf_paths)
    if not docs:
        raise RuntimeError(
            "No pages loaded — check your PDFs."
        )

    texts, metadata = docs_to_texts_and_metadata(docs)
    print(f"\n📄 {len(docs)} pages loaded across {len(pdf_paths)} file(s)")

    # 2. Chunk each page independently (prevents semantic bleeding across docs)
    all_chunks: list[str]  = []
    all_meta:   list[dict] = []

    for text, meta in zip(texts, metadata):
        page_chunks = hybrid_chunking(
            text,
            max_words=CHUNK_MAX_WORDS,
            semantic_threshold=SEMANTIC_THRESH,
        )
        all_chunks.extend(page_chunks)
        all_meta.extend([meta] * len(page_chunks))

    print(f"📦 {len(all_chunks)} chunks created")

    if not all_chunks:
        raise RuntimeError(
            "No chunks created — your PDFs may be scanned/image-based. "
            "Try enabling ocr_fallback=True in load_pdfs()."
        )

    # 3. Embed — uses BGE doc prefix + L2 normalization
    print("\n🔢 Embedding chunks (this may take a minute)...")
    embedded: list[EmbeddedChunk] = embed_documents(
        all_chunks,
        metadata=all_meta,
    )
    print(f"✅ {len(embedded)} embeddings created")

    # 4. Build and persist vector store
    vs = VectorStore(dim=EMBEDDING_DIM)
    vs.add(
        embeddings=np.stack([e.embedding for e in embedded]),
        texts=[e.text for e in embedded],
        metadata=[e.metadata for e in embedded],
    )

    # 5. Build and persist BM25 store
    bm25 = BM25Store()
    bm25.add(all_chunks, metadata=all_meta)

    # 6. Save both indexes to disk
    INDEX_DIR.mkdir(exist_ok=True)
    vs.save(INDEX_DIR / "vector_store")
    bm25.save(INDEX_DIR / "bm25_store")

    print(f"\n💾 Indexes saved to {INDEX_DIR}/")
    return vs, bm25


def load_indexes() -> tuple[VectorStore, BM25Store]:
    """Load persisted indexes from disk — fast path for all runs after the first."""
    print("\n📂 Loading existing indexes...")
    vs   = VectorStore.load(INDEX_DIR / "vector_store")
    bm25 = BM25Store.load(INDEX_DIR / "bm25_store")
    return vs, bm25

def async_add_document(file_path: Path | str, vs: VectorStore, bm25: BM25Store) -> None:
    """Spawns a thread to process a new document sequentially without blocking the UI."""
    filename = Path(file_path).name
    UPLOAD_TASKS[filename] = {"status": "Starting index...", "progress": 0.0}

    def bg_task():
        try:
            UPLOAD_TASKS[filename] = {"status": "Loading PDF pages...", "progress": 0.1}
            docs = load_pdfs([str(file_path)])
            if not docs:
                UPLOAD_TASKS[filename] = {"status": "Error: No pages found.", "progress": 1.0}
                return

            UPLOAD_TASKS[filename] = {"status": "Extracting text...", "progress": 0.2}
            texts, metadata = docs_to_texts_and_metadata(docs)
            
            all_chunks = []
            all_meta = []
            
            UPLOAD_TASKS[filename] = {"status": "Semantic chunking...", "progress": 0.3}
            for text, meta in zip(texts, metadata):
                page_chunks = hybrid_chunking(
                    text,
                    max_words=CHUNK_MAX_WORDS,
                    semantic_threshold=SEMANTIC_THRESH,
                )
                all_chunks.extend(page_chunks)
                all_meta.extend([meta] * len(page_chunks))
                
            if not all_chunks:
                UPLOAD_TASKS[filename] = {"status": "Error: No text extracted.", "progress": 1.0}
                return
                
            def progress_cb(current, total):
                # Scale from 0.4 to 0.9 depending on progress
                pct = 0.4 + (current / total) * 0.5
                UPLOAD_TASKS[filename] = {
                    "status": f"Embedding {current} of {total} chunks...", 
                    "progress": pct
                }

            embedded = embed_documents(all_chunks, metadata=all_meta, progress_callback=progress_cb)
            
            UPLOAD_TASKS[filename] = {"status": "Saving to indexes...", "progress": 0.95}

            # Safely lock the globals to prevent concurrent search/add corruption
            with INDEX_LOCK:
                vs.add(
                    embeddings=np.stack([e.embedding for e in embedded]),
                    texts=[e.text for e in embedded],
                    metadata=[e.metadata for e in embedded],
                )
                bm25.add(all_chunks, metadata=all_meta)
                INDEX_DIR.mkdir(exist_ok=True)
                vs.save(INDEX_DIR / "vector_store")
                bm25.save(INDEX_DIR / "bm25_store")
                
            UPLOAD_TASKS[filename] = {"status": "Syncing indices to cloud...", "progress": 0.98}
            upload_indexes(INDEX_DIR)

            UPLOAD_TASKS[filename] = {"status": "Done", "progress": 1.0}
            print(f"✅ Background logic for {filename} finished successfully.")

        except Exception as e:
            UPLOAD_TASKS[filename] = {"status": f"Error: {e}", "progress": 1.0}

    thread = threading.Thread(target=bg_task, daemon=True)
    thread.start()

def remove_document_fast(file_path: Path | str, vs: VectorStore, bm25: BM25Store) -> bool:
    """Instantly remove a document's chunks from memory and save, bypassing full index rebuilds."""
    filename = Path(file_path).name
    print(f"\n🗑️ Fast-removing {filename} from indexes...")
    
    # Filter VectorStore
    keep_indices = [i for i, meta in enumerate(vs.metadata) if meta.get("source") != filename]
    
    if len(keep_indices) == len(vs.metadata):
        print(f"⚠️ Document {filename} not found in index.")
        return False
        
    if not keep_indices:
        print("⚠️ All documents removed. Creating empty indexes.")
        vs_empty = VectorStore(dim=EMBEDDING_DIM)
        bm_empty = BM25Store()
        INDEX_DIR.mkdir(exist_ok=True)
        vs_empty.save(INDEX_DIR / "vector_store")
        bm_empty.save(INDEX_DIR / "bm25_store")
        return True
        
    # Reconstruct vectors for kept items
    vectors = np.vstack([vs.index.reconstruct(i) for i in keep_indices])
    new_texts = [vs.texts[i] for i in keep_indices]
    new_meta = [vs.metadata[i] for i in keep_indices]
    
    new_vs = VectorStore(dim=EMBEDDING_DIM)
    new_vs.add(embeddings=vectors, texts=new_texts, metadata=new_meta)
    
    # Filter BM25Store
    bm_keep = [i for i, meta in enumerate(bm25.metadata) if meta.get("source") != filename]
    new_bm_texts = [bm25.chunks[i] for i in bm_keep]
    new_bm_meta = [bm25.metadata[i] for i in bm_keep]
    
    new_bm = BM25Store()
    new_bm.add(chunks=new_bm_texts, metadata=new_bm_meta)
    
    new_vs.save(INDEX_DIR / "vector_store")
    new_bm.save(INDEX_DIR / "bm25_store")
    print(f"✅ Fast removal complete. Remaining chunks: {len(new_texts)}")
    return True


# ── Query pipeline ─────────────────────────────────────────────────────────────

def query_pipeline(
    query: str,
    vs: VectorStore,
    bm25: BM25Store,
    is_summary_mode: bool = False,
    metadata_filter: dict | None = None,
) -> list[RankedResult]:
    if is_summary_mode:
        config = SummaryConfig()
    else:
        config = PipelineConfig(
            top_k_retrieve=TOP_K_RETRIEVE,
            top_k_final=TOP_K_FINAL,
            score_threshold=SCORE_THRESHOLD,
        )

    results = run_pipeline(query, vs, bm25, config=config, metadata_filter=metadata_filter)
    print(f"\n🔍 Reranked hits: {len(results)}")
    return results


# ── Generation ─────────────────────────────────────────────────────────────────

def run_generation(
    query: str,
    results: list[RankedResult],
    *,
    stream: bool = False,
    chat_history: list[dict] | None = None,
) -> str:
    """Convert RankedResults → chunk dicts → Groq answer."""
    if not results:
        return "No relevant context found to answer this question."

    if not stream:
        print("🤖 Generating answer with Groq...")
    return generate(query, results, chat_history=chat_history, stream=stream)


# ── Display ────────────────────────────────────────────────────────────────────

def print_results(results: list[RankedResult], query: str) -> None:
    """Print reranked source chunks with scores and provenance."""
    print(f"\n{'='*60}")
    print(f"RETRIEVED CHUNKS for: {query}")
    print(f"{'='*60}")

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        src  = r.metadata.get("source", "unknown")
        page = r.metadata.get("page_number", "?")
        preview = r.text[:300] + ("..." if len(r.text) > 300 else "")
        print(f"\n{i}. [{src} — p.{page}] (reranker score: {r.score:.3f})")
        print(f"   {preview}")
        print("-" * 60)


def print_answer(answer: str, query: str) -> None:
    """Print the final generated answer."""
    print(f"\n{'='*60}")
    print(f"ANSWER for: {query}")
    print(f"{'='*60}")
    print(f"\n{answer}\n")


# ── Interactive mode ───────────────────────────────────────────────────────────

def interactive_mode(
    vs: VectorStore,
    bm25: BM25Store,
    *,
    stream: bool = False,
    show_sources: bool = False,
) -> None:
    """REPL loop — ask questions until the user types 'exit'."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE  (type 'exit' to quit)")
    print("="*60)

    chat_history: list[dict] = []

    while True:
        try:
            query = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        results = query_pipeline(query, vs, bm25)
        if stream:
            print(f"\n{'='*60}")
            print(f"ANSWER for: {query}")
            print(f"{'='*60}")
            answer = run_generation(query, results, stream=True, chat_history=chat_history)
        else:
            answer  = run_generation(query, results, chat_history=chat_history)
            print_answer(answer, query)
        if show_sources:
            print_results(results, query)

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load or build indexes ──────────────────────────────────────────────────
    faiss_path = INDEX_DIR / "vector_store.faiss"

    if faiss_path.exists():
        vs, bm25 = load_indexes()
    else:
        print("🔨 First run — building indexes from PDFs...")
        vs, bm25 = build_indexes()

    # ── Query mode ─────────────────────────────────────────────────────────────
    #
    #   Single query:     python main.py "What is generative AI?"
    #   Interactive mode: python main.py
    #   Rebuild indexes:  python main.py --rebuild
    #
    args = sys.argv[1:]

    if args and args[0] == "--rebuild":
        print("🔄 Rebuilding indexes from scratch...")
        vs, bm25 = build_indexes()
        args = args[1:]             # allow a query after --rebuild

    debug = False
    stream = False
    while args and args[0] in {"--debug", "--stream"}:
        if args[0] == "--debug":
            debug = True
        elif args[0] == "--stream":
            stream = True
        args = args[1:]

    if args:
        # Single query from CLI
        query   = " ".join(args)
        results = query_pipeline(query, vs, bm25)
        if stream:
            print(f"\n{'='*60}")
            print(f"ANSWER for: {query}")
            print(f"{'='*60}")
            run_generation(query, results, stream=True)
        else:
            answer  = run_generation(query, results)
            print_answer(answer, query)
        if debug:
            print_results(results, query)
    else:
        # No query provided — drop into interactive REPL
        interactive_mode(vs, bm25, stream=stream, show_sources=debug)