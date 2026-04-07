from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ── Try to load sentence-transformers, fall back to TF-IDF ─────────────────────

_model: Optional[object] = None
_use_tfidf: bool = False

def get_model(model_name: str = "BAAI/bge-small-en"):
    """
    Try to load sentence-transformers model.
    Falls back to random embeddings if not available.
    """
    global _model, _use_tfidf
    
    if _model is not None:
        return _model
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading embedding model on {device}...")
        _model = SentenceTransformer(model_name, device=device)
        return _model
        
    except Exception as e:
        print(f"⚠️ Sentence-transformers unavailable: {e}")
        print("📊 Using random embeddings (search may be degraded)")
        _use_tfidf = True
        return None

# ── BGE configuration ───────────────────────────────────────────────────────────
BGE_QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "
BGE_DOC_PREFIX    = ""

# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    """Pairs a text chunk with its embedding + any metadata you want to carry."""
    text:      str
    embedding: np.ndarray
    metadata:  dict = field(default_factory=dict)

# ── Embedding functions ───────────────────────────────────────────────────────

def _random_embedding(text: str, dim: int = 1024) -> np.ndarray:
    """Generate a pseudo-random but consistent embedding (fallback)."""
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(dim).astype(np.float32)

def embed_documents(
    chunks: list[str],
    batch_size: int = 32,
    normalize: bool = True,
    metadata: Optional[list[dict]] = None,
    model_name: str = "BAAI/bge-small-en",
    progress_callback: Optional[callable] = None,
) -> list[EmbeddedChunk]:
    """
    Embed document chunks (with fallback to random embeddings).
    """
    if not chunks:
        return []

    if metadata and len(metadata) != len(chunks):
        raise ValueError(
            f"metadata length ({len(metadata)}) must match chunks length ({len(chunks)})"
        )

    model = get_model(model_name)

    # Generate embeddings
    if model is not None:
        # Use sentence-transformers
        prefixed = [BGE_DOC_PREFIX + c for c in chunks] if BGE_DOC_PREFIX else chunks
        raw = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(chunks) > 100,
            convert_to_numpy=True,
        )
    else:
        # Fallback: random embeddings
        print("⚠️ Using random embeddings (install sentence-transformers for better search)")
        raw = np.array([_random_embedding(c) for c in chunks], dtype=np.float32)

    return [
        EmbeddedChunk(
            text=chunks[i],
            embedding=raw[i],
            metadata=metadata[i] if metadata else {},
        )
        for i in range(len(chunks))
    ]

def embed_query(
    query: str,
    model_name: str = "BAAI/bge-small-en",
) -> np.ndarray:
    """Embed a single query (with fallback)."""
    model = get_model(model_name)
    
    if model is not None:
        query_with_prefix = BGE_QUERY_PREFIX + query
        embedding = model.encode(
            query_with_prefix,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding
    else:
        # Fallback
        return _random_embedding(query)

def embed_chunks(chunks: list[str], **kwargs) -> list[EmbeddedChunk]:
    """Drop-in replacement for your original embed_chunks — same name, richer output."""
    return embed_documents(chunks, **kwargs)