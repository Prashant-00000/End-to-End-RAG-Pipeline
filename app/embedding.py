from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Model singleton (lazy-loaded) ───────────────────────────────────────────────

_model: Optional[object] = None
_model_name: Optional[str] = None

def get_model(model_name: str = "BAAI/bge-small-en"):
    """Lazy-load embedding model only when needed."""
    global _model, _model_name
    
    if _model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 Loading embedding model on {device}...")
            _model = SentenceTransformer(model_name, device=device)
            _model_name = model_name
        except ImportError as e:
            print(f"❌ sentence-transformers not available: {e}")
            raise
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    return _model

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

def embed_documents(
    chunks: list[str],
    batch_size: int = 32,
    normalize: bool = True,
    metadata: Optional[list[dict]] = None,
    model_name: str = "BAAI/bge-small-en",
    progress_callback: Optional[callable] = None,
) -> list[EmbeddedChunk]:
    """
    Embed document chunks for indexing (lazy-loads model).
    """
    if not chunks:
        return []

    if metadata and len(metadata) != len(chunks):
        raise ValueError(
            f"metadata length ({len(metadata)}) must match chunks length ({len(chunks)})"
        )

    model = get_model(model_name)

    # BGE doc prefix
    prefixed = [BGE_DOC_PREFIX + c for c in chunks] if BGE_DOC_PREFIX else chunks

    if progress_callback is None:
        raw = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(chunks) > 100,
            convert_to_numpy=True,
        )
    else:
        raw_list = []
        total = len(prefixed)
        for i in range(0, total, batch_size):
            batch = prefixed[i:i + batch_size]
            emb = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            raw_list.append(emb)
            progress_callback(min(total, i + batch_size), total)
        raw = np.vstack(raw_list)

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
    """Embed a single query."""
    model = get_model(model_name)
    query_with_prefix = BGE_QUERY_PREFIX + query
    embedding = model.encode(
        query_with_prefix,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding

def embed_chunks(chunks: list[str], **kwargs) -> list[EmbeddedChunk]:
    """Drop-in replacement for your original embed_chunks — same name, richer output."""
    return embed_documents(chunks, **kwargs)