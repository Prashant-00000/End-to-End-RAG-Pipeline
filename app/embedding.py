from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# ── BGE requires different prefixes for docs vs queries ───────────────────────
# https://huggingface.co/BAAI/bge-large-en
BGE_QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "
BGE_DOC_PREFIX    = ""   # docs get no prefix for bge-large-en

# ── Model singleton ───────────────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None

def get_model(model_name: str = "BAAI/bge-small-en") -> SentenceTransformer:
    """Lazy-load once, reuse forever. Picks GPU if available."""
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading embedding model on {device}...")
        _model = SentenceTransformer(model_name, device=device)
    return _model


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    """Pairs a text chunk with its embedding + any metadata you want to carry."""
    text:      str
    embedding: np.ndarray
    metadata:  dict = field(default_factory=dict)   # doc_id, page, source, etc.


# ── Embedding functions ───────────────────────────────────────────────────────

def embed_documents(
    chunks: list[str],
    batch_size: int = 64,
    normalize: bool = True,
    metadata: Optional[list[dict]] = None,
    model_name: str = "BAAI/bge-small-en",
    progress_callback: Optional[callable] = None,
) -> list[EmbeddedChunk]:
    """
    Embed document chunks for indexing.

    Args:
        chunks:     Raw text chunks from your chunking pipeline.
        batch_size: Tune down if you hit OOM on large docs.
        normalize:  L2-normalize embeddings — required for correct cosine sim
                    with BGE models. Keep True unless you know what you're doing.
        metadata:   Optional list of dicts (one per chunk) — e.g. page number,
                    source filename, chunk index. Stored on EmbeddedChunk.
        model_name: Swap model without changing call sites.

    Returns:
        List of EmbeddedChunk — text + embedding + metadata together.
    """
    if not chunks:
        return []

    if metadata and len(metadata) != len(chunks):
        raise ValueError(
            f"metadata length ({len(metadata)}) must match chunks length ({len(chunks)})"
        )

    model = get_model(model_name)

    # BGE doc prefix is empty for bge-large-en, but explicit is safer
    prefixed = [BGE_DOC_PREFIX + c for c in chunks] if BGE_DOC_PREFIX else chunks

    if progress_callback is None:
        raw = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=normalize,   # ← critical for BGE
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
            text=chunk,
            embedding=raw[i],
            metadata=metadata[i] if metadata else {},
        )
        for i, chunk in enumerate(chunks)
    ]


def embed_query(
    query: str,
    normalize: bool = True,
    model_name: str = "BAAI/bge-small-en",
) -> np.ndarray:
    """
    Embed a user query for retrieval.

    BGE uses a different instruction prefix for queries than documents —
    this is what makes it an 'instruction-tuned' retrieval model.
    Skipping the prefix silently degrades retrieval quality.
    """
    model = get_model(model_name)

    embedding = model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embedding


# ── Convenience re-export ─────────────────────────────────────────────────────

def embed_chunks(chunks: list[str], **kwargs) -> list[EmbeddedChunk]:
    """Drop-in replacement for your original embed_chunks — same name, richer output."""
    return embed_documents(chunks, **kwargs)