from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Try to import CrossEncoder, provide fallback ────────────────────────────────

try:
    from sentence_transformers import CrossEncoder
    _crossencoder_available = True
except ImportError:
    print("⚠️ sentence-transformers not available, reranking will use mock scores")
    _crossencoder_available = False
    CrossEncoder = None

# ── Model registry ─────────────────────────────────────────────────────────────
#
#  Speed  ←————————————————————————→  Quality
#  MiniLM-L-6   MiniLM-L-12   bge-reranker-base   bge-reranker-large
#
RERANKER_MODELS = {
    "fast":    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "balanced":"cross-encoder/ms-marco-MiniLM-L-12-v2",
    "best":    "BAAI/bge-reranker-large",
}

_model: Optional[object] = None
_model_name: Optional[str] = None


def get_model(model_name: str = RERANKER_MODELS["balanced"]):
    """Lazy-load reranker model (with fallback)."""
    global _model, _model_name
    
    if not _crossencoder_available:
        return None
    
    if _model is None or _model_name != model_name:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 Loading reranker '{model_name}' on {device}...")
            _model = CrossEncoder(model_name, device=device, max_length=512)
            _model_name = model_name
        except Exception as e:
            print(f"⚠️ Reranker loading failed: {e}")
            _model = None
    
    return _model


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class RankedResult:
    """A reranked document with its relevance score and original position."""
    text:             str
    score:            float
    original_rank:    int
    metadata:         dict

# ── Core reranker ──────────────────────────────────────────────────────────────

def rerank(
    query:           str,
    documents:       list[str] | list[dict],
    top_k:           int = 5,
    score_threshold: Optional[float] = None,
    batch_size:      int = 32,
    model_name:      str = RERANKER_MODELS["balanced"],
) -> list[RankedResult]:
    """
    Rerank documents against a query using a cross-encoder.

    Args:
        query:            User query string.
        documents:        Either plain strings, or dicts with at least a 'text'
                          key (metadata is preserved on RankedResult).
        top_k:            Max results to return.
        score_threshold:  Drop results below this score. None = keep all top_k.
                          Typical useful range: -5 (loose) to 5 (strict).
        batch_size:       Pairs per forward pass — tune down if OOM.
        model_name:       Override the model (use RERANKER_MODELS constants).

    Returns:
        List of RankedResult sorted by score descending, length ≤ top_k.
    """
    if not documents:
        return []

    # Normalise input — accept plain strings or dicts from your vector store
    texts, metas = _parse_documents(documents)

    pairs = [(query, text) for text in texts]

    model = get_model(model_name)
    
    if model is not None:
        # Use neural reranking
        scores: list[float] = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=len(pairs) > 50,
            convert_to_numpy=True,
        ).tolist()
    else:
        # Fallback: use text similarity or constant scores
        print("⚠️ Using mock scores (install sentence-transformers for better reranking)")
        scores = [float(i) for i in range(len(texts))]  # Just use original order

    # Build results with original rank before sorting
    results = [
        RankedResult(
            text=texts[i],
            score=scores[i],
            original_rank=i,
            metadata=metas[i],
        )
        for i in range(len(texts))
    ]

    # Sort by score
    results.sort(key=lambda r: r.score, reverse=True)

    # Apply threshold before top_k slice
    if score_threshold is not None:
        results = [r for r in results if r.score >= score_threshold]

    return results[:top_k]


# ── Reciprocal Rank Fusion (bonus) ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """
    Merge multiple ranked result lists without needing a trained model.

    Useful for combining BM25 + vector results before passing to the
    cross-encoder — better than interleaving or score normalisation.

    Args:
        result_lists: Each list is a ranked set of doc texts (best first).
        k:            Smoothing constant — 60 is the standard default.

    Returns:
        Merged list sorted by fused score, deduplicated.
    """
    scores: dict[str, float] = {}

    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores, key=lambda d: scores[d], reverse=True)


# ── Backward-compatible shim ───────────────────────────────────────────────────

def rerank_simple(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """Drop-in replacement for your original rerank() — same signature, same return type."""
    results = rerank(query, documents, top_k=top_k)
    return [r.text for r in results]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_documents(
    documents: list[str] | list[dict],
) -> tuple[list[str], list[dict]]:
    """Normalise plain strings or dicts into parallel text + metadata lists."""
    texts, metas = [], []
    for doc in documents:
        if isinstance(doc, str):
            texts.append(doc)
            metas.append({})
        elif isinstance(doc, dict):
            if "text" not in doc:
                raise ValueError(f"Document dict missing 'text' key: {doc}")
            texts.append(doc["text"])
            metas.append({k: v for k, v in doc.items() if k != "text"})
        else:
            raise TypeError(f"Expected str or dict, got {type(doc)}")
    return texts, metas