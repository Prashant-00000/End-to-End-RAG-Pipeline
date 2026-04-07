from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Try to import FAISS, but allow fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ FAISS not available - vector search disabled")
    FAISS_AVAILABLE = False


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    text:     str
    score:    float        # cosine similarity — higher is better
    index:    int          # position in the store
    metadata: dict = field(default_factory=dict)


# ── Vector store ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS-backed vector store (falls back gracefully if FAISS unavailable).
    """

    def __init__(self, dim: int, index_type: str = "flat"):
        """Initialize vector store."""
        self.dim = dim
        self.index = None
        self.texts:    list[str]  = []
        self.metadata: list[dict] = []
        self._index_type = index_type
        
        if FAISS_AVAILABLE:
            self.index = self._build_index(index_type, dim)
        else:
            print("⚠️ Using mock vector store (FAISS unavailable)")

    # ── Indexing ───────────────────────────────────────────────────────────────

    def add(
        self,
        embeddings: np.ndarray | list,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """Add embeddings + texts to the store."""
        vectors = self._validate_embeddings(embeddings)

        if len(vectors) != len(texts):
            raise ValueError(f"embeddings and texts must have equal length")

        meta = metadata or [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata length must match texts length")

        self.texts.extend(texts)
        self.metadata.extend(meta)

        if FAISS_AVAILABLE and self.index is not None:
            try:
                self.index.add(vectors)
            except Exception as e:
                print(f"⚠️ FAISS add failed: {e}")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Find the k most similar chunks to a query embedding.
        """
        if len(self.texts) == 0:
            return []

        # If FAISS not available, return empty results
        if not FAISS_AVAILABLE:
            return []

        vec = self._validate_embeddings(query_embedding).reshape(1, -1)

        if metadata_filter:
            valid_indices = [
                i for i, meta in enumerate(self.metadata)
                if all(meta.get(key) == val for key, val in metadata_filter.items())
            ]
            if not valid_indices:
                return []
                
            sub_vecs = np.vstack([self.index.reconstruct(i) for i in valid_indices])
            scores = np.dot(sub_vecs, vec[0])
            
            top_k_idx = np.argsort(scores)[::-1][:min(k, len(valid_indices))]
            
            results = []
            for local_i in top_k_idx:
                score = float(scores[local_i])
                if score_threshold is not None and score < score_threshold:
                    continue
                global_i = valid_indices[local_i]
                results.append(SearchResult(
                    text=self.texts[global_i],
                    score=score,
                    index=global_i,
                    metadata=self.metadata[global_i],
                ))
            return results

        try:
            safe_k = min(k, len(self.texts))
            scores, indices = self.index.search(vec, safe_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if score_threshold is not None and score < score_threshold:
                    continue
                results.append(SearchResult(
                    text=self.texts[idx],
                    score=float(score),
                    index=int(idx),
                    metadata=self.metadata[idx],
                ))
            return results
        except Exception as e:
            print(f"⚠️ Search failed: {e}")
            return []

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save index + texts + metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Only save FAISS index if we have vectors
        if len(self.texts) > 0 and FAISS_AVAILABLE and self.index is not None:
            try:
                import faiss
                faiss.write_index(self.index, str(path.with_suffix(".faiss")))
            except Exception as e:
                print(f"⚠️ Could not save FAISS index: {e}")

        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(
                {"texts": self.texts, "metadata": self.metadata, "dim": self.dim},
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"💾 Saved {len(self.texts)} vectors → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load a previously saved store from disk."""
        path = Path(path)
        json_path  = path.with_suffix(".json")

        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        store = cls.__new__(cls)
        store.dim      = data["dim"]
        store.texts    = data["texts"]
        store.metadata = data.get("metadata", [{} for _ in data["texts"]])
        store._index_type = "loaded"
        store.index = None

        if FAISS_AVAILABLE:
            try:
                import faiss
                faiss_path = path.with_suffix(".faiss")
                if faiss_path.exists():
                    store.index = faiss.read_index(str(faiss_path))
            except Exception as e:
                print(f"⚠️ Could not load FAISS index: {e}")

        print(f"📂 Loaded {len(store.texts)} vectors from {json_path}")
        return store

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _validate_embeddings(self, embeddings: np.ndarray | list) -> np.ndarray:
        """Ensure float32 contiguous array of correct dimension."""
        arr = np.array(embeddings, dtype=np.float32)

        # Handle flat 1-D vector (single embedding)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"Expected shape (N, {self.dim}), got {arr.shape}"
            )

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        return arr

    @staticmethod
    def _build_index(index_type: str, dim: int):
        """Build FAISS index if available, otherwise return None."""
        if not FAISS_AVAILABLE:
            return None

        try:
            import faiss
            if index_type == "flat":
                return faiss.IndexFlatIP(dim)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dim)
                return faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
            elif index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 64
                return index
            else:
                raise ValueError(f"Unknown index_type '{index_type}'")
        except Exception as e:
            print(f"⚠️ Could not build FAISS index: {e}")
            return None

    # ── Stats ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        return (
            f"VectorStore(docs={len(self.texts)}, dim={self.dim}, "
            f"index={self.index.__class__.__name__})"
        )