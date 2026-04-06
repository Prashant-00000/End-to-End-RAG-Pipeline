from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


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
    FAISS-backed vector store with persistence, metadata, and safe search.

    Expects L2-normalized embeddings (produced by embedding.py).
    Uses IndexFlatIP (inner product) which equals cosine sim on unit vectors.
    """

    def __init__(self, dim: int, index_type: str = "flat"):
        """
        Args:
            dim:        Embedding dimension — must match your model output.
                        bge-large-en → 1024.
            index_type: "flat"  — exact search, best for < 100k docs.
                        "ivf"   — approximate, faster for > 100k docs.
                        "hnsw"  — approximate, best latency at scale.
        """
        self.dim = dim
        self.index = self._build_index(index_type, dim)
        self.texts:    list[str]  = []
        self.metadata: list[dict] = []
        self._index_type = index_type

    # ── Indexing ───────────────────────────────────────────────────────────────

    def add(
        self,
        embeddings: np.ndarray | list,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Add embeddings + texts to the store atomically.

        Args:
            embeddings: Shape (N, dim), float32, L2-normalized.
            texts:      N strings — one per embedding.
            metadata:   Optional N dicts (source file, page, chunk_id, etc.)
        """
        vectors = self._validate_embeddings(embeddings)

        if len(vectors) != len(texts):
            raise ValueError(
                f"embeddings ({len(vectors)}) and texts ({len(texts)}) must have equal length"
            )

        meta = metadata or [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata length must match texts length")

        # Update texts + metadata BEFORE adding to index so any error
        # above leaves the store in a consistent state
        self.texts.extend(texts)
        self.metadata.extend(meta)

        # IVF index needs training before first add
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            if len(vectors) < self.index.nlist:
                raise RuntimeError(
                    f"IVF index needs at least {self.index.nlist} vectors to train "
                    f"(got {len(vectors)}). Add more documents or use index_type='flat'."
                )
            self.index.train(vectors)

        self.index.add(vectors)

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

        Args:
            query_embedding: Shape (dim,) or (1, dim), L2-normalized.
            k:               Max results. Automatically clamped to store size.
            score_threshold: Drop results with cosine similarity below this.
                             Range is [-1, 1]. Suggested starting point: 0.3.
            metadata_filter: Optional dict of key/values (e.g. {"source": "pdf1.pdf"}).

        Returns:
            List of SearchResult sorted by score descending.
        """
        if len(self.texts) == 0:
            return []

        vec = self._validate_embeddings(query_embedding).reshape(1, -1)

        if metadata_filter:
            # ── Fast NumPy pre-filtering ──
            valid_indices = [
                i for i, meta in enumerate(self.metadata)
                if all(meta.get(key) == val for key, val in metadata_filter.items())
            ]
            if not valid_indices:
                return []
                
            sub_vecs = np.vstack([self.index.reconstruct(i) for i in valid_indices])
            scores = np.dot(sub_vecs, vec[0])
            
            # Sort top K
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

        # Clamp k so FAISS never returns -1 sentinel indices
        safe_k = min(k, len(self.texts))

        scores, indices = self.index.search(vec, safe_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:           # FAISS sentinel for "not enough results"
                continue
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(SearchResult(
                text=self.texts[idx],
                score=float(score),
                index=int(idx),
                metadata=self.metadata[idx],
            ))

        return results          # already sorted best-first by FAISS

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save index + texts + metadata to disk.

        Creates two files:
          <path>.faiss  — the FAISS binary index
          <path>.json   — texts and metadata
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path.with_suffix(".faiss")))

        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(
                {"texts": self.texts, "metadata": self.metadata, "dim": self.dim},
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"💾 Saved {len(self.texts)} vectors → {path.with_suffix('.faiss')}")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """
        Load a previously saved store from disk.

        Usage:
            store = VectorStore.load("indexes/my_docs")
        """
        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        json_path  = path.with_suffix(".json")

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        store = cls.__new__(cls)
        store.dim      = data["dim"]
        store.texts    = data["texts"]
        store.metadata = data.get("metadata", [{} for _ in data["texts"]])
        store.index    = faiss.read_index(str(faiss_path))
        store._index_type = "loaded"

        print(f"📂 Loaded {len(store.texts)} vectors ← {faiss_path}")
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
    def _build_index(index_type: str, dim: int) -> faiss.Index:
        """
        Flat  → exact, no training needed.      Best up to ~100k docs.
        IVF   → approximate, needs training.    Best 100k–1M docs.
        HNSW  → approximate, graph-based.       Best latency at any scale.

        All use inner product (= cosine sim on normalized vectors).
        """
        if index_type == "flat":
            return faiss.IndexFlatIP(dim)

        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            # nlist = number of Voronoi cells; sqrt(N) is the standard heuristic
            return faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)

        elif index_type == "hnsw":
            # M = connections per node; 32 is a good default
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200   # build quality (higher = better, slower)
            index.hnsw.efSearch = 64          # search quality (tune at query time)
            return index

        else:
            raise ValueError(f"Unknown index_type '{index_type}'. Choose: flat, ivf, hnsw")

    # ── Stats ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        return (
            f"VectorStore(docs={len(self.texts)}, dim={self.dim}, "
            f"index={self.index.__class__.__name__})"
        )