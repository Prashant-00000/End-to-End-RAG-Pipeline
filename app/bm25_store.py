from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class BM25Result:
    text:          str
    score:         float
    original_index: int
    metadata:      dict = field(default_factory=dict)


# ── Tokenizer ──────────────────────────────────────────────────────────────────

# Stopwords matter for BM25 — they inflate IDF scores for common words
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "not", "this", "that",
    "it", "its", "as", "if", "so", "than", "then", "when", "which",
})

def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Lowercase → strip punctuation → split → remove stopwords.

    Much better than .split() for BM25:
      - "AI,"  →  "ai"   (not a separate token from "AI")
      - "U.S." →  "us"   (punctuation stripped)
      - "the"  →  dropped (stopword)

    For domain-specific corpora (medical, legal, code) you may want
    to pass remove_stopwords=False to keep technical short tokens.
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation except hyphens (keep "state-of-the-art" as one token)
    text = re.sub(r"[^\w\s-]", " ", text)

    # Split on whitespace and hyphens
    tokens = re.split(r"[\s\-]+", text)

    # Filter empty strings and stopwords
    if remove_stopwords:
        return [t for t in tokens if t and t not in _STOPWORDS]
    return [t for t in tokens if t]


# ── BM25 store ─────────────────────────────────────────────────────────────────

class BM25Store:
    """
    BM25Okapi keyword store with proper tokenization, persistence,
    incremental updates, scores, and metadata passthrough.
    """

    def __init__(self, chunks: Optional[list[str]] = None, remove_stopwords: bool = True):
        """
        Create an empty or initialized store.

        Args:
            chunks:           Optional list of text chunks to add initially.
            remove_stopwords: Strip common words before indexing.
                              Set False for technical/code corpora.
        """
        self.chunks:    list[str]        = []
        self.metadata:  list[dict]       = []
        self._tokenized: list[list[str]] = []
        self._bm25:     Optional[BM25Okapi] = None
        self._remove_stopwords = remove_stopwords
        self._dirty = False              # tracks whether index needs rebuild
        
        if chunks:
            self.add(chunks)

    # ── Indexing ───────────────────────────────────────────────────────────────

    def add(
        self,
        chunks: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Add documents to the store. Can be called multiple times.

        BM25Okapi must be rebuilt on every add (library limitation),
        but we batch the rebuild so adding 100 docs costs one rebuild,
        not 100.

        Args:
            chunks:   Raw text chunks.
            metadata: Optional list of dicts — one per chunk.
        """
        if not chunks:
            return

        meta = metadata or [{} for _ in chunks]
        if len(meta) != len(chunks):
            raise ValueError("metadata length must match chunks length")

        self.chunks.extend(chunks)
        self.metadata.extend(meta)
        self._tokenized.extend(tokenize(c, self._remove_stopwords) for c in chunks)
        self._dirty = True              # defer rebuild until next search

    def _rebuild_index(self) -> None:
        """Rebuild BM25Okapi from current tokenized corpus."""
        if not self._tokenized:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokenized)
        self._dirty = False

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[BM25Result]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query:           Raw query string — tokenized the same way as docs.
            k:               Max results to return.
            score_threshold: Drop results with BM25 score below this.
                             BM25 scores are corpus-relative (not [0,1]),
                             so start at 0.0 to filter zero-match results.

        Returns:
            List of BM25Result sorted by score descending.
        """
        if not self.chunks:
            return []

        if self._dirty:
            self._rebuild_index()

        if self._bm25 is None:
            return []

        query_tokens = tokenize(query, self._remove_stopwords)
        if not query_tokens:
            return []

        scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

        # Pair scores with indices, filter, sort
        pairs = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build results up to k, with optional threshold
        results: list[BM25Result] = []
        for idx, score in pairs:
            if len(results) >= k:
                break
            if score_threshold is not None and score < score_threshold:
                break                   # sorted desc — no point continuing
            
            if metadata_filter and not all(self.metadata[idx].get(key) == val for key, val in metadata_filter.items()):
                continue

            results.append(BM25Result(
                text=self.chunks[idx],
                score=score,
                original_index=idx,
                metadata=self.metadata[idx],
            ))

        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save the corpus to disk as JSON.

        BM25Okapi is not serialisable — it's rebuilt from the corpus
        on load, which is fast and avoids pickle security issues.

        Creates one file: <path>.bm25.json
        """
        path = Path(path).with_suffix(".bm25.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "chunks":           self.chunks,
            "metadata":         self.metadata,
            "remove_stopwords": self._remove_stopwords,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved {len(self.chunks)} BM25 docs → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Store":
        """
        Load a previously saved store and rebuild the BM25 index.

        Usage:
            store = BM25Store.load("indexes/my_docs")
        """
        path = Path(path).with_suffix(".bm25.json")
        if not path.exists():
            raise FileNotFoundError(f"BM25 store not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        store = cls(remove_stopwords=data.get("remove_stopwords", True))
        store.add(data["chunks"], data.get("metadata"))
        store._rebuild_index()          # eager build on load

        print(f"📂 Loaded {len(store.chunks)} BM25 docs ← {path}")
        return store

    # ── Stats ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        status = "dirty" if self._dirty else "indexed"
        return f"BM25Store(docs={len(self.chunks)}, status={status})"