from __future__ import annotations

from dataclasses import dataclass

from app.bm25_store import BM25Store
from app.embedding import embed_query
from app.reranker import RankedResult, reciprocal_rank_fusion, rerank
from app.vector_store import VectorStore


@dataclass(frozen=True)
class PipelineConfig:
    top_k_retrieve: int = 20
    top_k_final: int = 3
    score_threshold: float = -5.0  # ✅ Fixed: was 0.0, which killed vague queries

@dataclass(frozen=True)
class SummaryConfig(PipelineConfig):
    top_k_retrieve: int = 60
    top_k_final: int = 10  # Reduced to prevent LLaMA-3 8k token limit exceptions
    score_threshold: float = -100.0  # Accept any score for summary to ensure chunk density


def retrieve(query: str, vs: VectorStore, bm25: BM25Store, *, top_k: int, metadata_filter: dict | None = None) -> tuple[list, list]:
    q_embedding = embed_query(query)
    dense_results = vs.search(q_embedding, k=top_k, metadata_filter=metadata_filter)
    sparse_results = bm25.search(query, k=top_k, metadata_filter=metadata_filter)
    return dense_results, sparse_results


def rerank_fused(
    query: str,
    dense_results: list,
    sparse_results: list,
    *,
    top_k: int,
    score_threshold: float,
) -> list[RankedResult]:
    dense_texts = [r.text for r in dense_results]
    sparse_texts = [r.text for r in sparse_results]
    fused_texts = reciprocal_rank_fusion([dense_texts, sparse_texts])

    meta_lookup: dict[str, dict] = {
        **{r.text: getattr(r, "metadata", {}) for r in sparse_results},
        **{r.text: getattr(r, "metadata", {}) for r in dense_results},
    }

    return rerank(
        query=query,
        documents=[{"text": t, **meta_lookup.get(t, {})} for t in fused_texts],
        top_k=top_k,
        score_threshold=score_threshold,
    )


def run_pipeline(
    query: str,
    vs: VectorStore,
    bm25: BM25Store,
    *,
    config: PipelineConfig = PipelineConfig(),
    metadata_filter: dict | None = None,
) -> list[RankedResult]:
    dense_results, sparse_results = retrieve(query, vs, bm25, top_k=config.top_k_retrieve, metadata_filter=metadata_filter)
    return rerank_fused(
        query,
        dense_results,
        sparse_results,
        top_k=config.top_k_final,
        score_threshold=config.score_threshold,
    )