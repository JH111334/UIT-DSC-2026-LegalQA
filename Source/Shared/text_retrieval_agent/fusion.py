"""Fuse sparse rankings while preserving retrieval evidence."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from text_retrieval_agent.contracts import ChunkRecord, RetrievedChunk, RetrieverCandidate


def reciprocal_rank_fusion(
    candidates: Mapping[str, Sequence[RetrieverCandidate]],
    chunks: Mapping[str, ChunkRecord],
    *,
    rrf_k: int,
    top_k: int,
    config_version: str,
) -> tuple[RetrievedChunk, ...]:
    """Fuse candidates with deterministic ordering and raw score retention."""
    fused: defaultdict[str, float] = defaultdict(float)
    scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for retriever_candidates in candidates.values():
        for candidate in retriever_candidates:
            fused[candidate.chunk_id] += 1.0 / (rrf_k + candidate.rank)
            scores[candidate.chunk_id][candidate.retriever] = candidate.raw_score
            ranks[candidate.chunk_id][candidate.retriever] = candidate.rank
    ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:top_k]
    return tuple(
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=chunks[chunk_id].document_id,
            title=chunks[chunk_id].title,
            excerpt=chunks[chunk_id].text,
            source_uri=chunks[chunk_id].source_uri,
            access_scope=chunks[chunk_id].access_scope,
            rank=rank,
            final_score=fused[chunk_id],
            component_scores=scores[chunk_id],
            component_ranks=ranks[chunk_id],
            corpus_version=chunks[chunk_id].corpus_version,
            retrieval_config_version=config_version,
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    )
