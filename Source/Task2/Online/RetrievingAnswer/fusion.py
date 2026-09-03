"""Fuse Task 2 BM25 and dense rankings with deterministic RRF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(slots=True)
class FusedCandidate:
    """Preserve a chunk's fused score and component provenance."""

    chunk_id: str
    score: float = 0.0
    component_scores: dict[str, float] = field(default_factory=dict)
    component_ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    constant: int = 60,
    limit: int = 50,
    weights: Mapping[str, float] | None = None,
) -> list[FusedCandidate]:
    """Fuse ranked chunk IDs while retaining channel score and rank traces."""
    if constant <= 0:
        raise ValueError("RRF constant must be positive.")
    if limit <= 0:
        raise ValueError("RRF limit must be positive.")
    channel_weights = dict(weights or {})
    if any(weight < 0 for weight in channel_weights.values()):
        raise ValueError("RRF weights must be non-negative.")

    fused: dict[str, FusedCandidate] = {}
    for channel, ranking in rankings.items():
        _accumulate_channel(fused, channel, ranking, constant, channel_weights.get(channel, 1.0))
    return sorted(fused.values(), key=lambda item: (-item.score, item.chunk_id))[:limit]


def _accumulate_channel(
    fused: dict[str, FusedCandidate],
    channel: str,
    ranking: Sequence[tuple[str, float]],
    constant: int,
    weight: float,
) -> None:
    if not channel.strip():
        raise ValueError("RRF channel name must be non-empty.")
    seen: set[str] = set()
    for rank, (chunk_id, raw_score) in enumerate(ranking, start=1):
        if not chunk_id.strip() or chunk_id in seen:
            continue
        seen.add(chunk_id)
        candidate = fused.setdefault(chunk_id, FusedCandidate(chunk_id=chunk_id))
        candidate.score += weight / (constant + rank)
        candidate.component_scores[channel] = float(raw_score)
        candidate.component_ranks[channel] = rank
