from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..core.config import ModelSpec
from ..core.io import load_qa, write_json_atomic
from ..core.schema import LegalChunk
from .engine import BM25Index, DenseEncoder, reciprocal_rank_fusion

NeighborStrategy = Literal["bm25", "dense", "rrf"]


def predict_from_training_neighbors(
    train_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    dense_spec: ModelSpec | None = None,
    strategy: NeighborStrategy = "rrf",
    retrieval_top_k: int = 20,
    dense_weight: float = 4.0,
    trace_path: str | Path | None = None,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Answer test questions with the nearest provided training QA example."""

    if retrieval_top_k < 1:
        raise ValueError("retrieval_top_k must be at least 1")
    if dense_weight <= 0:
        raise ValueError("dense_weight must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if strategy in {"dense", "rrf"} and (dense_spec is None or not dense_spec.enabled):
        raise ValueError(f"strategy={strategy!r} requires an enabled dense model")

    notify = progress or (lambda _message: None)
    training = [value for value in load_qa(train_path) if value.answer]
    examples = load_qa(input_path)
    if limit is not None:
        examples = examples[:limit]
    if not training:
        raise ValueError("Training file has no answered examples")

    answers = {value.id: value.answer or "" for value in training}
    chunks = [
        LegalChunk(value.id, value.id, value.id, value.question, "qa", value.id)
        for value in training
    ]
    bm25 = BM25Index.build(chunks)
    bm25_rankings = []
    for index, example in enumerate(examples, 1):
        bm25_rankings.append(bm25.search(example.question, retrieval_top_k))
        if index % 100 == 0 or index == len(examples):
            notify(f"Retrieved training questions with BM25 {index}/{len(examples)}")

    dense_rankings: list[list[tuple[str, float]]] | None = None
    if strategy in {"dense", "rrf"}:
        assert dense_spec is not None
        encoder = DenseEncoder(dense_spec)
        notify(f"Encoding {len(training)} training questions")
        training_embeddings = encoder.encode_documents([value.question for value in training])
        notify(f"Encoding {len(examples)} input questions")
        query_embeddings = encoder.encode_queries([value.question for value in examples])
        dense_rankings = []
        count = min(retrieval_top_k, len(training))
        for index, query_embedding in enumerate(query_embeddings, 1):
            scores = training_embeddings @ query_embedding
            if count == len(scores):
                positions = np.arange(len(scores))
            else:
                positions = np.argpartition(scores, -count)[-count:]
            positions = positions[np.argsort(scores[positions])[::-1]]
            dense_rankings.append(
                [(training[int(position)].id, float(scores[position])) for position in positions]
            )
            if index % 100 == 0 or index == len(examples):
                notify(f"Retrieved training questions with dense search {index}/{len(examples)}")

    predictions: dict[str, dict[str, str]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for index, example in enumerate(examples):
        if strategy == "bm25":
            selected_id = bm25_rankings[index][0][0]
        elif strategy == "dense":
            assert dense_rankings is not None
            selected_id = dense_rankings[index][0][0]
        else:
            assert dense_rankings is not None
            fused = reciprocal_rank_fusion(
                {"bm25": bm25_rankings[index], "dense": dense_rankings[index]},
                weights={"bm25": 1.0, "dense": dense_weight},
                top_k=1,
            )
            selected_id = fused[0].chunk_id
        predictions[example.id] = {
            "question": example.question,
            "answer": answers[selected_id],
        }
        traces[example.id] = {
            "strategy": strategy,
            "neighbor_id": selected_id,
            "bm25_top": bm25_rankings[index][0],
            "dense_top": dense_rankings[index][0] if dense_rankings is not None else None,
        }

    write_json_atomic(output_path, predictions)
    if trace_path:
        write_json_atomic(trace_path, traces)
    return {
        "train": str(train_path),
        "input": str(input_path),
        "output": str(output_path),
        "strategy": strategy,
        "dense_weight": dense_weight if strategy == "rrf" else None,
        "total": len(examples),
        "complete": len(predictions) == len(examples),
    }
