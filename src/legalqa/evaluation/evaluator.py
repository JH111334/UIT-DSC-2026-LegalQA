from __future__ import annotations

import json
import math
from collections.abc import Callable
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from ..core.config import PipelineConfig, artifact_dir, data_path
from ..core.io import load_qa, read_jsonl, write_json, write_jsonl
from ..core.schema import Candidate
from ..generation.generators import create_generator
from ..retrieval.complexity import (
    analyze_query_complexity,
    analyze_retrieval_confidence,
    choose_dynamic_top_k,
)
from ..retrieval.engine import HybridRetriever
from ..retrieval.evidence import EvidenceAssembler
from .metrics import meteor_exact, rouge_l_f1


def evaluate_diagnostic_retrieval(
    config: PipelineConfig,
    *,
    limits: tuple[int, ...] = (1, 3, 5, 10, 20),
    max_examples: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    records = [
        value
        for value in read_jsonl(artifact_dir(config) / "diagnostic_labels.jsonl")
        if value["status"] == "matched"
    ]
    if max_examples is not None:
        records = records[:max_examples]
    retriever = HybridRetriever(config)
    recalls = {limit: 0 for limit in limits}
    reciprocal_rank = 0.0
    ndcg = 0.0
    evaluated = 0
    for record in records:
        candidates = retriever.retrieve(record["question"])
        gold_chunks = set(record["matched_chunk_ids"])
        gold_parents = set(record["matched_parent_ids"])
        ranked_relevant = []
        for candidate in candidates:
            chunk = retriever.get_chunk(candidate.chunk_id)
            ranked_relevant.append(candidate.chunk_id in gold_chunks or chunk.parent_id in gold_parents)
        if not any(ranked_relevant):
            rank = None
        else:
            rank = ranked_relevant.index(True) + 1
            reciprocal_rank += 1.0 / rank
            ndcg += 1.0 / math.log2(rank + 1)
        for limit in limits:
            recalls[limit] += int(any(ranked_relevant[:limit]))
        evaluated += 1
    report: dict[str, object] = {
        "evaluated": evaluated,
        **{f"recall@{limit}": recalls[limit] / max(1, evaluated) for limit in limits},
        "mrr": reciprocal_rank / max(1, evaluated),
        "ndcg": ndcg / max(1, evaluated),
        "diagnostic_only": True,
        "training_use_prohibited": True,
        "retrieval_settings": {
            "bm25_enabled": config.retrieval.bm25_enabled,
            "dense_enabled": config.retrieval.dense_enabled,
            "legal_metadata_boost": config.retrieval.legal_metadata_boost,
            "rerank_enabled": config.retrieval.rerank_enabled,
            "rrf_top_k": config.retrieval.rrf_top_k,
        },
    }
    destination = Path(output_path) if output_path else artifact_dir(config) / "retrieval_diagnostic_metrics.json"
    write_json(destination, report)
    return report


def evaluate_retrieval_complementarity(
    config: PipelineConfig,
    *,
    limit: int = 10,
    max_examples: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Measure which diagnostic questions are recovered by BM25 and dense retrieval."""

    records = [
        value
        for value in read_jsonl(artifact_dir(config) / "diagnostic_labels.jsonl")
        if value["status"] == "matched"
    ]
    if max_examples is not None:
        records = records[:max_examples]

    bm25_config = deepcopy(config)
    bm25_config.retrieval.bm25_enabled = True
    bm25_config.retrieval.dense_enabled = False
    bm25_config.retrieval.rerank_enabled = False
    bm25_config.retrieval.legal_metadata_boost = 0.0
    bm25_config.retrieval.rrf_top_k = max(limit, bm25_config.retrieval.rrf_top_k)

    dense_config = deepcopy(config)
    dense_config.retrieval.bm25_enabled = False
    dense_config.retrieval.dense_enabled = True
    dense_config.retrieval.rerank_enabled = False
    dense_config.retrieval.legal_metadata_boost = 0.0
    dense_config.retrieval.rrf_top_k = max(limit, dense_config.retrieval.rrf_top_k)

    bm25 = HybridRetriever(bm25_config)
    dense = HybridRetriever(dense_config)
    counts = {"both": 0, "bm25_only": 0, "dense_only": 0, "neither": 0}

    def is_hit(retriever: HybridRetriever, candidates: list[Any], record: dict[str, Any]) -> bool:
        gold_chunks = set(record["matched_chunk_ids"])
        gold_parents = set(record["matched_parent_ids"])
        return any(
            candidate.chunk_id in gold_chunks
            or retriever.get_chunk(candidate.chunk_id).parent_id in gold_parents
            for candidate in candidates[:limit]
        )

    for record in records:
        bm25_hit = is_hit(bm25, bm25.retrieve(record["question"]), record)
        dense_hit = is_hit(dense, dense.retrieve(record["question"]), record)
        if bm25_hit and dense_hit:
            counts["both"] += 1
        elif bm25_hit:
            counts["bm25_only"] += 1
        elif dense_hit:
            counts["dense_only"] += 1
        else:
            counts["neither"] += 1

    evaluated = len(records)
    report: dict[str, object] = {
        "evaluated": evaluated,
        "limit": limit,
        "counts": counts,
        "rates": {key: value / max(1, evaluated) for key, value in counts.items()},
        "bm25_recall": (counts["both"] + counts["bm25_only"]) / max(1, evaluated),
        "dense_recall": (counts["both"] + counts["dense_only"]) / max(1, evaluated),
        "union_recall": (counts["both"] + counts["bm25_only"] + counts["dense_only"])
        / max(1, evaluated),
        "diagnostic_only": True,
        "training_use_prohibited": True,
    }
    destination = (
        Path(output_path)
        if output_path
        else artifact_dir(config) / "dense_bm25_complementarity.json"
    )
    write_json(destination, report)
    return report


def build_diagnostic_candidate_cache(
    config: PipelineConfig,
    *,
    max_examples: int | None = None,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run expensive retrieval once so evidence ablations reuse identical rankings."""

    records = [
        value
        for value in read_jsonl(artifact_dir(config) / "diagnostic_labels.jsonl")
        if value["status"] == "matched"
    ]
    if max_examples is not None:
        records = records[:max_examples]
    retriever = HybridRetriever(config)
    notify = progress or (lambda _message: None)
    cached = []
    for index, record in enumerate(records, 1):
        candidates = retriever.retrieve(record["question"])
        cached.append(
            {
                "id": str(record.get("example_id") or record.get("id") or ""),
                "question": record["question"],
                "matched_chunk_ids": record["matched_chunk_ids"],
                "matched_parent_ids": record["matched_parent_ids"],
                "candidates": [
                    {
                        "chunk_id": value.chunk_id,
                        "score": value.score,
                        "channel_scores": value.channel_scores,
                        "channel_ranks": value.channel_ranks,
                        "provenance": list(value.provenance),
                    }
                    for value in candidates
                ],
            }
        )
        if index % 10 == 0 or index == len(records):
            notify(f"Cached diagnostic candidates {index}/{len(records)}")
    destination = (
        Path(output_path)
        if output_path
        else artifact_dir(config) / "diagnostic_candidate_cache.jsonl"
    )
    write_jsonl(destination, cached)
    report = {
        "evaluated": len(cached),
        "candidate_count_min": min((len(value["candidates"]) for value in cached), default=0),
        "candidate_count_max": max((len(value["candidates"]) for value in cached), default=0),
        "output": str(destination),
        "diagnostic_only": True,
        "training_use_prohibited": True,
    }
    write_json(destination.with_suffix(".report.json"), report)
    return report


def evaluate_diagnostic_evidence(
    config: PipelineConfig,
    cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate Dynamic-K and expansion at the evidence actually packed for QA."""

    records = list(read_jsonl(cache_path))
    assembler = EvidenceAssembler(config)
    hits = 0
    top_k_total = 0
    packed_total = 0
    chars_total = 0
    graph_total = 0
    top_k_distribution: dict[str, int] = {}
    for record in records:
        candidates = [
            Candidate(
                chunk_id=value["chunk_id"],
                score=float(value["score"]),
                channel_scores=dict(value.get("channel_scores") or {}),
                channel_ranks={
                    key: int(rank) for key, rank in (value.get("channel_ranks") or {}).items()
                },
                provenance=tuple(value.get("provenance") or ()),
            )
            for value in record["candidates"]
        ]
        evidence, decision = assembler.assemble(record["question"], candidates)
        gold_chunks = set(record["matched_chunk_ids"])
        gold_parents = set(record["matched_parent_ids"])

        def evidence_is_relevant(item: Any) -> bool:
            if (
                item.id in gold_chunks
                or item.id in gold_parents
                or bool(set(item.chunk_ids) & gold_chunks)
            ):
                return True
            if not assembler.store.has_parent(item.id):
                try:
                    return assembler.store.get_chunk(item.id).parent_id in gold_parents
                except KeyError:
                    return False
            return False

        hits += int(any(evidence_is_relevant(item) for item in evidence))
        selected_k = int(decision["top_k"])
        top_k_total += selected_k
        packed_total += len(evidence)
        chars_total += sum(len(item.text) for item in evidence)
        graph_total += sum("graph_expansion" in item.provenance for item in evidence)
        key = str(selected_k)
        top_k_distribution[key] = top_k_distribution.get(key, 0) + 1
    evaluated = len(records)
    report: dict[str, object] = {
        "evaluated": evaluated,
        "evidence_recall": hits / max(1, evaluated),
        "average_dynamic_top_k": top_k_total / max(1, evaluated),
        "top_k_distribution": dict(sorted(top_k_distribution.items(), key=lambda item: int(item[0]))),
        "average_packed_evidence": packed_total / max(1, evaluated),
        "average_context_chars": chars_total / max(1, evaluated),
        "average_graph_evidence": graph_total / max(1, evaluated),
        "settings": {
            "dynamic_k": config.complexity.enabled,
            "confidence_enabled": config.complexity.confidence_enabled,
            "parent_enabled": config.expansion.parent_enabled,
            "graph_enabled": config.expansion.graph_enabled,
        },
        "candidate_cache": str(cache_path),
        "diagnostic_only": True,
        "training_use_prohibited": True,
    }
    destination = (
        Path(output_path)
        if output_path
        else artifact_dir(config) / "diagnostic_evidence_metrics.json"
    )
    write_json(destination, report)
    return report


def tune_dynamic_k_rules(
    config: PipelineConfig,
    cache_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Grid-search deterministic Dynamic-K rules as a diagnostic-only ablation."""

    records = list(read_jsonl(cache_path))
    from ..retrieval.store import CorpusStore

    store = CorpusStore(artifact_dir(config) / "corpus.sqlite")
    prepared = []
    for record in records:
        gold_chunks = set(record["matched_chunk_ids"])
        gold_parents = set(record["matched_parent_ids"])
        scores = [float(value["score"]) for value in record["candidates"]]
        relevant = []
        for value in record["candidates"]:
            chunk_id = value["chunk_id"]
            chunk = store.get_chunk(chunk_id)
            relevant.append(chunk_id in gold_chunks or chunk.parent_id in gold_parents)
        prepared.append((record["question"], scores, relevant))
    store.close()

    fixed_k = 5
    baseline_hits = sum(any(relevant[:fixed_k]) for _question, _scores, relevant in prepared)
    baseline_recall = baseline_hits / max(1, len(prepared))
    trials = []
    for low, high, confident_gap, ambiguous_gap, minimum_k in product(
        (1.5, 2.0, 2.2, 2.5),
        (3.5, 4.0, 5.0),
        (0.20, 0.25, 0.30, 0.35),
        (0.03, 0.05),
        (2, 3),
    ):
        if low >= high:
            continue
        rules = deepcopy(config.complexity)
        rules.enabled = True
        rules.confidence_enabled = True
        rules.low_threshold = low
        rules.high_threshold = high
        rules.confident_gap = confident_gap
        rules.ambiguous_gap = ambiguous_gap
        rules.confident_min_k = minimum_k
        hits = 0
        selected_total = 0
        distribution: dict[str, int] = {}
        for question, scores, relevant in prepared:
            complexity = analyze_query_complexity(question, rules)
            confidence = analyze_retrieval_confidence(scores)
            selected = choose_dynamic_top_k(complexity, confidence, rules)
            hits += int(any(relevant[:selected]))
            selected_total += selected
            key = str(selected)
            distribution[key] = distribution.get(key, 0) + 1
        trials.append(
            {
                "recall": hits / max(1, len(prepared)),
                "average_k": selected_total / max(1, len(prepared)),
                "distribution": dict(sorted(distribution.items(), key=lambda item: int(item[0]))),
                "low_threshold": low,
                "high_threshold": high,
                "confident_gap": confident_gap,
                "ambiguous_gap": ambiguous_gap,
                "confident_min_k": minimum_k,
            }
        )
    trials.sort(key=lambda value: (-value["recall"], value["average_k"]))
    by_efficiency = sorted(trials, key=lambda value: (value["average_k"], -value["recall"]))
    pareto = []
    best_recall = -1.0
    for value in by_efficiency:
        if value["recall"] > best_recall:
            pareto.append(value)
            best_recall = value["recall"]
    eligible = [value for value in trials if value["recall"] >= baseline_recall]
    recommended = min(eligible, key=lambda value: (value["average_k"], -value["recall"])) if eligible else trials[0]
    report: dict[str, object] = {
        "evaluated": len(prepared),
        "fixed_k_baseline": {"k": fixed_k, "recall": baseline_recall},
        "trial_count": len(trials),
        "recommended": recommended,
        "pareto_frontier": pareto,
        "top_by_recall": trials[:20],
        "candidate_cache": str(cache_path),
        "diagnostic_only": True,
        "training_use_prohibited": True,
    }
    destination = (
        Path(output_path)
        if output_path
        else artifact_dir(config) / "dynamic_k_tuning.json"
    )
    write_json(destination, report)
    return report


def evaluate_cached_qa(
    config: PipelineConfig,
    cache_path: str | Path,
    *,
    output_path: str | Path,
    predictions_path: str | Path | None = None,
    max_examples: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Evaluate generation on diagnostic questions without rerunning retrieval."""

    records = list(read_jsonl(cache_path))
    if max_examples is not None:
        records = records[:max_examples]
    references = {
        value.id: value.answer or ""
        for value in load_qa(data_path(config, config.data.train_file))
    }
    assembler = EvidenceAssembler(config)
    generator = create_generator(config.models.generator, config.generation)
    notify = progress or (lambda _message: None)
    meteor_scores = []
    rouge_scores = []
    predictions: dict[str, dict[str, Any]] = {}
    prediction_words = 0
    reference_words = 0
    progress_stride = 1 if len(records) <= 10 else 10
    for index, record in enumerate(records, 1):
        candidates = [
            Candidate(
                chunk_id=value["chunk_id"],
                score=float(value["score"]),
                channel_scores=dict(value.get("channel_scores") or {}),
                channel_ranks={
                    key: int(rank) for key, rank in (value.get("channel_ranks") or {}).items()
                },
                provenance=tuple(value.get("provenance") or ()),
            )
            for value in record["candidates"]
        ]
        evidence, decision = assembler.assemble(record["question"], candidates)
        prediction = generator.generate(record["question"], evidence)
        reference = references.get(str(record["id"]), "")
        meteor_scores.append(meteor_exact(prediction, reference))
        rouge_scores.append(rouge_l_f1(prediction, reference))
        prediction_words += len(prediction.split())
        reference_words += len(reference.split())
        predictions[str(record["id"])] = {
            "question": record["question"],
            "answer": prediction,
            "reference": reference,
            "decision": decision,
            "evidence_ids": [value.id for value in evidence],
        }
        if index % progress_stride == 0 or index == len(records):
            notify(f"Generated diagnostic answers {index}/{len(records)}")
    evaluated = len(records)
    report: dict[str, object] = {
        "evaluated": evaluated,
        "meteor_exact": sum(meteor_scores) / max(1, evaluated),
        "rouge_l_f1": sum(rouge_scores) / max(1, evaluated),
        "average_prediction_words": prediction_words / max(1, evaluated),
        "average_reference_words": reference_words / max(1, evaluated),
        "generator": {
            "backend": config.models.generator.backend,
            "model": config.models.generator.name_or_path,
            "parameters_b": config.models.generator.parameters_b,
        },
        "evidence_settings": {
            "dynamic_k": config.complexity.enabled,
            "parent_enabled": config.expansion.parent_enabled,
            "graph_enabled": config.expansion.graph_enabled,
        },
        "candidate_cache": str(cache_path),
        "diagnostic_only": True,
        "training_use_prohibited": True,
        "metric_note": "Local exact-token METEOR proxy; verify against organizer scorer.",
    }
    write_json(output_path, report)
    if predictions_path:
        write_json(predictions_path, predictions)
    return report


def evaluate_qa_files(predictions_path: str | Path, references_path: str | Path) -> dict[str, object]:
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8-sig"))
    references = json.loads(Path(references_path).read_text(encoding="utf-8-sig"))
    meteor_scores = []
    rouge_scores = []
    missing = []
    for example_id, reference_value in references.items():
        reference = reference_value.get("answer")
        if reference is None:
            continue
        prediction_value = predictions.get(example_id)
        if prediction_value is None:
            missing.append(example_id)
            prediction = ""
        else:
            prediction = prediction_value.get("answer") or ""
        meteor_scores.append(meteor_exact(prediction, reference))
        rouge_scores.append(rouge_l_f1(prediction, reference))
    return {
        "examples": len(meteor_scores),
        "meteor_exact": sum(meteor_scores) / max(1, len(meteor_scores)),
        "rouge_l_f1": sum(rouge_scores) / max(1, len(rouge_scores)),
        "missing_prediction_count": len(missing),
        "note": "METEOR is exact-token and language-neutral; verify against the organizer scorer before final reporting.",
    }
