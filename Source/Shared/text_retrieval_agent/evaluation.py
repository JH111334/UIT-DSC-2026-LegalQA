"""Evaluate retrieval, abstention, and citation integrity."""

import json
from pathlib import Path
from typing import Any

from text_retrieval_agent.agent import EvidenceAgent
from text_retrieval_agent.contracts import RetrievalQuery
from text_retrieval_agent.pipeline import RetrievalPipeline


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(
    pipeline: RetrievalPipeline,
    queries_path: Path,
    qrels_path: Path,
    *,
    top_k: int,
) -> dict[str, float | int]:
    """Compute retrieval, abstention, and citation-integrity metrics."""
    agent = EvidenceAgent(pipeline)
    qrels: dict[str, set[str]] = {}
    for row in _read_jsonl(qrels_path):
        qrels.setdefault(str(row["query_id"]), set()).add(str(row["document_id"]))
    reciprocal_ranks: list[float] = []
    recalled = 0
    abstention_correct = 0
    citations_valid = 0
    query_rows = _read_jsonl(queries_path)
    for row in query_rows:
        query = RetrievalQuery(
            query_id=str(row["query_id"]),
            text=str(row["text"]),
            allowed_scopes=tuple(str(scope) for scope in row.get("allowed_scopes", ["public"])),
            top_k=top_k,
        )
        response = pipeline.search(query)
        answer = agent.ask(query)
        answerable = bool(row["answerable"])
        relevant = qrels.get(query.query_id, set())
        ranks = [result.rank for result in response.results if result.document_id in relevant]
        if answerable:
            recalled += int(bool(ranks))
            reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        abstention_correct += int(answer.abstained is not answerable)
        result_ids = {result.chunk_id for result in response.results}
        citations_valid += int(
            all(citation.chunk_id in result_ids for citation in answer.citations)
        )
    answerable_count = len(reciprocal_ranks)
    count = len(query_rows)
    return {
        "query_count": count,
        "answerable_query_count": answerable_count,
        f"recall_at_{top_k}": recalled / answerable_count if answerable_count else 0.0,
        "mrr": sum(reciprocal_ranks) / answerable_count if answerable_count else 0.0,
        "abstention_accuracy": abstention_correct / count if count else 0.0,
        "citation_integrity": citations_valid / count if count else 0.0,
    }
