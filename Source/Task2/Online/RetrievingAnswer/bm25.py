"""Build and query a deterministic Task2-only SQLite FTS5 BM25 index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUERY_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """Carry one ranked legal chunk and its parent context."""

    chunk_id: str
    doc_id: str
    parent_chunk_id: str
    score: float
    retrieval_text: str
    parent_text: str
    doc_number: str
    article: str


class BM25Index:
    """Open a read-only FTS5 index and return stable BM25 rankings."""

    def __init__(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"BM25 index is missing: {resolved}")
        self.path = resolved

    def search(self, query: str, *, limit: int = 20) -> list[BM25Hit]:
        """Search lexical candidates using an OR query and deterministic ties."""
        if limit <= 0:
            raise ValueError("BM25 limit must be positive.")
        expression = _match_expression(query)
        if not expression:
            return []
        uri = f"file:{self.path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, doc_id, parent_chunk_id, bm25(chunks) AS rank,
                       retrieval_text, parent_text, doc_number, article
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY rank ASC, chunk_id ASC
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        return [
            BM25Hit(
                chunk_id=str(row[0]),
                doc_id=str(row[1]),
                parent_chunk_id=str(row[2]),
                score=-float(row[3]),
                retrieval_text=str(row[4]),
                parent_text=str(row[5]),
                doc_number=str(row[6]),
                article=str(row[7]),
            )
            for row in rows
        ]


def build_bm25_index(release_root: Path, run_root: Path) -> dict[str, Any]:
    """Build the canonical E1 index atomically outside the immutable release."""
    chunks_path = release_root / "corpus" / "chunks.jsonl"
    corpus_manifest = release_root / "corpus" / "corpus_manifest.json"
    if not chunks_path.is_file() or not corpus_manifest.is_file():
        raise FileNotFoundError("Canonical corpus chunks/manifest are required for E1.")
    index_root = run_root / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    target = index_root / "bm25.sqlite3"
    manifest_path = index_root / "index_manifest.json"
    if target.exists() or manifest_path.exists():
        raise FileExistsError(
            "BM25 index artifacts already exist; use a new run root to preserve replayability."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=index_root,
        prefix=".bm25.",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    indexed = 0
    try:
        connection = sqlite3.connect(temporary)
        try:
            _configure_builder(connection)
            _create_schema(connection)
            batch: list[tuple[str, ...]] = []
            for row in _read_jsonl(chunks_path):
                if row.get("source_task") != "Task2" or row.get("indexable") is not True:
                    continue
                batch.append(_index_row(row))
                if len(batch) >= 1000:
                    connection.executemany(
                        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    indexed += len(batch)
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                indexed += len(batch)
            optimize = connection.execute("INSERT INTO chunks(chunks) VALUES('optimize')")
            optimize.close()
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, target)
    except (OSError, sqlite3.Error, ValueError, json.JSONDecodeError):
        temporary.unlink(missing_ok=True)
        raise
    manifest = {
        "schema_version": "task2-index-manifest-v1",
        "task_id": "Task2",
        "status": "READY",
        "index_id": "task2-e1-bm25-v1",
        "index_type": "BM25",
        "corpus_manifest_sha256": _sha256(corpus_manifest),
        "chunks_sha256": _sha256(chunks_path),
        "config": {
            "engine": "sqlite-fts5",
            "tokenizer": "unicode61 remove_diacritics 0",
            "query": "unique_word_tokens_or",
            "score": "-sqlite_bm25",
        },
        "artifact": target.name,
        "artifact_sha256": _sha256(target),
        "indexed_chunks": indexed,
    }
    _write_json(manifest_path, manifest)
    return manifest


def evaluate_bm25(
    index: BM25Index,
    questions_path: Path,
    qrels_path: Path,
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10, 20),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate BM25 on proxy qrels, with HIGH-confidence results as primary."""
    if not cutoffs or min(cutoffs) <= 0:
        raise ValueError("Retrieval cutoffs must be positive.")
    qrels_all: dict[str, set[str]] = {}
    qrels_high: dict[str, set[str]] = {}
    for row in _read_jsonl(qrels_path):
        if row.get("qrels_status") != "diagnostic_proxy":
            continue
        question_id = str(row["question_id"])
        chunk_id = str(row["chunk_id"])
        qrels_all.setdefault(question_id, set()).add(chunk_id)
        if row.get("confidence") == "HIGH":
            qrels_high.setdefault(question_id, set()).add(chunk_id)
    questions = {
        str(row["question_id"]): str(row["question_model"])
        for row in _read_jsonl(questions_path)
        if str(row["question_id"]) in qrels_all
    }
    largest = max(cutoffs)
    ranked_by_question: dict[str, list[str]] = {}
    traces: list[dict[str, Any]] = []
    for question_id in sorted(questions):
        hits = index.search(questions[question_id], limit=largest)
        ranked = [hit.chunk_id for hit in hits]
        ranked_by_question[question_id] = ranked
        traces.append(
            {
                "question_id": question_id,
                "judged_relevant": sorted(qrels_all[question_id]),
                "judged_relevant_high": sorted(qrels_high.get(question_id, set())),
                "ranked": [
                    {"chunk_id": hit.chunk_id, "rank": rank, "bm25_score": hit.score}
                    for rank, hit in enumerate(hits, start=1)
                ],
            }
        )
    high_metrics = _ranking_metrics(ranked_by_question, qrels_high, cutoffs)
    all_metrics = _ranking_metrics(ranked_by_question, qrels_all, cutoffs)
    metrics = {
        "schema_version": "task2-retrieval-metrics-v1",
        "task_id": "Task2",
        "profile": "e1-bm25",
        "qrels_status": "diagnostic_proxy",
        "primary_proxy_scope": "HIGH_confidence",
        "judged_queries": high_metrics["judged_queries"],
        "mrr": high_metrics["mrr"],
        "recall_at_k": high_metrics["recall_at_k"],
        "ndcg_at_k": high_metrics["ndcg_at_k"],
        "proxy_scopes": {
            "HIGH_confidence": high_metrics,
            "HIGH_and_MEDIUM_confidence": all_metrics,
        },
        "promotion_allowed": False,
        "promotion_blocker": "manual citation audit and end-to-end comparison required",
    }
    return metrics, traces


def _ranking_metrics(
    ranked_by_question: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    recalls = {cutoff: 0.0 for cutoff in cutoffs}
    ndcgs = {cutoff: 0.0 for cutoff in cutoffs}
    reciprocal_ranks = 0.0
    judged = [question_id for question_id in sorted(qrels) if question_id in ranked_by_question]
    for question_id in judged:
        ranked = ranked_by_question[question_id]
        relevant = qrels[question_id]
        for cutoff in cutoffs:
            prefix = ranked[:cutoff]
            recalls[cutoff] += len(relevant & set(prefix)) / max(1, len(relevant))
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, chunk_id in enumerate(prefix, start=1)
                if chunk_id in relevant
            )
            ideal_hits = min(len(relevant), cutoff)
            ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcgs[cutoff] += dcg / ideal if ideal else 0.0
        first_rank = next(
            (rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant),
            None,
        )
        reciprocal_ranks += 0.0 if first_rank is None else 1.0 / first_rank
    count = len(judged)
    return {
        "judged_queries": count,
        "mrr": reciprocal_ranks / max(1, count),
        "recall_at_k": {str(cutoff): recalls[cutoff] / max(1, count) for cutoff in cutoffs},
        "ndcg_at_k": {str(cutoff): ndcgs[cutoff] / max(1, count) for cutoff in cutoffs},
    }


def _configure_builder(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")


def _create_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunks USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                parent_chunk_id UNINDEXED,
                retrieval_text,
                parent_text UNINDEXED,
                doc_number UNINDEXED,
                article UNINDEXED,
                source_task UNINDEXED,
                tokenize='unicode61 remove_diacritics 0'
            )
            """
        )
    except sqlite3.OperationalError as exc:
        raise RuntimeError("Python SQLite runtime does not include FTS5.") from exc


def _index_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    retrieval_text = str(row.get("retrieval_text", "")).strip()
    if not retrieval_text:
        raise ValueError(f"Indexable chunk has empty retrieval_text: {row.get('chunk_id')}")
    return (
        str(row["chunk_id"]),
        str(row["doc_id"]),
        str(row.get("parent_chunk_id") or row["chunk_id"]),
        retrieval_text,
        str(row.get("parent_text") or retrieval_text),
        str(row.get("doc_number_canonical") or ""),
        str(row.get("article") or ""),
        "Task2",
    )


def _match_expression(query: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for token in QUERY_TOKEN.findall(query.casefold()):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        terms.append(token.replace('"', '""'))
        if len(terms) >= 32:
            break
    return " OR ".join(f'"{term}"' for term in terms)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write diagnostic rows without embedding question/evidence text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
