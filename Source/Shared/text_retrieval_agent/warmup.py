"""Validate local UIT DSC warm-up files without exposing their content."""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from text_retrieval_agent.contracts import ContractError


@dataclass(frozen=True, slots=True)
class Task1WarmupRecord:
    """Represent one LegalIR question and its relevant document IDs."""

    query_id: str
    question: str
    answer_document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Task2WarmupRecord:
    """Represent one LegalQA question and reference answer."""

    query_id: str
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class WarmupSummary:
    """Expose non-sensitive shape and integrity evidence for one local file."""

    task: str
    records: int
    unique_questions: int
    empty_answers: int
    sha256: str


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"Cannot load warm-up JSON: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ContractError("Warm-up JSON must be a non-empty object keyed by query ID.")
    return payload


def load_task1_warmup(path: Path) -> tuple[Task1WarmupRecord, ...]:
    """Load Task 1 labels while preserving organizer query and document IDs."""
    records: list[Task1WarmupRecord] = []
    for query_id, raw in _load_mapping(path).items():
        if not isinstance(raw, dict):
            raise ContractError(f"Task 1 record {query_id!r} must be an object.")
        question = raw.get("question")
        answer = raw.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ContractError(f"Task 1 record {query_id!r} has no question.")
        if not isinstance(answer, list) or not answer:
            raise ContractError(f"Task 1 record {query_id!r} needs document labels.")
        document_ids = tuple(str(item) for item in answer)
        if any(not item.strip() or any(char.isspace() for char in item) for item in document_ids):
            raise ContractError(f"Task 1 record {query_id!r} has an invalid document ID.")
        if len(set(document_ids)) != len(document_ids):
            raise ContractError(f"Task 1 record {query_id!r} repeats a document ID.")
        records.append(Task1WarmupRecord(str(query_id), question.strip(), document_ids))
    return tuple(records)


def load_task2_warmup(path: Path) -> tuple[Task2WarmupRecord, ...]:
    """Load Task 2 reference answers without interpreting them as current law."""
    records: list[Task2WarmupRecord] = []
    for query_id, raw in _load_mapping(path).items():
        if not isinstance(raw, dict):
            raise ContractError(f"Task 2 record {query_id!r} must be an object.")
        question = raw.get("question")
        answer = raw.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ContractError(f"Task 2 record {query_id!r} has no question.")
        if not isinstance(answer, str) or not answer.strip():
            raise ContractError(f"Task 2 record {query_id!r} has no reference answer.")
        records.append(Task2WarmupRecord(str(query_id), question.strip(), answer.strip()))
    return tuple(records)


def summarize_warmup(path: Path, task: str) -> WarmupSummary:
    """Validate a local warm-up file and return reproducible aggregate evidence."""
    if task == "Task1":
        task1_records = load_task1_warmup(path)
        return WarmupSummary(
            task=task,
            records=len(task1_records),
            unique_questions=len({record.question for record in task1_records}),
            empty_answers=sum(not record.answer_document_ids for record in task1_records),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    if task == "Task2":
        task2_records = load_task2_warmup(path)
        return WarmupSummary(
            task=task,
            records=len(task2_records),
            unique_questions=len({record.question for record in task2_records}),
            empty_answers=sum(not record.answer for record in task2_records),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    raise ValueError("task must be Task1 or Task2")


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one local warm-up file and print an aggregate JSON summary."""
    parser = argparse.ArgumentParser(prog="egta-warmup")
    parser.add_argument("task", choices=("Task1", "Task2"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args(argv)
    print(json.dumps(asdict(summarize_warmup(arguments.path, arguments.task)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
