"""Run deterministic answer generation over canonical Task 2 question JSONL."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    """Represent one canonical Public, Private, or validation question."""

    question_id: str
    question: str


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Carry one generated answer and an internal provenance trace."""

    answer: str
    trace: Mapping[str, Any] = field(default_factory=dict)


class AnswerEngine(Protocol):
    """Define the minimal retriever-generator boundary for batch inference."""

    def answer(self, question: str) -> AnswerResult:
        """Generate one non-empty answer for a question."""


class BatchedAnswerEngine(AnswerEngine, Protocol):
    """Optionally expose bounded multi-question generation."""

    inference_batch_size: int

    def answer_many(self, questions: Sequence[str]) -> Sequence[AnswerResult]:
        """Generate one answer per input question in stable order."""


def load_questions(path: Path, *, expected_split: str) -> tuple[QuestionRecord, ...]:
    """Load canonical question JSONL and reject duplicate IDs or visible answers."""
    records: list[QuestionRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(_question_from_payload(payload, line_number, expected_split, seen))
    if not records:
        raise ValueError("Question JSONL must contain at least one record.")
    return tuple(records)


def _question_from_payload(
    payload: object, line_number: int, expected_split: str, seen: set[str]
) -> QuestionRecord:
    if not isinstance(payload, dict):
        raise ValueError(f"Question line {line_number} must be an object.")
    if payload.get("source_task") != "Task2" or payload.get("split") != expected_split:
        raise ValueError(f"Question line {line_number} has wrong task or split.")
    if expected_split in {"public", "private"} and any(
        payload.get(key) is not None for key in ("answer", "answer_raw", "answer_model")
    ):
        raise ValueError(f"Question line {line_number} exposes an answer.")
    question_id = str(payload.get("question_id", ""))
    question = str(payload.get("question_model", ""))
    if not question_id or question_id in seen or not question.strip():
        raise ValueError(f"Question line {line_number} has invalid or duplicate content.")
    seen.add(question_id)
    return QuestionRecord(question_id=question_id, question=question)


class BatchInferenceRunner:
    """Generate resumable Task 2 predictions through an injected local engine."""

    def __init__(self, engine: AnswerEngine) -> None:
        """Bind a preloaded local answer engine."""
        self.engine = engine

    def __repr__(self) -> str:
        """Show the bound engine without exposing model or data content."""
        return f"BatchInferenceRunner(engine={type(self.engine).__name__})"

    def run(
        self,
        questions_path: Path,
        predictions_path: Path,
        *,
        expected_split: str,
        trace_path: Path | None = None,
        resume: bool = False,
        resume_trace_requirements: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate all answers and atomically checkpoint predictions and traces."""
        questions = load_questions(questions_path, expected_split=expected_split)
        target_ids = {item.question_id for item in questions}
        predictions = _resume_predictions(predictions_path, target_ids) if resume else {}
        traces = _resume_predictions(trace_path, target_ids) if resume and trace_path else {}
        if resume_trace_requirements:
            _validate_resume_trace(predictions, traces, resume_trace_requirements)
        pending: list[QuestionRecord] = []
        for item in questions:
            trace_complete = trace_path is None or item.question_id in traces
            if _complete_prediction(predictions.get(item.question_id)) and trace_complete:
                continue
            pending.append(item)

        generated = 0
        answer_many = getattr(self.engine, "answer_many", None)
        batch_size = max(1, int(getattr(self.engine, "inference_batch_size", 1)))
        for offset in range(0, len(pending), batch_size):
            items = pending[offset : offset + batch_size]
            if callable(answer_many):
                results = tuple(answer_many([item.question for item in items]))
            else:
                results = tuple(self.engine.answer(item.question) for item in items)
            if len(results) != len(items):
                raise ValueError("Batched engine returned a mismatched result count.")
            for item, result in zip(items, results, strict=True):
                if not result.answer.strip():
                    raise ValueError(f"Engine returned a blank answer for {item.question_id}.")
                predictions[item.question_id] = {"answer": result.answer}
                traces[item.question_id] = dict(result.trace)
                generated += 1
            _write_json_atomic(predictions_path, predictions)
            if trace_path is not None:
                _write_json_atomic(trace_path, traces)
        return {"questions": len(questions), "generated": generated, "complete": True}


def _read_object(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resume_predictions(path: Path, target_ids: set[str]) -> dict[str, Any]:
    return {key: value for key, value in _read_object(path).items() if key in target_ids}


def _complete_prediction(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("answer"), str)
        and bool(value["answer"].strip())
    )


def _validate_resume_trace(
    predictions: Mapping[str, Any],
    traces: Mapping[str, Any],
    requirements: Mapping[str, Any],
) -> None:
    """Reject mixing completed predictions from a different runtime configuration."""
    for question_id in predictions.keys() & traces.keys():
        trace = traces[question_id]
        if not isinstance(trace, dict):
            raise ValueError(f"Resume trace {question_id} must be an object.")
        mismatched = [key for key, value in requirements.items() if trace.get(key) != value]
        if mismatched:
            raise ValueError(
                f"Resume trace configuration mismatch for {question_id}: {mismatched}."
            )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        temporary.unlink(missing_ok=True)
        raise
