"""Create provisional metric, slice, paired, and error artifacts for Task 2."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from legal_qa.metrics import meteor_score, rouge_l_score, tokenize

LEGAL_MARKER = re.compile(r"\b(?:điều|khoản|nghị\s*định|thông\s*tư|luật)\b", re.IGNORECASE)
NUMBER = re.compile(r"\b\d+(?:[.,/]\d+)*\b")


def evaluate_run(
    release_root: Path,
    run_root: Path,
    *,
    parent_predictions: Path | None = None,
) -> dict[str, Any]:
    """Score validation predictions and emit observability artifacts."""
    references = {
        str(row["question_id"]): row
        for row in _read_jsonl(release_root / "qa" / "validation.jsonl")
    }
    predictions = _prediction_map(run_root / "validation_predictions.json")
    missing = sorted(set(references) - set(predictions))
    extra = sorted(set(predictions) - set(references))
    if missing or extra:
        raise ValueError(
            f"Validation prediction IDs mismatch; missing={missing[:5]}, extra={extra[:5]}."
        )
    traces = _trace_map(run_root / "validation_trace.json")
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    slice_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for question_id in sorted(references):
        reference = str(references[question_id]["answer_model"])
        prediction = predictions[question_id]
        meteor = meteor_score(prediction, reference)
        rouge = rouge_l_score(prediction, reference)
        labels = _error_labels(prediction, reference)
        slices = _slice_labels(str(references[question_id]["question_model"]), reference)
        for label in slices:
            slice_values[label].append((meteor, rouge))
        case = {
            "question_id": question_id,
            "meteor": meteor,
            "rouge_l": rouge,
            "prediction_tokens": len(tokenize(prediction)),
            "reference_tokens": len(tokenize(reference)),
            "length_ratio": len(tokenize(prediction)) / max(1, len(tokenize(reference))),
            "slices": slices,
            "error_labels": labels,
        }
        cases.append(case)
        if labels:
            errors.append(
                {
                    **case,
                    "question": str(references[question_id]["question_model"]),
                    "reference": reference,
                    "prediction": prediction,
                    "pipeline_trace": traces.get(question_id, {}),
                    "error_label_status": "HEURISTIC_REVIEW_REQUIRED",
                }
            )
    metric_report = {
        "schema_version": "task2-metrics-v1",
        "task_id": "Task2",
        "status": "PASS",
        "metric_status": "PROVISIONAL_UNTIL_ORGANIZER_SCORER_PARITY",
        "evaluated_pairs": len(cases),
        "meteor": _mean(float(case["meteor"]) for case in cases),
        "rouge_l": _mean(float(case["rouge_l"]) for case in cases),
        "empty_rate": sum(not predictions[key].strip() for key in predictions) / max(1, len(cases)),
        "error_counts": dict(Counter(label for case in cases for label in case["error_labels"])),
    }
    slice_report = {
        label: {
            "count": len(values),
            "meteor": _mean(value[0] for value in values),
            "rouge_l": _mean(value[1] for value in values),
        }
        for label, values in sorted(slice_values.items())
    }
    _write_json(run_root / "metrics.json", metric_report)
    _write_json(run_root / "slice_metrics.json", slice_report)
    _write_jsonl(run_root / "case_metrics.jsonl", cases)
    _write_jsonl(run_root / "error_table.jsonl", errors)
    if parent_predictions is not None and parent_predictions.is_file():
        _write_json(
            run_root / "paired_comparison.json",
            _paired_report(cases, _prediction_map(parent_predictions), references),
        )
    return metric_report


def _paired_report(
    candidate_cases: Iterable[Mapping[str, Any]],
    parent_predictions: Mapping[str, str],
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    slice_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_cases:
        question_id = str(candidate["question_id"])
        if question_id not in parent_predictions:
            continue
        reference = str(references[question_id]["answer_model"])
        parent_meteor = meteor_score(parent_predictions[question_id], reference)
        parent_rouge = rouge_l_score(parent_predictions[question_id], reference)
        row = {
            "question_id": question_id,
            "meteor_delta": float(candidate["meteor"]) - parent_meteor,
            "rouge_l_delta": float(candidate["rouge_l"]) - parent_rouge,
        }
        rows.append(row)
        for label in candidate.get("slices", []):
            slice_rows[str(label)].append(row)
    ordered_gain = sorted(rows, key=lambda row: (-float(row["meteor_delta"]), row["question_id"]))
    ordered_loss = sorted(rows, key=lambda row: (float(row["meteor_delta"]), row["question_id"]))
    return {
        "schema_version": "task2-paired-comparison-v1",
        "status": "DIAGNOSTIC",
        "paired_cases": len(rows),
        "meteor_delta": _mean(float(row["meteor_delta"]) for row in rows),
        "rouge_l_delta": _mean(float(row["rouge_l_delta"]) for row in rows),
        "candidate_wins": sum(float(row["meteor_delta"]) > 0 for row in rows),
        "candidate_losses": sum(float(row["meteor_delta"]) < 0 for row in rows),
        "candidate_ties": sum(float(row["meteor_delta"]) == 0 for row in rows),
        "slice_deltas": {
            label: {
                "count": len(values),
                "meteor_delta": _mean(float(row["meteor_delta"]) for row in values),
                "rouge_l_delta": _mean(float(row["rouge_l_delta"]) for row in values),
            }
            for label, values in sorted(slice_rows.items())
        },
        "best_gains": ordered_gain[:20],
        "worst_regressions": ordered_loss[:20],
        "counterexamples": ordered_loss[:20],
        "decision": "REVIEW_REQUIRED",
    }


def _error_labels(prediction: str, reference: str) -> list[str]:
    labels: list[str] = []
    predicted_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    ratio = len(predicted_tokens) / max(1, len(reference_tokens))
    if not prediction.strip():
        labels.append("EMPTY_ANSWER")
    if ratio < 0.55:
        labels.append("LIKELY_MISSING_CONTENT_OR_TRUNCATION")
    if ratio > 1.8:
        labels.append("EXCESSIVE_VERBOSITY")
    if _repetition_rate(predicted_tokens) > 0.35:
        labels.append("TOKEN_REPETITION")
    if LEGAL_MARKER.search(reference) and not LEGAL_MARKER.search(prediction):
        labels.append("MISSING_LEGAL_STRUCTURE")
    reference_numbers = set(NUMBER.findall(reference))
    prediction_numbers = set(NUMBER.findall(prediction))
    if reference_numbers - prediction_numbers:
        labels.append("MISSING_REFERENCE_NUMBER")
    if prediction_numbers - reference_numbers:
        labels.append("EXTRA_OR_WRONG_NUMBER")
    return labels


def _slice_labels(question: str, reference: str) -> list[str]:
    tokens = len(tokenize(reference))
    if tokens < 200:
        length = "answer_short"
    elif tokens < 500:
        length = "answer_medium"
    else:
        length = "answer_long"
    labels = ["overall", length]
    labels.append("legal_structure_yes" if LEGAL_MARKER.search(reference) else "legal_structure_no")
    labels.append("list_yes" if re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", reference) else "list_no")
    labels.append("multiline_yes" if "\n" in reference else "multiline_no")
    labels.append(_question_type(question))
    return labels


def _question_type(question: str) -> str:
    lowered = question.casefold()
    rules = (
        ("penalty", ("xử phạt", "mức phạt", "phạt bao nhiêu")),
        ("condition", ("điều kiện", "yêu cầu gì")),
        ("procedure", ("thủ tục", "hồ sơ", "trình tự")),
        ("authority", ("thẩm quyền", "cơ quan nào", "ai có quyền")),
        ("definition", ("là gì", "được hiểu")),
    )
    for label, terms in rules:
        if any(term in lowered for term in terms):
            return f"question_{label}"
    return "question_other"


def _repetition_rate(tokens: list[str]) -> float:
    size = 4
    if len(tokens) < size:
        return 0.0
    ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    counts = Counter(ngrams)
    return sum(count - 1 for count in counts.values()) / len(ngrams)


def _prediction_map(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Prediction file must be an object: {path}")
    result: dict[str, str] = {}
    for question_id, record in value.items():
        if not isinstance(record, dict) or set(record) != {"answer"}:
            raise ValueError(f"Prediction {question_id} must contain only answer.")
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Prediction {question_id} has an empty answer.")
        result[str(question_id)] = answer
    return result


def _trace_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Trace file must be an object: {path}")
    return {
        str(question_id): dict(trace)
        for question_id, trace in value.items()
        if isinstance(trace, dict)
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            yield value


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
