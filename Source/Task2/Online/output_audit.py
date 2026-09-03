"""Audit unlabeled Task 2 outputs without turning heuristics into quality claims."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from legal_qa.metrics import tokenize

LEGAL_MARKER = re.compile(
    r"\b(?:điều|khoản|nghị\s*định|thông\s*tư|luật)\b",
    re.IGNORECASE,
)
LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")


def audit_unlabeled_output(
    source: Path,
    report_path: Path,
    case_path: Path,
    *,
    expected_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Write aggregate and ID-only diagnostics for a submission or prediction JSON."""
    payload = _load_payload(source)
    if expected_ids is not None and set(payload) != expected_ids:
        missing = sorted(expected_ids - set(payload))
        extra = sorted(set(payload) - expected_ids)
        raise ValueError(f"Output IDs mismatch; missing={missing[:5]}, extra={extra[:5]}.")

    cases: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    for question_id in sorted(payload):
        answer = payload[question_id]
        tokens = tokenize(answer)
        repetition_rate = _ngram_repetition_rate(tokens)
        flags: list[str] = []
        if len(tokens) < 20:
            flags.append("VERY_SHORT_REVIEW")
        if repetition_rate > 0.10:
            flags.append("REPEATED_4GRAM_REVIEW")
        if not LEGAL_MARKER.search(answer):
            flags.append("LEGAL_MARKER_ABSENT")
        if not answer.rstrip().endswith((".", "!", "?", ":", ";", ")", "]")):
            flags.append("ENDING_REVIEW")
        flag_counts.update(flags)
        cases.append(
            {
                "question_id": question_id,
                "character_count": len(answer),
                "word_token_count": len(tokens),
                "newline_count": answer.count("\n"),
                "has_legal_marker": bool(LEGAL_MARKER.search(answer)),
                "has_list_marker": bool(LIST_MARKER.search(answer)),
                "repeated_4gram_rate": repetition_rate,
                "flags": flags,
                "review_status": "HEURISTIC_REVIEW_REQUIRED" if flags else "NO_FLAG",
            }
        )

    char_lengths = [int(row["character_count"]) for row in cases]
    token_lengths = [int(row["word_token_count"]) for row in cases]
    report = {
        "schema_version": "task2-unlabeled-output-audit-v1",
        "task_id": "Task2",
        "status": "PASS",
        "evidence_class": "measured_local_behavior_only",
        "gold_status": "absent",
        "metric_status": "UNAVAILABLE_WITHOUT_REFERENCE_ANSWERS",
        "quality_claim_allowed": False,
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "records": len(cases),
        "empty_answers": sum(not answer.strip() for answer in payload.values()),
        "duplicate_answers": len(payload) - len(set(payload.values())),
        "multiline_answers": sum(int(row["newline_count"]) > 0 for row in cases),
        "answers_with_legal_marker": sum(bool(row["has_legal_marker"]) for row in cases),
        "answers_with_list_marker": sum(bool(row["has_list_marker"]) for row in cases),
        "character_count": _distribution(char_lengths),
        "word_token_count": _distribution(token_lengths),
        "heuristic_flag_counts": dict(sorted(flag_counts.items())),
        "case_artifact": str(case_path),
        "limitations": [
            "Public has no reference answers, so METEOR, ROUGE-L and correctness are unavailable.",
            "Flags are review cues, not legal or quality judgments.",
            "Generation-cap truncation requires per-sample model trace "
            "and cannot be inferred here.",
        ],
    }
    _write_json(report_path, report)
    _write_jsonl(case_path, cases)
    return report


def _load_payload(path: Path) -> dict[str, str]:
    if path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["submission.json"]:
                raise ValueError("Submission ZIP must contain exactly submission.json.")
            raw = archive.read("submission.json")
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("submission.json must be UTF-8 without BOM.")
        value = json.loads(raw.decode("utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Output payload must be a JSON object.")
    result: dict[str, str] = {}
    for question_id, record in value.items():
        if not isinstance(record, dict) or set(record) != {"answer"}:
            raise ValueError(f"Output {question_id} must contain only answer.")
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Output {question_id} has an empty answer.")
        result[str(question_id)] = answer
    return result


def _ngram_repetition_rate(tokens: list[str], size: int = 4) -> float:
    if len(tokens) < size:
        return 0.0
    ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values())
    return repeated / len(ngrams)


def _distribution(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {key: 0 for key in ("min", "p50", "p95", "p99", "max")}
    return {
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: list[int], fraction: float) -> int:
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
