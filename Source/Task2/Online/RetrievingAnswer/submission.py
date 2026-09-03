"""Validate and package deterministic Task 2 answers."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from .batch import load_questions


def validate_predictions(
    questions_path: Path, predictions_path: Path, *, expected_split: str
) -> dict[str, Any]:
    """Validate exact IDs and non-empty answer-only prediction records."""
    questions = load_questions(questions_path, expected_split=expected_split)
    payload = _load_json_object(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Predictions must be a JSON object.")
    if any(not _valid_answer_record(value) for value in payload.values()):
        raise ValueError("Predictions contain an invalid answer record.")
    expected_ids = {item.question_id for item in questions}
    predicted_ids = {str(key) for key in payload}
    invalid = sorted(
        question_id
        for question_id in expected_ids & predicted_ids
        if not _valid_answer_record(payload[question_id])
    )
    missing = sorted(expected_ids - predicted_ids)
    extra = sorted(predicted_ids - expected_ids)
    return {
        "status": "PASS" if not (invalid or missing or extra) else "FAIL",
        "expected": len(expected_ids),
        "predicted": len(predicted_ids),
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
        "predictions_sha256": _sha256(predictions_path),
    }


def package_submission(predictions_path: Path, zip_path: Path, *, overwrite: bool = False) -> str:
    """Write a deterministic ZIP containing only UTF-8 submission.json."""
    if zip_path.exists() and not overwrite:
        raise FileExistsError(f"Submission ZIP already exists: {zip_path}")
    payload = _load_json_object(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Predictions must be a JSON object.")
    content = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("submission.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(info, content)
    return _sha256(zip_path)


def build_submission(
    questions_path: Path,
    predictions_path: Path,
    zip_path: Path,
    *,
    expected_split: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate exact phase IDs, package, and replay-validate the final ZIP."""
    prediction_report = validate_predictions(
        questions_path,
        predictions_path,
        expected_split=expected_split,
    )
    if prediction_report["status"] != "PASS":
        raise ValueError("Predictions do not satisfy the phase ID/answer contract.")
    package_sha256 = package_submission(predictions_path, zip_path, overwrite=overwrite)
    archive_report = validate_submission_zip(
        zip_path,
        expected_ids={
            item.question_id
            for item in load_questions(questions_path, expected_split=expected_split)
        },
    )
    if archive_report["status"] != "PASS":
        raise ValueError("Generated submission ZIP failed replay validation.")
    return {
        "status": "PASS",
        "phase": expected_split,
        "predictions": prediction_report,
        "archive": archive_report,
        "submission_sha256": package_sha256,
    }


def validate_submission_zip(zip_path: Path, *, expected_ids: set[str]) -> dict[str, Any]:
    """Require one safe UTF-8 submission.json with exact answer-only records."""
    errors: list[str] = []
    if not zip_path.is_file():
        return {"status": "FAIL", "errors": ["Submission ZIP is missing."], "zip_sha256": None}
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if names != ["submission.json"]:
                errors.append("ZIP must contain exactly submission.json at archive root.")
                payload: object = {}
            else:
                info = archive.getinfo("submission.json")
                if info.is_dir() or info.file_size <= 0:
                    errors.append("submission.json must be a non-empty regular member.")
                raw = archive.read("submission.json")
                if raw.startswith(b"\xef\xbb\xbf"):
                    errors.append("submission.json must be UTF-8 without BOM.")
                payload = _load_json_object(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        return {"status": "FAIL", "errors": [str(exc)], "zip_sha256": _sha256(zip_path)}
    if not isinstance(payload, dict):
        errors.append("submission.json must be a JSON object.")
        payload = {}
    actual_ids = {str(key) for key in payload}
    if actual_ids != expected_ids:
        errors.append(
            f"Submission IDs mismatch: missing={sorted(expected_ids - actual_ids)[:5]}, "
            f"extra={sorted(actual_ids - expected_ids)[:5]}."
        )
    if any(not _valid_answer_record(value) for value in payload.values()):
        errors.append("Every submission record must contain only a non-empty string answer.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "records": len(payload),
        "members": ["submission.json"] if not errors else names,
        "zip_sha256": _sha256(zip_path),
    }


def _valid_answer_record(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"answer"}
        and isinstance(value.get("answer"), str)
        and bool(value["answer"].strip())
    )


def _load_json_object(text: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
