"""Inventory Task 2 artifacts and select the next safe pipeline operation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .request import PipelineRequest
from .RetrievingAnswer.submission import validate_predictions, validate_submission_zip


def inventory_pipeline(request: PipelineRequest) -> dict[str, Any]:
    """Report completed gates and the next operation without loading ML packages."""
    control_root, run_root = _roots(request)
    release_root = request.release_root
    checks = {
        "active_operation": _active_operation(run_root / ".pipeline.lock"),
        "release": _status(release_root / "manifest.json", "READY"),
        "tokenizer": _status(control_root / "tokenizer" / "tokenizer_report_qwen.json", "PASS"),
        "checkpoint": _ready_checkpoint(run_root),
        "validation": _validation_ready(release_root, run_root),
        "public_submission": _submission_ready(release_root, run_root),
    }
    if request.profile == "e1-bm25":
        checks["parent_e0_validation"] = _parent_validation_ready(
            release_root, request.parent_run_root
        )
        checks["index"] = _status(run_root / "index" / "index_manifest.json", "READY")
    next_operation, next_stage = _next_step(request.profile, checks)
    blocked = checks["active_operation"]["status"] == "RUNNING" or (
        request.profile == "e1-bm25" and checks["parent_e0_validation"]["status"] != "PASS"
    )
    return {
        "schema_version": "task2-pipeline-inventory-v1",
        "task_id": "Task2",
        "profile": request.profile,
        "status": (
            "BLOCKED" if blocked else "COMPLETE" if next_operation is None else "READY_TO_ADVANCE"
        ),
        "checks": checks,
        "next_operation": next_operation,
        "next_stage": next_stage,
        "advance_policy": "one_operation_per_request",
    }


def _roots(request: PipelineRequest) -> tuple[Path, Path]:
    if request.control_root is None or request.run_root is None:
        raise ValueError("Inventory requires control_root and run_root.")
    return request.control_root, request.run_root


def _status(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "INVALID", "path": str(path), "reason": str(exc)}
    actual = payload.get("status") if isinstance(payload, dict) else None
    return {
        "status": "PASS" if actual == expected else "NOT_READY",
        "artifact_status": actual,
        "path": str(path),
    }


def _active_operation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "IDLE", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "INVALID_LOCK", "path": str(path), "reason": str(exc)}
    safe = {
        key: payload.get(key)
        for key in ("schema_version", "operation", "pid", "started_at")
        if isinstance(payload, dict)
    }
    return {"status": "RUNNING", "path": str(path), "lock": safe}


def _ready_checkpoint(run_root: Path) -> dict[str, Any]:
    run = _status(run_root / "run_manifest.json", "READY")
    checkpoint = _status(run_root / "checkpoint_manifest.json", "READY")
    status = "PASS" if run["status"] == checkpoint["status"] == "PASS" else "NOT_READY"
    return {"status": status, "run": run, "checkpoint": checkpoint}


def _validation_ready(release_root: Path, run_root: Path) -> dict[str, Any]:
    metrics = _status(run_root / "metrics.json", "PASS")
    predictions = run_root / "validation_predictions.json"
    if metrics["status"] != "PASS" or not predictions.is_file():
        return {"status": "NOT_READY", "metrics": metrics, "predictions": str(predictions)}
    try:
        report = validate_predictions(
            release_root / "qa" / "validation.jsonl",
            predictions,
            expected_split="validation",
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "INVALID", "reason": str(exc), "metrics": metrics}
    return {"status": report["status"], "metrics": metrics, "predictions": report}


def _submission_ready(release_root: Path, run_root: Path) -> dict[str, Any]:
    report = _status(run_root / "public_submission_report.json", "PASS")
    zip_path = run_root / "public_submission.zip"
    if report["status"] != "PASS" or not zip_path.is_file():
        return {"status": "NOT_READY", "report": report, "zip": str(zip_path)}
    try:
        predictions_path = run_root / "public_predictions.json"
        prediction_report = validate_predictions(
            release_root / "qa" / "public.jsonl",
            predictions_path,
            expected_split="public",
        )
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        archive = validate_submission_zip(zip_path, expected_ids={str(key) for key in predictions})
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "INVALID", "reason": str(exc)}
    status = "PASS" if prediction_report["status"] == archive["status"] == "PASS" else "FAIL"
    return {
        "status": status,
        "report": report,
        "predictions": prediction_report,
        "archive": archive,
    }


def _parent_validation_ready(release_root: Path, parent_root: Path | None) -> dict[str, Any]:
    if parent_root is None:
        return {"status": "MISSING", "reason": "E1 requires parent_run_root."}
    checkpoint = _ready_checkpoint(parent_root)
    validation = _validation_ready(release_root, parent_root)
    status = "PASS" if checkpoint["status"] == validation["status"] == "PASS" else "NOT_READY"
    return {"status": status, "checkpoint": checkpoint, "validation": validation}


def _next_step(profile: str, checks: dict[str, dict[str, Any]]) -> tuple[str | None, str | None]:
    if checks["release"]["status"] != "PASS":
        return "preflight", "training"
    if checks["tokenizer"]["status"] != "PASS":
        return "audit-tokenizer", "tokenizer"
    if profile == "e1-bm25":
        if checks["parent_e0_validation"]["status"] != "PASS":
            return None, None
        if checks["index"]["status"] != "PASS":
            return "build-index", "indexing"
    elif checks["checkpoint"]["status"] != "PASS":
        return "train", "training"
    if checks["validation"]["status"] != "PASS":
        return "evaluate", "evaluation"
    if checks["public_submission"]["status"] != "PASS":
        return "predict", "public"
    return None, None
