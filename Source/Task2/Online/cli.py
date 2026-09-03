"""Expose one compact request-driven Task 2 execution entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Offline.contracts import ContractError, load_contract
from Offline.gate import PreflightGate

from .evaluation import evaluate_run
from .inventory import inventory_pipeline
from .output_audit import audit_unlabeled_output
from .request import PipelineRequest, RequestError, load_request
from .RetrievingAnswer.batch import BatchInferenceRunner
from .RetrievingAnswer.bm25 import (
    BM25Index,
    build_bm25_index,
    evaluate_bm25,
    write_jsonl,
)
from .RetrievingAnswer.engine import BM25Evidence, TransformersAnswerEngine
from .RetrievingAnswer.submission import build_submission
from .Training.runtime import TrainingRuntimeError, audit_tokenizer, train_e0

EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    """Create the required two-command parser; details remain in versioned JSON."""
    parser = argparse.ArgumentParser(description="Run the governed Task 2 local pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract = subparsers.add_parser("contract", help="Print the artifact contract.")
    contract.add_argument("--output", type=Path)
    run = subparsers.add_parser("run", help="Execute one versioned request.")
    run.add_argument("--request", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_preflight(request: PipelineRequest) -> dict[str, Any]:
    contract = load_contract()
    gate = PreflightGate(
        contract=contract,
        release_root=request.release_root,
        control_root=request.control_root,
        run_root=request.run_root,
    )
    report = gate.run(stage=request.stage, profile=request.profile, scope=request.scope)
    _write_json(request.report, report)
    if report["status"] != "PASS":
        raise RequestError(f"Preflight failed; inspect {request.report}.")
    print(f"PREFLIGHT_PASS: {request.report}")
    return report


def run_request(path: Path) -> int:
    """Inventory or execute exactly one approval-gated operation."""
    request = load_request(path)
    if request.operation == "inventory":
        inventory = inventory_pipeline(request)
        _write_json(request.report, inventory)
        print(f"INVENTORY_{inventory['status']}: {request.report}")
        return 0
    if request.operation == "advance":
        return _advance(request)
    return _execute(request)


def _advance(request: PipelineRequest) -> int:
    """Select and run one next step while preserving every stage preflight."""
    inventory = inventory_pipeline(request)
    if inventory["status"] == "BLOCKED":
        _write_json(request.report, inventory)
        raise RequestError(f"Pipeline advance is blocked; inspect {request.report}.")
    operation = inventory.get("next_operation")
    stage = inventory.get("next_stage")
    if operation is None or stage is None:
        _write_json(request.report, inventory)
        print(f"PIPELINE_COMPLETE: {request.report}")
        return 0
    selected = replace(request, operation=str(operation), stage=str(stage))
    selected.validate()
    return _execute(selected, inventory_before=inventory)


def _execute(
    request: PipelineRequest,
    *,
    inventory_before: dict[str, Any] | None = None,
) -> int:
    """Preflight and execute one concrete operation."""
    report = _run_preflight(request)
    if request.operation == "preflight":
        return 0
    control_root, run_root = _execution_roots(request)
    with _operation_lock(run_root, request.operation):
        if request.operation == "audit-tokenizer":
            result = audit_tokenizer(request.release_root, control_root)
        elif request.operation == "build-index":
            parent = request.parent_run_root
            if parent is None:
                raise RequestError("build-index requires parent_run_root.")
            _initialize_e1_run(request.release_root, parent, run_root)
            result = build_bm25_index(request.release_root, run_root)
        elif request.operation == "train":
            result = train_e0(request.release_root, control_root, run_root)
        elif request.operation == "evaluate":
            result = _evaluate(request, control_root, run_root)
        elif request.operation == "predict":
            result = _predict(request, control_root, run_root)
        else:
            raise RequestError(f"Unsupported executable operation: {request.operation}")
    report["operation"] = request.operation
    report["operation_result"] = result
    if inventory_before is not None:
        report["inventory_before"] = inventory_before
    _write_json(request.report, report)
    print(f"OPERATION_PASS: {request.operation}")
    return 0


def _evaluate(
    request: PipelineRequest,
    control_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    questions = request.release_root / "qa" / "validation.jsonl"
    predictions = run_root / "validation_predictions.json"
    traces = run_root / "validation_trace.json"
    engine = _answer_engine(request, control_root, run_root)
    generation = BatchInferenceRunner(engine).run(
        questions,
        predictions,
        expected_split="validation",
        trace_path=traces,
        resume=predictions.is_file(),
        resume_trace_requirements=engine.trace_contract(),
    )
    parent_predictions = run_root / "parent_validation_predictions.json"
    metrics = evaluate_run(
        request.release_root,
        run_root,
        parent_predictions=parent_predictions if parent_predictions.is_file() else None,
    )
    result: dict[str, Any] = {"generation": generation, "metrics": metrics}
    if request.profile == "e1-bm25":
        retrieval_metrics, retrieval_trace = evaluate_bm25(
            BM25Index(run_root / "index" / "bm25.sqlite3"),
            questions,
            request.release_root / "diagnostics" / "citation_qrels.jsonl",
        )
        _write_json(run_root / "retrieval_metrics.json", retrieval_metrics)
        write_jsonl(run_root / "retrieval_trace.jsonl", retrieval_trace)
        result["retrieval"] = retrieval_metrics
    return result


def _predict(
    request: PipelineRequest,
    control_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    questions = request.release_root / "qa" / f"{request.stage}.jsonl"
    predictions = run_root / f"{request.stage}_predictions.json"
    traces = run_root / f"{request.stage}_trace.json"
    engine = _answer_engine(request, control_root, run_root)
    generation = BatchInferenceRunner(engine).run(
        questions,
        predictions,
        expected_split=request.stage,
        trace_path=traces,
        resume=predictions.is_file(),
        resume_trace_requirements=engine.trace_contract(),
    )
    submission = build_submission(
        questions,
        predictions,
        run_root / f"{request.stage}_submission.zip",
        expected_split=request.stage,
        overwrite=True,
    )
    _write_json(run_root / f"{request.stage}_submission_report.json", submission)
    output_audit = audit_unlabeled_output(
        run_root / f"{request.stage}_submission.zip",
        run_root / f"{request.stage}_output_audit.json",
        run_root / f"{request.stage}_case_flags.jsonl",
        expected_ids=_question_ids(questions),
    )
    return {
        "generation": generation,
        "submission": submission,
        "output_audit": output_audit,
    }


def _answer_engine(
    request: PipelineRequest,
    control_root: Path,
    run_root: Path,
) -> TransformersAnswerEngine:
    if request.profile == "e0-direct":
        return TransformersAnswerEngine(control_root, run_root)
    config = _load_object(control_root / "training" / "training_config.json")
    evidence = BM25Evidence(
        BM25Index(run_root / "index" / "bm25.sqlite3"),
        top_k=int(config.get("retrieval_top_k", 5)),
        candidate_k=int(config.get("retrieval_candidate_k", 20)),
        max_context_chars=int(config.get("max_context_chars", 8000)),
    )
    return TransformersAnswerEngine(control_root, run_root, evidence_provider=evidence)


def _initialize_e1_run(release_root: Path, parent_root: Path, run_root: Path) -> None:
    parent_manifest = _load_object(parent_root / "run_manifest.json")
    checkpoint = _load_object(parent_root / "checkpoint_manifest.json")
    if parent_manifest.get("profile") != "e0-direct" or parent_manifest.get("status") != "READY":
        raise RequestError("E1 must inherit a READY E0 parent run.")
    parent_metrics = _load_object(parent_root / "metrics.json")
    if parent_metrics.get("status") != "PASS":
        raise RequestError("E1 requires a validated E0 parent with metrics status PASS.")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(parent_manifest)
    manifest.update(
        {
            "run_id": run_root.name,
            "profile": "e1-bm25",
            "parent_run_id": parent_manifest["run_id"],
            "data_manifest_sha256": _sha256(release_root / "manifest.json"),
            "status": "READY",
        }
    )
    _write_json(run_root / "run_manifest.json", manifest)
    checkpoint_copy = dict(checkpoint)
    checkpoint_copy["inherited_from_run"] = parent_manifest["run_id"]
    _write_json(run_root / "checkpoint_manifest.json", checkpoint_copy)
    parent_predictions = parent_root / "validation_predictions.json"
    if parent_predictions.is_file():
        shutil.copyfile(parent_predictions, run_root / "parent_validation_predictions.json")


def _execution_roots(request: PipelineRequest) -> tuple[Path, Path]:
    if request.control_root is None or request.run_root is None:
        raise RequestError(f"Operation {request.operation} requires control_root and run_root.")
    return request.control_root, request.run_root


@contextmanager
def _operation_lock(run_root: Path, operation: str) -> Iterator[None]:
    """Prevent concurrent writers from mutating one run bundle."""
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".pipeline.lock"
    payload = {
        "schema_version": "task2-operation-lock-v1",
        "operation": operation,
        "pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    try:
        with lock_path.open("x", encoding="utf-8", newline="\n") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2)
            target.write("\n")
    except FileExistsError as exc:
        raise RequestError(f"Run is locked by another operation: {lock_path}") from exc
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RequestError(f"Expected JSON object: {path}")
    return value


def _question_ids(path: Path) -> set[str]:
    """Read canonical question IDs without exposing question or answer content."""
    question_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("question_id"), str):
                raise RequestError(f"Invalid question_id at {path}:{line_number}.")
            question_id = value["question_id"]
            if question_id in question_ids:
                raise RequestError(f"Duplicate question_id at {path}:{line_number}.")
            question_ids.add(question_id)
    if not question_ids:
        raise RequestError(f"Question file is empty: {path}")
    return question_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the contract or execute one request."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "contract":
            payload = load_contract().raw
            if args.output:
                _write_json(args.output, payload)
            else:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        return run_request(args.request)
    except (
        ContractError,
        RequestError,
        TrainingRuntimeError,
        OSError,
        RuntimeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as exc:
        print(f"PIPELINE_ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
