"""Build or validate one Task 2 preprocessing release from a JSON request."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "task2-preprocess-request-v1"
OPERATIONS = {"build", "build-index", "preflight"}
PROFILES = {"e0-direct", "e1-bm25"}
STAGES = {"training", "evaluation", "public", "promotion"}
EXIT_INVALID = 2


class PreprocessRequestError(ValueError):
    """Report an invalid preprocessing request."""


def contract_payload() -> dict[str, Any]:
    """Return the small, dependency-free preprocessing contract."""
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "Task2",
        "commands": ["contract", "run --request"],
        "operations": sorted(OPERATIONS),
        "profiles": sorted(PROFILES),
        "stages": sorted(STAGES),
        "boundaries": {
            "canonical_release": "QA, corpus, reports, and diagnostic qrels only",
            "candidate_index": "external run_root/index; Phase B acceptance remains required",
            "excluded": ["model", "tokenizer control", "training", "inference"],
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_request(path: Path) -> tuple[dict[str, Any], Path]:
    request_path = path.resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PreprocessRequestError("Preprocessing request must be a JSON object.")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PreprocessRequestError(f"schema_version must be {SCHEMA_VERSION}.")
    operation = payload.get("operation")
    if operation not in OPERATIONS:
        raise PreprocessRequestError(f"Unsupported operation: {operation}")
    return payload, request_path.parent


def _path(payload: dict[str, Any], key: str, base: Path) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreprocessRequestError(f"Request field {key} must be a non-empty path.")
    path = Path(value)
    return path if path.is_absolute() else base / path


def run_request(path: Path) -> int:
    """Run one build or deep release preflight without model-side effects."""
    payload, base = _load_request(path)
    operation = str(payload["operation"])
    release_root = _path(payload, "release_root", base)
    report_path = _path(payload, "report", base)

    if operation == "build":
        if __package__:
            from .core.config import load_config
            from .release_builder import build_data_release
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from Offline.core.config import load_config
            from Offline.release_builder import build_data_release

        config = load_config(_path(payload, "config", base))
        report = build_data_release(
            config,
            release_root,
            release_id=str(payload.get("release_id") or release_root.name),
            validation_size=int(payload.get("validation_size", 700)),
            progress=print,
        )
    elif operation == "build-index":
        if __package__:
            from Online.RetrievingAnswer.bm25 import build_bm25_index

            from .release_audit import validate_data_release
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from Online.RetrievingAnswer.bm25 import build_bm25_index

            from Offline.release_audit import validate_data_release

        preflight = validate_data_release(
            release_root,
            profile="e1-bm25",
            stage="evaluation",
        )
        if preflight.get("status") != "PASS":
            report = {
                "schema_version": SCHEMA_VERSION,
                "task_id": "Task2",
                "status": "FAIL",
                "operation": operation,
                "release_preflight": preflight,
            }
        else:
            index_run_root = _path(payload, "index_run_root", base)
            index_manifest = build_bm25_index(
                release_root,
                index_run_root,
                index_id=str(
                    payload.get("index_id") or f"{release_root.name}-bm25-v1"
                ),
                manifest_status="CANDIDATE",
            )
            report = {
                "schema_version": SCHEMA_VERSION,
                "task_id": "Task2",
                "status": "PASS",
                "operation": operation,
                "release_id": release_root.name,
                "release_preflight": preflight,
                "index_manifest": index_manifest,
                "phase_b_acceptance_required": True,
            }
    else:
        if __package__:
            from .release_audit import validate_data_release
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from Offline.release_audit import validate_data_release

        profile = str(payload.get("profile", "e1-bm25"))
        stage = str(payload.get("stage", "evaluation"))
        if profile not in PROFILES:
            raise PreprocessRequestError(f"Unsupported profile: {profile}")
        if stage not in STAGES:
            raise PreprocessRequestError(f"Unsupported stage: {stage}")
        report = validate_data_release(
            release_root,
            profile=profile,
            stage=stage,
        )
    _write_json(report_path, report)
    status = str(report.get("status", "FAIL"))
    print(f"PREPROCESS_{status}: {report_path}")
    return 0 if status in {"PASS", "READY"} else EXIT_INVALID


def build_parser() -> argparse.ArgumentParser:
    """Expose only contract inspection and request execution."""
    parser = argparse.ArgumentParser(description="Task 2 preprocessing control plane.")
    commands = parser.add_subparsers(dest="command", required=True)
    contract = commands.add_parser("contract")
    contract.add_argument("--output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the contract or execute one versioned request."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "contract":
            payload = contract_payload()
            if args.output:
                _write_json(args.output, payload)
            else:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        return run_request(args.request)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"PREPROCESS_ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
