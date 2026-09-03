"""Expose the approved private Kaggle batch control plane for DSC Task 2."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SourceAPI.kaggle_artifacts import KaggleRunError
from SourceAPI.run_kaggle_auto import run_request

SCHEMA_VERSION = "task2-kaggle-batch-request-v1"
EXIT_INVALID = 2


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def contract_payload() -> dict[str, Any]:
    """Return the narrowly approved Task 2 remote-compute contract."""
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "Task2",
        "status": "APPROVED_PRIVATE_KAGGLE_BATCH",
        "authority": "OPERATOR_CONFIRMED_WITH_BTC_2026-09-03",
        "commands": ["contract", "run --request"],
        "supported_operation": "train-evaluate-predict",
        "private_kernel_required": True,
        "private_dataset_required": True,
        "api_inference": False,
        "kaggle_api_role": "orchestration_only",
        "allowed_huggingface_use": "download_allowlisted_weights_only",
        "input_scope": "Task2 E0 QA train/validation/public plus hashed runtime controls",
        "forbidden": [
            "Task1 data or checkpoint",
            "external training data or augmentation",
            "Hugging Face inference API",
            "secret embedded in source or request",
            "Public tuning without canonical validation",
        ],
        "canonical_local_runtime": "Source/Task2/pipeline.py",
        "decision_record": "Documents/Decision-making/Task2/0006-private-kaggle-batch.md",
    }


def build_parser() -> argparse.ArgumentParser:
    """Expose contract inspection and one request-driven run command."""
    parser = argparse.ArgumentParser(description="Task 2 private Kaggle batch control plane.")
    commands = parser.add_subparsers(dest="command", required=True)
    contract = commands.add_parser("contract")
    contract.add_argument("--output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the contract or execute one approved Kaggle batch."""
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
    except (KaggleRunError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"KAGGLE_PIPELINE_ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
