"""Route the full Task 2 control plane through one compact JSON request."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from Offline.pipeline import (
    SCHEMA_VERSION as PREPROCESS_SCHEMA,
)
from Offline.pipeline import (
    contract_payload as preprocessing_contract,
)
from Offline.pipeline import (
    run_request as run_preprocessing,
)
from Online.cli import run_request as run_online
from Online.request import SCHEMA_VERSION as ONLINE_SCHEMA

EXIT_INVALID = 2


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def contract_payload() -> dict[str, Any]:
    """Expose stage ownership without loading ML dependencies."""
    from Offline.contracts import load_contract

    return {
        "schema_version": "task2-control-plane-v1",
        "task_id": "Task2",
        "commands": ["contract", "run --request"],
        "offline": preprocessing_contract(),
        "online": {
            "request_schema": ONLINE_SCHEMA,
            "artifact_contract": load_contract().raw,
        },
    }


def _request_schema(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pipeline request must be a JSON object.")
    return str(payload.get("schema_version", ""))


def run_request(path: Path) -> int:
    """Dispatch preprocessing or online work by its versioned schema."""
    schema = _request_schema(path)
    if schema == PREPROCESS_SCHEMA:
        return run_preprocessing(path)
    if schema == ONLINE_SCHEMA:
        return run_online(path)
    raise ValueError(
        f"Unsupported request schema_version: {schema or '<missing>'}. "
        f"Expected {PREPROCESS_SCHEMA} or {ONLINE_SCHEMA}."
    )


def build_parser() -> argparse.ArgumentParser:
    """Expose only contract inspection and request execution."""
    parser = argparse.ArgumentParser(description="Task 2 local control plane.")
    commands = parser.add_subparsers(dest="command", required=True)
    contract = commands.add_parser("contract")
    contract.add_argument("--output", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect the contract or execute one request."""
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
        print(f"PIPELINE_ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
