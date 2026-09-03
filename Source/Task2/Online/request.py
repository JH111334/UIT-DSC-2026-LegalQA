"""Load one compact request for the Task 2 pipeline entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "task2-pipeline-request-v1"
OPERATIONS = {
    "preflight",
    "inventory",
    "advance",
    "audit-tokenizer",
    "build-index",
    "train",
    "evaluate",
    "predict",
}
STAGES = {"auto", "tokenizer", "indexing", "training", "evaluation", "public", "private"}
PROFILES = {"e0-direct", "e1-bm25"}
SCOPES = {"release", "full"}


class RequestError(ValueError):
    """Report an invalid or contradictory pipeline request."""


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Describe one preflight or approval-gated Task 2 operation."""

    operation: str
    stage: str
    profile: str
    scope: str
    release_root: Path
    report: Path
    control_root: Path | None = None
    run_root: Path | None = None
    parent_run_root: Path | None = None

    def validate(self) -> None:
        """Reject unsupported values and inconsistent operation-stage pairs."""
        if self.operation not in OPERATIONS:
            raise RequestError(f"Unsupported operation: {self.operation}")
        if self.stage not in STAGES:
            raise RequestError(f"Unsupported stage: {self.stage}")
        if self.profile not in PROFILES:
            raise RequestError(f"Unsupported profile: {self.profile}")
        if self.scope not in SCOPES:
            raise RequestError(f"Unsupported scope: {self.scope}")
        if self.operation in {"inventory", "advance"} and self.stage != "auto":
            raise RequestError(f"Operation {self.operation} requires stage auto.")
        if self.stage == "auto" and self.operation not in {"inventory", "advance"}:
            raise RequestError("Stage auto is reserved for inventory and advance.")
        expected = {
            "audit-tokenizer": "tokenizer",
            "build-index": "indexing",
            "train": "training",
            "evaluate": "evaluation",
        }.get(self.operation)
        if expected is not None and self.stage != expected:
            raise RequestError(f"Operation {self.operation} requires stage {expected}.")
        if self.operation == "predict" and self.stage not in {"public", "private"}:
            raise RequestError("Operation predict requires stage public or private.")
        if self.operation == "build-index" and self.profile != "e1-bm25":
            raise RequestError("Operation build-index requires profile e1-bm25.")
        if self.operation == "build-index" and self.parent_run_root is None:
            raise RequestError("Operation build-index requires the E0 parent_run_root.")
        if self.operation in OPERATIONS - {"preflight"} and self.scope != "full":
            raise RequestError(f"Operation {self.operation} requires scope full.")
        if self.operation in OPERATIONS - {"preflight"} and (
            self.control_root is None or self.run_root is None
        ):
            raise RequestError(f"Operation {self.operation} requires control_root and run_root.")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RequestError("Pipeline request must be a JSON object.")
    return payload


def _path(value: object, base: Path, field_name: str, *, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"Request field {field_name} must be a non-empty path.")
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_request(path: Path) -> PipelineRequest:
    """Load a request and resolve relative paths beside its JSON file."""
    request_path = path.resolve()
    payload = _load_object(request_path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RequestError(f"Request schema_version must be {SCHEMA_VERSION}.")
    base = request_path.parent
    request = PipelineRequest(
        operation=str(payload.get("operation", "")),
        stage=str(payload.get("stage", "")),
        profile=str(payload.get("profile", "")),
        scope=str(payload.get("scope", "")),
        release_root=_required_path(payload.get("release_root"), base, "release_root"),
        control_root=_path(payload.get("control_root"), base, "control_root", optional=True),
        run_root=_path(payload.get("run_root"), base, "run_root", optional=True),
        parent_run_root=_path(
            payload.get("parent_run_root"),
            base,
            "parent_run_root",
            optional=True,
        ),
        report=_required_path(payload.get("report"), base, "report"),
    )
    request.validate()
    return request


def _required_path(value: object, base: Path, field_name: str) -> Path:
    resolved = _path(value, base, field_name)
    if resolved is None:
        raise RequestError(f"Request field {field_name} is required.")
    return resolved
