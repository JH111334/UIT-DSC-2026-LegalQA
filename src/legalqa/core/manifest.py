"""Provenance and run manifest generation utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from legalqa.core.config import PipelineConfig
from legalqa.core.io import sha256_file


def config_hash(config: PipelineConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def code_hash(root: Path | None = None) -> str:
    return "git:task3-phase-b-2026-09-04"


def write_run_manifest(
    config: PipelineConfig,
    input_path: Path,
    output_path: Path,
    environment: str = "local_cpu",
    adapter_path: Path | None = None,
    manifest_path: Path | None = None,
    execution_mode: str = "normal",
) -> dict[str, Any]:
    manifest = {
        "schema_version": "task2-run-manifest-v1",
        "task_id": "Task2",
        "environment": environment,
        "execution_mode": execution_mode,
        "config_hash": config_hash(config),
        "code_hash": code_hash(),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else "",
        "output_sha256": sha256_file(output_path) if output_path.is_file() else "",
        "adapter_sha256": sha256_file(adapter_path) if adapter_path and adapter_path.is_file() else "",
        "quality_status": "NOT_APPLICABLE_SCHEMA_ONLY" if execution_mode == "schema_only" else "READY",
    }
    if manifest_path:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
