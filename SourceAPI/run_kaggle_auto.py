"""Orchestrate one private, hash-locked Kaggle Task 2 E0 batch."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from SourceAPI.kaggle_artifacts import (
    ROOT,
    KaggleRunError,
    import_bundle,
    load_object,
    prepare_kernel,
    prepare_payload,
    replace_staging,
    workspace_path,
    write_json,
)

SCHEMA_VERSION = "task2-kaggle-batch-request-v1"


def run_request(path: Path) -> int:
    """Prepare, execute, monitor, download, and validate one Kaggle request."""
    request = load_object(path.resolve())
    _validate_request(request)
    release_root = workspace_path(request["release_root"])
    control_root = workspace_path(request["control_root"])
    run_root = workspace_path(request["run_root"])
    submission_path = workspace_path(request["submission_path"])
    report_path = workspace_path(request["report"])
    if run_root.exists() or submission_path.exists():
        raise KaggleRunError("Run root and submission target must be new immutable paths.")

    staging = ROOT / ".tmp" / "kaggle-task2" / str(request["run_id"])
    replace_staging(staging)
    dataset_dir = staging / "dataset"
    kernel_dir = staging / "kernel"
    download_dir = staging / "download"
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)
    download_dir.mkdir(parents=True)

    payload = prepare_payload(request, release_root, control_root, dataset_dir)
    prepare_kernel(request, kernel_dir)
    started = time.time()
    _publish_dataset(dataset_dir, str(request["dataset_slug"]), payload["payload_sha256"])
    _command(
        ["kaggle", "kernels", "push", "-p", str(kernel_dir), "--accelerator", "NvidiaTeslaT4"],
        timeout=600,
    )
    _monitor(str(request["kernel_slug"]), int(request.get("timeout_minutes", 720)))
    _command(
        [
            "kaggle",
            "kernels",
            "output",
            str(request["kernel_slug"]),
            "-p",
            str(download_dir),
            "-o",
        ],
        timeout=600,
    )
    imported = import_bundle(
        download_dir,
        run_root,
        submission_path,
        release_root,
        payload["payload_sha256"],
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "run_id": request["run_id"],
        "dataset_slug": request["dataset_slug"],
        "kernel_slug": request["kernel_slug"],
        "payload_sha256": payload["payload_sha256"],
        "elapsed_seconds": time.time() - started,
        **imported,
    }
    write_json(report_path, report)
    print(f"KAGGLE_TASK2_PASS: {report_path}")
    print(f"SUBMISSION_ZIP: {submission_path}")
    return 0


def _validate_request(value: dict[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise KaggleRunError(f"Request schema must be {SCHEMA_VERSION}.")
    if value.get("operation") != "train-evaluate-predict" or value.get("stage") != "e0":
        raise KaggleRunError("Only the Task 2 E0 train-evaluate-predict batch is approved.")
    if value.get("private_kernel") is not True or value.get("private_dataset") is not True:
        raise KaggleRunError("Kaggle dataset and kernel must both be private.")
    for field in ("dataset_slug", "kernel_slug"):
        if not _valid_slug(str(value.get(field, ""))):
            raise KaggleRunError(f"Invalid {field}.")
    if value.get("model_id") != "Qwen/Qwen2.5-1.5B-Instruct":
        raise KaggleRunError("Only the allowlisted E0 Qwen anchor is approved.")
    revision = str(value.get("model_revision", ""))
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise KaggleRunError("model_revision must be a pinned 40-character commit hash.")


def _valid_slug(value: str) -> bool:
    """Accept one lowercase owner/resource Kaggle slug."""
    parts = value.split("/")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    return len(parts) == 2 and all(part and set(part) <= allowed for part in parts)


def _publish_dataset(target: Path, slug: str, payload_hash: str) -> None:
    probe = subprocess.run(
        ["kaggle", "datasets", "files", slug], capture_output=True, text=True, check=False
    )
    if probe.returncode == 0:
        command = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(target),
            "-m",
            f"Task2 E0 payload {payload_hash[:12]}",
            "-q",
            "-t",
        ]
    else:
        command = ["kaggle", "datasets", "create", "-p", str(target), "-q", "-t"]
    _command(command, timeout=1800)
    _wait_for_dataset_payload(slug)


def _wait_for_dataset_payload(slug: str, timeout_seconds: int = 600) -> None:
    """Wait until Kaggle exposes every top-level payload file to kernels."""
    required = {
        "payload_manifest.json",
        "release_manifest.json",
        "runtime.bundle",
        "train.jsonl",
        "validation.jsonl",
        "public.jsonl",
    }
    deadline = time.monotonic() + timeout_seconds
    missing: list[str] = list(required)
    while time.monotonic() < deadline:
        status_proc = _command(["kaggle", "datasets", "status", slug], timeout=60, capture=True)
        if "ready" in status_proc.stdout.casefold():
            result = _command(
                ["kaggle", "datasets", "files", slug], timeout=60, capture=True
            )
            missing = sorted(name for name in required if name not in result.stdout)
            if not missing:
                time.sleep(15)
                return
        time.sleep(10)
    raise KaggleRunError(f"Kaggle dataset payload is not ready: {missing}")


def _monitor(slug: str, timeout_minutes: int) -> None:
    deadline = time.monotonic() + timeout_minutes * 60
    while time.monotonic() < deadline:
        result = _command(["kaggle", "kernels", "status", slug], timeout=60, capture=True)
        status = result.stdout.strip()
        print(status, flush=True)
        lowered = status.casefold()
        if "complete" in lowered:
            return
        if any(value in lowered for value in ("error", "fail", "cancel")):
            raise KaggleRunError(f"Kaggle kernel failed: {status}")
        time.sleep(30)
    raise KaggleRunError("Kaggle kernel monitoring timed out.")


def _command(
    command: list[str], *, timeout: int, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        command,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        capture_output=capture,
        env=env,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() if capture else "see Kaggle CLI output"
        raise KaggleRunError(f"Command failed ({command[:3]}): {detail}")
    return result
