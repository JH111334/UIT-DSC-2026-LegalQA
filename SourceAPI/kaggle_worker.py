"""Run the canonical Task 2 E0 train, validation, and Public batch on Kaggle."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

PAYLOAD_SCHEMA = "task2-kaggle-payload-v1"
OUTPUT_ROOT = Path("/kaggle/working")


class WorkerError(RuntimeError):
    """Report invalid payload provenance or runtime output."""


def main(dataset_root: Path) -> int:
    """Validate the payload and execute the full E0 batch."""
    started = time.time()
    try:
        payload_path = dataset_root / "payload_manifest.json"
        payload = _load_object(payload_path)
        _verify_payload(dataset_root, payload)
        runtime_root = OUTPUT_ROOT / "runtime"
        release_root = OUTPUT_ROOT / "release"
        control_root = OUTPUT_ROOT / "control"
        run_root = OUTPUT_ROOT / "run"
        _prepare_workspace(dataset_root, release_root, control_root)
        _verify_gpu_contract()
        snapshot = _download_snapshot(payload, control_root)
        model_manifest = _load_object(control_root / "model/model_snapshot_manifest.json")
        _verify_snapshot(snapshot, model_manifest)
        sys.path.insert(0, str(runtime_root / "Source" / "Task2"))
        result = _execute_pipeline(release_root, control_root, run_root, payload)
        remote = _remote_manifest(
            payload_path,
            payload,
            run_root,
            elapsed_seconds=time.time() - started,
            operation_result=result,
        )
        _write_json(OUTPUT_ROOT / "remote_execution_manifest.json", remote)
        _write_bundle(run_root, OUTPUT_ROOT / "remote_execution_manifest.json")
        print(json.dumps({"status": "PASS", "run_id": payload["run_id"]}), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "task2-kaggle-failure-v1",
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(OUTPUT_ROOT / "task2_failure.json", failure)
        raise


def _verify_payload(root: Path, payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != PAYLOAD_SCHEMA or payload.get("task_id") != "Task2":
        raise WorkerError("Invalid Task 2 Kaggle payload contract.")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise WorkerError("Payload manifest has no file hash map.")
    for name, expected in files.items():
        path = root / str(name)
        if not path.is_file() or _sha256(path) != expected:
            raise WorkerError(f"Payload hash mismatch: {name}")
    release = _load_object(root / "release_manifest.json")
    if _sha256(root / "release_manifest.json") != payload.get("release_manifest_sha256"):
        raise WorkerError("Release manifest hash does not match the local handoff.")
    if release.get("release_id") != payload.get("release_id") or release.get("status") != "READY":
        raise WorkerError("Release identity/status mismatch.")
    for split in ("train", "validation", "public"):
        expected = release.get("artifacts", {}).get(f"qa/{split}.jsonl", {}).get("sha256")
        if _sha256(root / f"{split}.jsonl") != expected:
            raise WorkerError(f"Canonical QA hash mismatch: {split}")


def _prepare_workspace(dataset: Path, release: Path, control: Path) -> None:
    (release / "qa").mkdir(parents=True)
    for split in ("train", "validation", "public"):
        shutil.copy2(dataset / f"{split}.jsonl", release / "qa" / f"{split}.jsonl")
    shutil.copy2(dataset / "release_manifest.json", release / "manifest.json")
    destinations = {
        "training_config.json": control / "training/training_config.json",
        "decoding_config.json": control / "inference/decoding_config.json",
        "model_snapshot_manifest.json": control / "model/model_snapshot_manifest.json",
        "tokenizer_report_qwen.json": control / "tokenizer/tokenizer_report_qwen.json",
    }
    for source_name, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset / source_name, destination)


def _verify_gpu_contract() -> None:
    """Fail before model download when Kaggle assigns an incompatible GPU."""
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise WorkerError("Kaggle GPU is unavailable.")
    capability = torch.cuda.get_device_capability(0)
    supported = {item for item in torch.cuda.get_arch_list() if item.startswith("sm_")}
    current = f"sm_{capability[0]}{capability[1]}"
    if current not in supported:
        raise WorkerError(
            f"Assigned GPU {torch.cuda.get_device_name(0)} ({current}) is unsupported by "
            f"the installed PyTorch architectures: {sorted(supported)}"
        )


def _download_snapshot(payload: dict[str, Any], control: Path) -> Path:
    hub = importlib.import_module("huggingface_hub")
    manifest_path = control / "model/model_snapshot_manifest.json"
    manifest = _load_object(manifest_path)
    if manifest.get("model_id") != payload.get("model_id"):
        raise WorkerError("Model ID differs from the approved payload.")
    if manifest.get("model_revision") != payload.get("model_revision"):
        raise WorkerError("Model revision differs from the approved payload.")
    snapshot = OUTPUT_ROOT / "model_snapshot"
    expected_names = [str(item["name"]) for item in manifest.get("snapshot_files", [])]
    hub.snapshot_download(
        repo_id=str(payload["model_id"]),
        revision=str(payload["model_revision"]),
        local_dir=snapshot,
        allow_patterns=expected_names,
    )
    manifest["local_source_snapshot_path"] = manifest.get("snapshot_path")
    manifest["snapshot_path"] = str(snapshot)
    _write_json(manifest_path, manifest)
    return snapshot


def _verify_snapshot(snapshot: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("snapshot_files")
    if not isinstance(files, list) or not files:
        raise WorkerError("Model manifest has no snapshot file inventory.")
    for record in files:
        if not isinstance(record, dict):
            raise WorkerError("Invalid model file inventory record.")
        path = snapshot / str(record.get("name", ""))
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise WorkerError(f"Pinned model file mismatch: {path.name}")


def _execute_pipeline(
    release: Path, control: Path, run: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise WorkerError("Kaggle GPU is unavailable.")
    evaluation = importlib.import_module("Online.evaluation")
    output_audit = importlib.import_module("Online.output_audit")
    batch = importlib.import_module("Online.RetrievingAnswer.batch")
    engine_module = importlib.import_module("Online.RetrievingAnswer.engine")
    submission_module = importlib.import_module("Online.RetrievingAnswer.submission")
    training_module = importlib.import_module("Online.Training.runtime")

    training = training_module.train_e0(release, control, run)
    _pin_runtime_code(run, str(payload["files"]["runtime.bundle"]))
    gc.collect()
    evidence_provider = None
    index_candidate = control / "index"
    if not (index_candidate / "index_manifest.json").is_file():
        index_candidate = release / "index"
    if (index_candidate / "index_manifest.json").is_file():
        bm25_module = importlib.import_module("Online.RetrievingAnswer.bm25")
        index = bm25_module.SQLiteBM25Index(index_candidate)
        evidence_provider = engine_module.BM25Evidence(index)
    engine = engine_module.TransformersAnswerEngine(
        control, run, evidence_provider=evidence_provider
    )
    validation = batch.BatchInferenceRunner(engine).run(
        release / "qa/validation.jsonl",
        run / "validation_predictions.json",
        expected_split="validation",
        trace_path=run / "validation_trace.json",
        resume=False,
    )
    metrics = evaluation.evaluate_run(release, run)
    public = batch.BatchInferenceRunner(engine).run(
        release / "qa/public.jsonl",
        run / "public_predictions.json",
        expected_split="public",
        trace_path=run / "public_trace.json",
        resume=False,
    )
    submission = submission_module.build_submission(
        release / "qa/public.jsonl",
        run / "public_predictions.json",
        run / "public_submission.zip",
        expected_split="public",
        overwrite=False,
    )
    audit = output_audit.audit_unlabeled_output(
        run / "public_submission.zip",
        run / "public_output_audit.json",
        run / "public_case_flags.jsonl",
        expected_ids=_question_ids(release / "qa/public.jsonl"),
    )
    _write_json(run / "public_submission_report.json", submission)
    return {
        "training": training,
        "validation_generation": validation,
        "metrics": metrics,
        "public_generation": public,
        "submission": submission,
        "output_audit": audit,
    }


def _pin_runtime_code(run: Path, bundle_hash: str) -> None:
    manifest_path = run / "run_manifest.json"
    manifest = _load_object(manifest_path)
    manifest["code_revision"] = f"runtime.bundle:{bundle_hash}"
    manifest["execution_environment"] = "private_kaggle_gpu_batch"
    _write_json(manifest_path, manifest)


def _remote_manifest(
    payload_path: Path,
    payload: dict[str, Any],
    run: Path,
    *,
    elapsed_seconds: float,
    operation_result: dict[str, Any],
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "schema_version": "task2-kaggle-execution-v1",
        "status": "PASS",
        "run_id": payload["run_id"],
        "payload_manifest_sha256": _sha256(payload_path),
        "release_manifest_sha256": payload["release_manifest_sha256"],
        "runtime_bundle_sha256": payload["files"]["runtime.bundle"],
        "base_model_id": payload["model_id"],
        "base_model_revision": payload["model_revision"],
        "adapter_tree_sha256": _tree_sha256(run / "checkpoint-or-adapter"),
        "public_submission_sha256": _sha256(run / "public_submission.zip"),
        "elapsed_seconds": elapsed_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        "python": platform.python_version(),
        "packages": packages,
        "operation_result": operation_result,
    }


def _write_bundle(run: Path, remote_manifest: Path) -> None:
    target = OUTPUT_ROOT / "task2_e0_bundle.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(remote_manifest, "remote_execution_manifest.json")
        for path in sorted(item for item in run.rglob("*") if item.is_file()):
            if "trainer-state" not in path.parts:
                archive.write(path, (Path("run") / path.relative_to(run)).as_posix())


def _question_ids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                result.add(str(json.loads(line)["question_id"]))
    return result


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkerError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


if __name__ == "__main__":
    dataset = Path(os.environ.get("TASK2_KAGGLE_DATASET", "/kaggle/input/task2"))
    raise SystemExit(main(dataset))
