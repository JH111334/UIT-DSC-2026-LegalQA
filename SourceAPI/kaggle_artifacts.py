"""Prepare and import immutable artifacts for the private Task 2 Kaggle batch."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_SCHEMA = "task2-kaggle-payload-v1"


class KaggleRunError(RuntimeError):
    """Report a failed local gate, Kaggle operation, or output import."""


def prepare_payload(
    request: dict[str, Any], release_root: Path, control_root: Path, target: Path
) -> dict[str, Any]:
    """Copy the minimal E0 payload and bind every input to SHA-256."""
    manifest_path = release_root / "manifest.json"
    manifest = load_object(manifest_path)
    if manifest.get("task_id") != "Task2" or manifest.get("status") != "READY":
        raise KaggleRunError("The source release is not a READY Task 2 release.")
    shutil.copy2(manifest_path, target / "release_manifest.json")
    copied: list[Path] = [target / "release_manifest.json"]
    for split in ("train", "validation", "public"):
        source = release_root / "qa" / f"{split}.jsonl"
        expected = manifest.get("artifacts", {}).get(f"qa/{split}.jsonl", {}).get("sha256")
        if not source.is_file() or sha256(source) != expected:
            raise KaggleRunError(f"Release QA hash mismatch for {split}.")
        destination = target / f"{split}.jsonl"
        shutil.copy2(source, destination)
        copied.append(destination)
    controls = {
        "training_config.json": control_root / "training" / "training_config.json",
        "decoding_config.json": control_root / "inference" / "decoding_config.json",
        "model_snapshot_manifest.json": control_root / "model" / "model_snapshot_manifest.json",
        "tokenizer_report_qwen.json": control_root / "tokenizer" / "tokenizer_report_qwen.json",
    }
    for name, source in controls.items():
        if not source.is_file():
            raise KaggleRunError(f"Missing control artifact: {source}")
        shutil.copy2(source, target / name)
        copied.append(target / name)
    # Kaggle expands uploaded files with a .zip suffix into dataset paths. Keep
    # the ZIP container under an opaque suffix so its hash and bootstrap path
    # remain stable after publication.
    runtime = target / "runtime.bundle"
    _build_runtime_zip(runtime)
    copied.append(runtime)
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task_id": "Task2",
        "run_id": request["run_id"],
        "release_id": manifest["release_id"],
        "release_manifest_sha256": sha256(manifest_path),
        "model_id": request["model_id"],
        "model_revision": request["model_revision"],
        "files": {item.name: sha256(item) for item in sorted(copied)},
    }
    write_json(target / "payload_manifest.json", payload)
    payload_hash = sha256(target / "payload_manifest.json")
    write_json(
        target / "dataset-metadata.json",
        {
            "title": str(request["dataset_slug"]).split("/", 1)[1],
            "id": request["dataset_slug"],
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )
    return {"payload_sha256": payload_hash, "release_id": manifest["release_id"]}


def prepare_kernel(request: dict[str, Any], target: Path) -> None:
    """Write a minimal bootstrap and private GPU kernel metadata."""
    packages = request.get("packages", [])
    if not isinstance(packages, list) or not all(isinstance(item, str) for item in packages):
        raise KaggleRunError("packages must be a list of pinned requirement strings.")
    dataset_name = str(request["dataset_slug"]).split("/", 1)[1]
    bootstrap = (
        "import json, pathlib, subprocess, sys, time, zipfile\n"
        f"PACKAGES = {json.dumps(packages)}\n"
        "if PACKAGES:\n"
        "    command = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q']\n"
        "    subprocess.check_call([*command, *PACKAGES])\n"
        f"dataset = pathlib.Path('/kaggle/input/{dataset_name}')\n"
        "for attempt in range(60):\n"
        "    if (dataset / 'runtime.bundle').is_file():\n"
        "        break\n"
        "    candidates = list(pathlib.Path('/kaggle/input').rglob('runtime.bundle'))\n"
        "    if candidates:\n"
        "        dataset = candidates[0].parent\n"
        "        break\n"
        "    time.sleep(5)\n"
        "runtime = pathlib.Path('/kaggle/working/runtime')\n"
        "runtime.mkdir(parents=True, exist_ok=True)\n"
        "with zipfile.ZipFile(dataset / 'runtime.bundle') as archive:\n"
        "    archive.extractall(runtime)\n"
        "sys.path.insert(0, str(runtime))\n"
        "from SourceAPI.kaggle_worker import main\n"
        "raise SystemExit(main(dataset))\n"
    )
    (target / "main.py").write_text(bootstrap, encoding="utf-8", newline="\n")
    write_json(
        target / "kernel-metadata.json",
        {
            "id": request["kernel_slug"],
            "title": str(request["kernel_slug"]).split("/", 1)[1],
            "code_file": "main.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [request["dataset_slug"]],
            "competition_sources": [],
            "kernel_sources": [],
        },
    )


def import_bundle(
    download: Path, run_root: Path, submission: Path, release_root: Path, payload_hash: str
) -> dict[str, Any]:
    """Import a matching output bundle and replay-validate the submission locally."""
    bundle = next(download.rglob("task2_e0_bundle.zip"), None)
    if bundle is None:
        raise KaggleRunError("Kaggle output does not contain task2_e0_bundle.zip.")
    extracted = download / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(extracted)
    remote = load_object(extracted / "remote_execution_manifest.json")
    if remote.get("status") != "PASS" or remote.get("payload_manifest_sha256") != payload_hash:
        raise KaggleRunError("Remote execution manifest does not match the uploaded payload.")
    source_run = extracted / "run"
    checkpoint = load_object(source_run / "checkpoint_manifest.json")
    checkpoint["remote_checkpoint_path"] = checkpoint["checkpoint_path"]
    checkpoint["checkpoint_path"] = str((run_root / "checkpoint-or-adapter").resolve())
    write_json(source_run / "checkpoint_manifest.json", checkpoint)
    shutil.copytree(source_run, run_root)
    sys.path.insert(0, str(ROOT / "Source" / "Task2"))
    submission_module = importlib.import_module("Online.RetrievingAnswer.submission")
    validate_submission_zip = submission_module.validate_submission_zip

    expected_ids = _question_ids(release_root / "qa" / "public.jsonl")
    report = validate_submission_zip(run_root / "public_submission.zip", expected_ids=expected_ids)
    if report["status"] != "PASS":
        raise KaggleRunError(f"Downloaded submission failed local replay: {report['errors']}")
    submission.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "public_submission.zip", submission)
    shutil.copy2(extracted / "remote_execution_manifest.json", run_root)
    return {
        "bundle_sha256": sha256(bundle),
        "submission_sha256": sha256(submission),
        "submission_validation": report,
        "run_root": str(run_root),
    }


def workspace_path(value: object) -> Path:
    """Resolve and constrain one request path to the project workspace."""
    path = Path(str(value))
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise KaggleRunError(f"Path escapes the workspace: {resolved}")
    return resolved


def replace_staging(path: Path) -> None:
    """Replace only the selected run staging directory under the safe temp root."""
    allowed = (ROOT / ".tmp" / "kaggle-task2").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(allowed) or resolved == allowed:
        raise KaggleRunError("Unsafe staging path.")
    if resolved.exists():
        shutil.rmtree(resolved)


def load_object(path: Path) -> dict[str, Any]:
    """Load a JSON object or reject its shape."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KaggleRunError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write one stable UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    """Hash one file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_runtime_zip(target: Path) -> None:
    roots = [ROOT / "Source" / "Task2", ROOT / "SourceAPI"]
    if (ROOT / "src").is_dir():
        roots.append(ROOT / "src")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(ROOT).as_posix())


def _question_ids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                result.add(str(json.loads(line)["question_id"]))
    return result
