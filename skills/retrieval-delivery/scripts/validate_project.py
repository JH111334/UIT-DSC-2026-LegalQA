"""Validate repository structure, model governance, and local data integrity."""

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

REQUIRED_PATHS = (
    "AGENTS.md",
    "Workflows/flowinfo.md",
    "configs/Shared/smoke.toml",
    "configs/Shared/model_allowlist.json",
    "configs/Task1/benchmark.toml",
    "configs/Task2/benchmark.toml",
    "Data/Shared/fixtures/documents.jsonl",
    "Data/Shared/fixtures/queries.jsonl",
    "Data/Shared/fixtures/qrels.jsonl",
    "Data/Task1/warmup_data/manifest.json",
    "Data/Task2/warmup_data/manifest.json",
    "Documents/Docs/Task1/architecture.md",
    "Documents/Docs/Task2/architecture.md",
    "Documents/Decision-making/Task1/0001-baseline-and-model-gate.md",
    "Documents/Decision-making/Task2/0001-evidence-first-answering.md",
    "Source/Shared/text_retrieval_agent/contracts.py",
    "Source/Task1/legal_ir/contracts.py",
    "Source/Task2/legal_qa/contracts.py",
    "tests/Shared/test_integration.py",
    "tests/Task1/test_legal_ir.py",
    "tests/Task2/test_legal_qa.py",
)

LOCAL_DATA_FILES = (
    "Data/Task1/warmup_data/warmup_Task1.json",
    "Data/Task2/warmup_data/warmup_Task2.json",
)

SELECTED_MODEL_FIELDS = {
    "configs/Task1/benchmark.toml": (
        ("candidate_generation", "dense_model"),
        ("reranking", "model"),
    ),
    "configs/Task2/benchmark.toml": (
        ("answer", "primary_generator"),
        ("answer", "rag_challenger"),
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _tracked_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    )
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line}


def _validate_allowlist(root: Path) -> list[str]:
    errors: list[str] = []
    registry = _load_json(root / "configs/Shared/model_allowlist.json")
    models = registry.get("models")
    duplicates = registry.get("duplicate_entries")
    if not isinstance(models, list) or not isinstance(duplicates, list):
        return ["Model registry needs models and duplicate_entries arrays."]
    ids = [str(model.get("id", "")) for model in models if isinstance(model, dict)]
    if len(ids) != 49 or len(set(ids)) != len(ids):
        errors.append("Model registry must contain 49 unique organizer entries.")
    if registry.get("raw_entry_count") != len(ids) + len(duplicates):
        errors.append("raw_entry_count does not reconcile with normalized duplicates.")
    if not set(str(item) for item in duplicates).issubset(ids):
        errors.append("Every duplicate entry must resolve to one normalized model ID.")
    allowed = set(ids)
    for relative_path, fields in SELECTED_MODEL_FIELDS.items():
        with (root / relative_path).open("rb") as source:
            config = tomllib.load(source)
        for section, key in fields:
            model_id = str(config[section][key])
            if model_id not in allowed:
                errors.append(f"Unallowlisted model in {relative_path}: {model_id}")
    return errors


def _validate_local_data(root: Path, tracked: set[str]) -> list[str]:
    errors: list[str] = []
    for relative_path in LOCAL_DATA_FILES:
        if relative_path in tracked:
            errors.append(f"Organizer data is tracked: {relative_path}")
        data_path = root / relative_path
        if not data_path.exists():
            continue
        manifest = _load_json(data_path.with_name("manifest.json"))
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if digest != manifest.get("sha256"):
            errors.append(f"Checksum mismatch: {relative_path}")
        if len(_load_json(data_path)) != manifest.get("records"):
            errors.append(f"Record count mismatch: {relative_path}")
    return errors


def main() -> int:
    """Validate required files, allowlisted models, and untracked organizer data."""
    root = Path(__file__).resolve().parents[3]
    errors = [
        f"Missing required file: {path}" for path in REQUIRED_PATHS if not (root / path).is_file()
    ]
    tracked = _tracked_paths(root)
    forbidden_prefixes = ("models/", "artifacts/", "checkpoints/")
    errors.extend(
        f"Generated artifact is tracked: {path}"
        for path in sorted(tracked)
        if path.startswith(forbidden_prefixes)
    )
    errors.extend(_validate_allowlist(root))
    errors.extend(_validate_local_data(root, tracked))
    if errors:
        raise SystemExit("Project validation failed:\n- " + "\n- ".join(errors))
    print("Project validation passed: structure, allowlist, and local-data policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
