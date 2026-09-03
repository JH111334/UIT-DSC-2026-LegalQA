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
    "Data/Task2/manifest.json",
    "Documents/Docs/Task1/architecture.md",
    "Documents/Docs/Task2/architecture.md",
    "Documents/Decision-making/Task1/0001-baseline-and-model-gate.md",
    "Documents/Decision-making/Task2/0001-evidence-first-answering.md",
    "Documents/Decision-making/TASK1.md",
    "Documents/Decision-making/BanGiao.md",
    "Documents/Decision-making/Task2/0003-task2-official-corpus-retrieval.md",
    "Documents/Decision-making/Task2/0004-hybrid-before-citation-graph.md",
    "Source/Shared/text_retrieval_agent/contracts.py",
    "Source/Task1/legal_ir/contracts.py",
    "Source/Task2/legal_qa/contracts.py",
    "Source/Task2/pipeline.py",
    "Source/Task2/Offline/QA/README.md",
    "Source/Task2/Offline/Corpus/README.md",
    "Source/Task2/Offline/required_artifacts.json",
    "Source/Task2/Online/request.py",
    "Source/Task2/Online/Training/runtime.py",
    "Source/Task2/Online/RetrievingAnswer/batch.py",
    "Source/Task2/Online/RetrievingAnswer/fusion.py",
    "Source/Task2/Online/RetrievingAnswer/submission.py",
    "requirements.txt",
    "tests/Shared/test_integration.py",
    "tests/Task1/test_legal_ir.py",
    "tests/Task2/test_legal_qa.py",
    "tests/Task2/test_training_test_scaffold.py",
    "tests/Task2/test_training_test_runtime.py",
)

LOCAL_DATA_FILES = (
    "Data/Task1/warmup_data/warmup_Task1.json",
    "Data/Task2/train.json",
    "Data/Task2/public-official.json",
)

SELECTED_MODEL_FIELDS = {
    "configs/Task1/benchmark.toml": (
        ("candidate_generation", "dense_model"),
        ("reranking", "model"),
    ),
    "configs/Task2/benchmark.toml": (
        ("answer", "primary_generator"),
        ("answer", "legal_base_challenger"),
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
    if not ids or len(set(ids)) != len(ids):
        errors.append("Model registry must contain non-empty unique normalized entries.")
    if registry.get("raw_entry_count") != len(ids) + len(duplicates):
        errors.append("raw_entry_count does not reconcile with normalized duplicates.")
    if not set(str(item) for item in duplicates).issubset(ids):
        errors.append("Every duplicate entry must resolve to one normalized model ID.")
    aliases = registry.get("aliases")
    if not isinstance(aliases, list):
        errors.append("Model registry needs an aliases array.")
    else:
        for alias in aliases:
            if not isinstance(alias, dict) or str(alias.get("canonical", "")) not in set(ids):
                errors.append("Every model alias must resolve to one normalized model ID.")
    audit = registry.get("parameter_audit")
    blocked: set[str] = set()
    held: set[str] = set()
    if not isinstance(audit, dict):
        errors.append("Model registry needs a parameter_audit object.")
    else:
        blocked = {
            str(item.get("id", ""))
            for item in audit.get("blocked_verified_at_or_above_limit", [])
            if isinstance(item, dict)
        }
        held = {str(item) for item in audit.get("hold_until_exact_count_verified", [])}
        if not blocked.issubset(set(ids)) or not held.issubset(set(ids)):
            errors.append("Parameter-audit IDs must resolve to normalized model IDs.")
    allowed = set(ids)
    for relative_path, fields in SELECTED_MODEL_FIELDS.items():
        with (root / relative_path).open("rb") as source:
            config = tomllib.load(source)
        for section, key in fields:
            model_id = str(config[section][key])
            if model_id not in allowed:
                errors.append(f"Unallowlisted model in {relative_path}: {model_id}")
            if model_id in blocked or model_id in held:
                errors.append(f"Parameter-ineligible or unverified model selected: {model_id}")
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
        filename = data_path.name
        if (
            "files" in manifest
            and isinstance(manifest["files"], dict)
            and filename in manifest["files"]
        ):
            file_meta = manifest["files"][filename]
            expected_sha256 = file_meta.get("sha256")
            expected_records = file_meta.get("records")
        else:
            expected_sha256 = manifest.get("sha256")
            expected_records = manifest.get("records")
        if digest != expected_sha256:
            errors.append(f"Checksum mismatch: {relative_path}")
        if len(_load_json(data_path)) != expected_records:
            errors.append(f"Record count mismatch: {relative_path}")
    return errors


def _validate_task2_pipeline_gate(root: Path) -> list[str]:
    errors: list[str] = []
    contract = _load_json(root / "Source/Task2/Offline/required_artifacts.json")
    if contract.get("schema_version") != "task2-training-gate-v2":
        errors.append("Task 2 artifact contract must use gate schema v2.")
    if contract.get("task_id") != "Task2":
        errors.append("Task 2 artifact contract must be Task2-only.")
    profiles = contract.get("profiles")
    if profiles != ["e0-direct", "e1-bm25"]:
        errors.append("Task 2 profiles must preserve E0 direct and E1 BM25.")
    if not isinstance(contract.get("requirements"), dict) or not isinstance(
        contract.get("artifacts"), dict
    ):
        errors.append("Task 2 contract needs requirements and artifacts objects.")
    groups = contract.get("groups")
    artifacts = contract.get("artifacts")
    if isinstance(groups, dict):
        if "tokenizer_report" in groups.get("qa_core", []):
            errors.append("Tokenizer must not block the immutable data-release preflight.")
        if "tokenizer_report" not in groups.get("training_controls", []):
            errors.append("Full training must require a verified tokenizer report.")
    if isinstance(artifacts, dict):
        tokenizer = artifacts.get("tokenizer_report")
        if not isinstance(tokenizer, dict) or tokenizer.get("root") != "control":
            errors.append("Tokenizer report must be a control artifact before training.")

    with (root / "configs/Task2/benchmark.toml").open("rb") as source:
        benchmark = tomllib.load(source)
    retrieval = benchmark.get("retrieval")
    if not isinstance(retrieval, dict):
        errors.append("Task 2 benchmark needs a retrieval policy section.")
    else:
        if retrieval.get("reuse_task1_runtime") is not False:
            errors.append("Task 2 retrieval must reject Task 1 runtime reuse.")
        if retrieval.get("enabled") and retrieval.get("adr_status") != "accepted":
            errors.append("Task 2 retrieval cannot be enabled before its ADR is accepted.")
        if retrieval.get("neural_retrieval_target") != "hybrid_bm25_dense_rrf":
            errors.append("Task 2 neural retrieval target must be BM25+dense through RRF.")
        if retrieval.get("dense_only_role") != "diagnostic_ablation":
            errors.append("Task 2 dense-only retrieval must remain a diagnostic ablation.")
        if retrieval.get("citation_graph_enabled") is not False:
            errors.append("Task 2 citation graph must remain disabled before its evidence gate.")
        if (
            retrieval.get("citation_graph_policy")
            != "optional_after_hybrid_and_relation_failure_gate"
        ):
            errors.append("Task 2 citation graph policy must stay optional after the hybrid gate.")
    if (root / "Source/Task2/Online/RetrievingAnswer/graph.py").exists():
        errors.append("Task 2 citation graph runtime must not exist before its evidence gate.")
    if (root / "Source/Training&Test").exists():
        errors.append("Legacy Source/Training&Test must stay removed after the Task 2 migration.")
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
    if "Source/ol.zip" in tracked:
        errors.append("Opaque OL source archive must remain local-only and untracked.")
    errors.extend(_validate_allowlist(root))
    errors.extend(_validate_local_data(root, tracked))
    errors.extend(_validate_task2_pipeline_gate(root))
    if errors:
        raise SystemExit("Project validation failed:\n- " + "\n- ".join(errors))
    print("Project validation passed: structure, allowlist, and local-data policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
