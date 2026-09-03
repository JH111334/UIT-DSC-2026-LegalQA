"""Check consistency across Task 2 offline and runtime artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PARAMETER_LIMIT_EXCLUSIVE = 4_000_000_000
RELEASE_JSON_IDS = (
    "data_report",
    "leakage_report",
    "split_manifest",
    "tokenizer_report",
    "corpus_manifest",
    "corpus_report",
    "citation_match_report",
    "public_manifest",
    "private_manifest",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _existing_payload(paths: dict[str, Path], artifact_id: str) -> dict[str, Any] | None:
    path = paths.get(artifact_id)
    if path is None or not path.is_file():
        return None
    return _load_object(path)


def _check_release_ids(paths: dict[str, Path], errors: list[str]) -> tuple[str, str] | None:
    manifest_path = paths.get("release_manifest")
    manifest = _existing_payload(paths, "release_manifest")
    if manifest_path is None or manifest is None:
        return None
    release_id = str(manifest.get("release_id", ""))
    for artifact_id in RELEASE_JSON_IDS:
        payload = _existing_payload(paths, artifact_id)
        if payload is not None and payload.get("release_id") != release_id:
            errors.append(f"Release ID mismatch: {artifact_id}")
    return release_id, _sha256(manifest_path)


def _check_preprocess_approval(
    paths: dict[str, Path], release_identity: tuple[str, str] | None, errors: list[str]
) -> None:
    approval = _existing_payload(paths, "preprocess_release_approval")
    if approval is None or release_identity is None:
        return
    release_id, manifest_sha256 = release_identity
    if approval.get("release_id") != release_id:
        errors.append("Preprocess approval release_id does not match the release manifest.")
    if approval.get("release_manifest_sha256") != manifest_sha256:
        errors.append("Preprocess approval does not pin the current release manifest hash.")


def _check_model_and_scorer(paths: dict[str, Path], errors: list[str]) -> None:
    model = _existing_payload(paths, "model_snapshot_manifest")
    if model is not None:
        parameter_count = model.get("parameter_count")
        if not isinstance(parameter_count, int) or parameter_count >= PARAMETER_LIMIT_EXCLUSIVE:
            errors.append("Model parameter_count must be an integer strictly below 4B.")

    scorer = _existing_payload(paths, "scorer_contract")
    if scorer is not None:
        if scorer.get("status") not in {"APPROVED", "PROVISIONAL_APPROVED"}:
            errors.append("Scorer status must be APPROVED or PROVISIONAL_APPROVED.")
        if scorer.get("metric_order") != ["METEOR", "ROUGE-L"]:
            errors.append("Scorer metric_order must be METEOR then ROUGE-L.")


def _check_stage_profile(
    paths: dict[str, Path], stage: str, profile: str, errors: list[str]
) -> None:
    training = _existing_payload(paths, "training_config")
    if stage == "training" and training is not None and training.get("profile") != "e0-direct":
        errors.append("The only approved training scaffold profile is e0-direct.")

    run = _existing_payload(paths, "run_manifest")
    if run is not None and run.get("profile") != profile:
        errors.append("Run manifest profile does not match the requested profile.")


def _check_run_lineage(
    paths: dict[str, Path], release_identity: tuple[str, str] | None, errors: list[str]
) -> None:
    run = _existing_payload(paths, "run_manifest")
    if run is not None and release_identity is not None:
        release_id, manifest_sha256 = release_identity
        if run.get("data_release_id") != release_id:
            errors.append("Run manifest data_release_id mismatch.")
        if run.get("data_manifest_sha256") != manifest_sha256:
            errors.append("Run manifest does not pin the current data manifest hash.")

    index = _existing_payload(paths, "index_manifest")
    corpus_manifest_path = paths.get("corpus_manifest")
    chunks_path = paths.get("corpus_chunks")
    if (
        index is not None
        and corpus_manifest_path
        and corpus_manifest_path.is_file()
        and index.get("corpus_manifest_sha256") != _sha256(corpus_manifest_path)
    ):
        errors.append("Index manifest corpus hash mismatch.")
    if (
        index is not None
        and chunks_path
        and chunks_path.is_file()
        and index.get("chunks_sha256") != _sha256(chunks_path)
    ):
        errors.append("Index manifest chunks hash mismatch.")


def _check_phase(paths: dict[str, Path], stage: str, errors: list[str]) -> None:
    if stage not in {"public", "private"}:
        return
    phase_manifest = _existing_payload(paths, f"{stage}_manifest")
    question_path = paths.get(f"qa_{stage}")
    if (
        phase_manifest is not None
        and question_path
        and question_path.is_file()
        and phase_manifest.get("artifact_sha256") != _sha256(question_path)
    ):
        errors.append(f"{stage.capitalize()} manifest question artifact hash mismatch.")

    submission = _existing_payload(paths, "submission_contract")
    if submission is not None:
        if submission.get("phase") != stage:
            errors.append("Submission contract phase mismatch.")
        if submission.get("zip_members") != ["submission.json"]:
            errors.append("Submission ZIP must contain only submission.json.")
        allowed_encodings = {"UTF-8", "UTF-8_NO_BOM", "UTF-8-NO-BOM"}
        if str(submission.get("encoding", "")).upper() not in allowed_encodings:
            errors.append("Submission encoding must be UTF-8.")


def validate_consistency(
    paths: dict[str, Path], stage: str, profile: str, errors: list[str]
) -> None:
    """Validate lineage and policy relations spanning multiple artifact files."""
    try:
        release_identity = _check_release_ids(paths, errors)
        _check_preprocess_approval(paths, release_identity, errors)
        _check_model_and_scorer(paths, errors)
        _check_stage_profile(paths, stage, profile, errors)
        _check_run_lineage(paths, release_identity, errors)
        _check_phase(paths, stage, errors)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"Cross-artifact consistency check failed: {exc}")
