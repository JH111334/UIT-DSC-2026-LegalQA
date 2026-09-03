"""Validate Task 2 release, control, and run artifacts before online stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .consistency import validate_consistency
from .contracts import ArtifactRule, ContractSpec


@dataclass
class GateState:
    """Mutable evidence collected during one preflight run."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    record_counts: dict[str, int] = field(default_factory=dict)


class PreflightGate:
    """Fail closed when any phase-specific artifact contract is missing or invalid."""

    def __init__(
        self,
        contract: ContractSpec,
        release_root: Path,
        control_root: Path | None = None,
        run_root: Path | None = None,
    ) -> None:
        """Bind a contract to roots without mutating any input artifact."""
        self.contract = contract
        self.roots = {
            "release": release_root,
            "control": control_root,
            "run": run_root,
        }

    def run(self, stage: str, profile: str, scope: str) -> dict[str, Any]:
        """Validate required artifacts and return a serializable report."""
        state = GateState()
        rules = self.contract.rules_for(stage, profile, scope)
        paths: dict[str, Path] = {}
        for rule in rules:
            path = self._resolve(rule, state)
            if path is None:
                continue
            paths[rule.artifact_id] = path
            self._validate_artifact(rule, path, state)

        self._validate_release_manifest(rules, paths, state)
        self._validate_split(paths, state)
        validate_consistency(paths, stage, profile, state.errors)
        return {
            "schema_version": "task2-preflight-report-v1",
            "task_id": "Task2",
            "stage": stage,
            "profile": profile,
            "scope": scope,
            "status": "PASS" if not state.errors else "FAIL",
            "checked_at": datetime.now(UTC).isoformat(),
            "required_artifacts": [rule.artifact_id for rule in rules],
            "checks": state.checks,
            "errors": state.errors,
            "warnings": state.warnings,
        }

    def _resolve(self, rule: ArtifactRule, state: GateState) -> Path | None:
        root = self.roots.get(rule.root)
        if root is None:
            state.errors.append(f"Root '{rule.root}' is required for {rule.artifact_id}.")
            state.checks.append(
                {"artifact_id": rule.artifact_id, "path": rule.path, "status": "MISSING_ROOT"}
            )
            return None
        return root / Path(rule.path)

    def _validate_artifact(self, rule: ArtifactRule, path: Path, state: GateState) -> None:
        if not path.is_file():
            state.errors.append(f"Missing artifact: {rule.root}:{rule.path}")
            state.checks.append(
                {"artifact_id": rule.artifact_id, "path": rule.path, "status": "MISSING"}
            )
            return

        check: dict[str, Any] = {
            "artifact_id": rule.artifact_id,
            "path": rule.path,
            "status": "PASS",
            "sha256": self._sha256(path),
        }
        try:
            if rule.kind == "json":
                self._validate_json(rule, path)
            elif rule.kind == "jsonl":
                count = self._validate_jsonl(rule, path)
                state.record_counts[rule.artifact_id] = count
                check["records"] = count
            else:
                raise ValueError(f"Unsupported kind '{rule.kind}'")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            check["status"] = "FAIL"
            state.errors.append(f"Invalid {rule.root}:{rule.path}: {exc}")
        state.checks.append(check)

    @staticmethod
    def _validate_json(rule: ArtifactRule, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        PreflightGate._check_mapping(payload, rule.required_keys, (), rule.expected)

    @staticmethod
    def _validate_jsonl(rule: ArtifactRule, path: Path) -> int:
        count = 0
        for line_number, payload in PreflightGate._iter_jsonl(path):
            try:
                PreflightGate._check_mapping(
                    payload,
                    rule.record_required_keys,
                    rule.record_forbidden_keys,
                    rule.record_expected,
                )
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            count += 1
        if count == 0 and not rule.allow_empty:
            raise ValueError("JSONL must contain at least one record")
        return count

    @staticmethod
    def _check_mapping(
        payload: dict[str, Any],
        required_keys: tuple[str, ...],
        forbidden_keys: tuple[str, ...],
        expected: dict[str, Any],
    ) -> None:
        missing = [key for key in required_keys if key not in payload]
        forbidden = [key for key in forbidden_keys if key in payload and payload[key] is not None]
        mismatched = [
            f"{key}={payload.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if payload.get(key) != value
        ]
        if missing:
            raise ValueError(f"missing keys {missing}")
        if forbidden:
            raise ValueError(f"forbidden non-null keys {forbidden}")
        if mismatched:
            raise ValueError("; ".join(mismatched))

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"line {line_number}: expected an object")
                yield line_number, payload

    def _validate_release_manifest(
        self, rules: tuple[ArtifactRule, ...], paths: dict[str, Path], state: GateState
    ) -> None:
        manifest_path = paths.get("release_manifest")
        if manifest_path is None or not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = manifest.get("artifacts", {})
            if not isinstance(artifacts, dict):
                raise ValueError("manifest.artifacts must be an object")
            for rule in rules:
                if rule.root != "release" or not rule.manifested:
                    continue
                path = paths.get(rule.artifact_id)
                if path is None or not path.is_file():
                    continue
                self._check_manifest_entry(rule, path, artifacts, state)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            state.errors.append(f"Cannot validate release manifest: {exc}")

    def _check_manifest_entry(
        self,
        rule: ArtifactRule,
        path: Path,
        artifacts: dict[str, Any],
        state: GateState,
    ) -> None:
        entry = artifacts.get(rule.path)
        if not isinstance(entry, dict):
            state.errors.append(f"Manifest lacks artifact entry: {rule.path}")
            return
        expected_hash = entry.get("sha256")
        actual_hash = self._sha256(path)
        if expected_hash != actual_hash:
            state.errors.append(f"Manifest hash mismatch: {rule.path}")
        if rule.kind == "jsonl" and "records" in entry:
            actual_records = state.record_counts.get(rule.artifact_id)
            if entry.get("records") != actual_records:
                state.errors.append(f"Manifest record count mismatch: {rule.path}")

    def _validate_split(self, paths: dict[str, Path], state: GateState) -> None:
        train_path = paths.get("qa_train")
        validation_path = paths.get("qa_validation")
        if (
            not train_path
            or not validation_path
            or not train_path.is_file()
            or not validation_path.is_file()
        ):
            return
        try:
            train_ids, train_groups = self._load_split_keys(train_path)
            validation_ids, validation_groups = self._load_split_keys(validation_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            state.errors.append(f"Cannot validate split disjointness: {exc}")
            return
        if len(train_ids) != state.record_counts.get("qa_train"):
            state.errors.append("Duplicate question_id inside training split.")
        if len(validation_ids) != state.record_counts.get("qa_validation"):
            state.errors.append("Duplicate question_id inside validation split.")
        overlap_ids = train_ids & validation_ids
        overlap_groups = train_groups & validation_groups
        if overlap_ids:
            state.errors.append(f"Train/validation question_id overlap: {len(overlap_ids)}")
        if overlap_groups:
            state.errors.append(f"Train/validation question_group overlap: {len(overlap_groups)}")

    @staticmethod
    def _load_split_keys(path: Path) -> tuple[set[str], set[str]]:
        ids: set[str] = set()
        groups: set[str] = set()
        for _, payload in PreflightGate._iter_jsonl(path):
            ids.add(str(payload["question_id"]))
            groups.add(str(payload["question_group"]))
        return ids, groups

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
