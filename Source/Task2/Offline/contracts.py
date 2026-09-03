"""Load and normalize the machine-readable Task 2 artifact contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTRACT_PATH = Path(__file__).resolve().parent / "required_artifacts.json"


class ContractError(RuntimeError):
    """Raised when the checked-in artifact contract is invalid."""


@dataclass(frozen=True)
class ArtifactRule:
    """Validation rule for one artifact path."""

    artifact_id: str
    root: str
    path: str
    kind: str
    manifested: bool
    allow_empty: bool
    required_keys: tuple[str, ...]
    expected: dict[str, Any]
    record_required_keys: tuple[str, ...]
    record_forbidden_keys: tuple[str, ...]
    record_expected: dict[str, Any]

    @classmethod
    def from_payload(cls, artifact_id: str, payload: dict[str, Any]) -> ArtifactRule:
        """Build a rule from checked-in JSON."""
        return cls(
            artifact_id=artifact_id,
            root=str(payload.get("root", "")),
            path=str(payload.get("path", "")),
            kind=str(payload.get("kind", "")),
            manifested=bool(payload.get("manifested", True)),
            allow_empty=bool(payload.get("allow_empty", False)),
            required_keys=tuple(str(item) for item in payload.get("required_keys", [])),
            expected=dict(payload.get("expected", {})),
            record_required_keys=tuple(
                str(item) for item in payload.get("record_required_keys", [])
            ),
            record_forbidden_keys=tuple(
                str(item) for item in payload.get("record_forbidden_keys", [])
            ),
            record_expected=dict(payload.get("record_expected", {})),
        )


@dataclass(frozen=True)
class ContractSpec:
    """Normalized artifact groups and stage/profile requirements."""

    raw: dict[str, Any]
    groups: dict[str, tuple[str, ...]]
    requirements: dict[str, dict[str, tuple[str, ...]]]
    artifacts: dict[str, ArtifactRule]

    def rules_for(self, stage: str, profile: str, scope: str) -> tuple[ArtifactRule, ...]:
        """Return de-duplicated rules required by a stage and profile."""
        stage_requirements = self.requirements.get(stage, {})
        group_ids = stage_requirements.get(profile)
        if group_ids is None:
            raise ContractError(f"Unsupported stage/profile: {stage}/{profile}")

        artifact_ids: list[str] = []
        for group_id in group_ids:
            if group_id not in self.groups:
                raise ContractError(f"Unknown artifact group: {group_id}")
            artifact_ids.extend(self.groups[group_id])

        rules: list[ArtifactRule] = []
        seen: set[str] = set()
        for artifact_id in artifact_ids:
            if artifact_id in seen:
                continue
            if artifact_id not in self.artifacts:
                raise ContractError(f"Unknown artifact rule: {artifact_id}")
            rule = self.artifacts[artifact_id]
            if scope == "release" and rule.root != "release":
                continue
            rules.append(rule)
            seen.add(artifact_id)
        return tuple(rules)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Contract must be a JSON object: {path}")
    return payload


def load_contract(path: Path = CONTRACT_PATH) -> ContractSpec:
    """Load the checked-in artifact contract and validate its top-level shape."""
    raw = _load_object(path)
    groups_payload = raw.get("groups")
    requirements_payload = raw.get("requirements")
    artifacts_payload = raw.get("artifacts")
    if not isinstance(groups_payload, dict):
        raise ContractError("Contract needs a groups object.")
    if not isinstance(requirements_payload, dict):
        raise ContractError("Contract needs a requirements object.")
    if not isinstance(artifacts_payload, dict):
        raise ContractError("Contract needs an artifacts object.")

    groups = {
        str(group_id): tuple(str(item) for item in members)
        for group_id, members in groups_payload.items()
        if isinstance(members, list)
    }
    requirements: dict[str, dict[str, tuple[str, ...]]] = {}
    for stage, profiles in requirements_payload.items():
        if not isinstance(profiles, dict):
            raise ContractError(f"Requirements for {stage} must be an object.")
        requirements[str(stage)] = {
            str(profile): tuple(str(item) for item in group_ids)
            for profile, group_ids in profiles.items()
            if isinstance(group_ids, list)
        }
    artifacts = {
        str(artifact_id): ArtifactRule.from_payload(str(artifact_id), payload)
        for artifact_id, payload in artifacts_payload.items()
        if isinstance(payload, dict)
    }
    if len(artifacts) != len(artifacts_payload):
        raise ContractError("Every artifact rule must be a JSON object.")
    return ContractSpec(raw=raw, groups=groups, requirements=requirements, artifacts=artifacts)
