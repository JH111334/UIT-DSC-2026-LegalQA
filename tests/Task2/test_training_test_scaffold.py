"""Contract tests for the fail-closed Task 2 Offline/Online scaffold."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "Source" / "Task2" / "pipeline.py"
SOURCE_API_ENTRYPOINT = ROOT / "SourceAPI" / "pipeline.py"
TEMP_PARENT = ROOT / ".tmp" / "training-test-scaffold"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _write_request(
    path: Path,
    *,
    operation: str,
    stage: str,
    release_root: Path,
    report: Path,
    profile: str = "e0-direct",
    scope: str = "release",
    control_root: Path | None = None,
    run_root: Path | None = None,
) -> None:
    _write_json(
        path,
        {
            "schema_version": "task2-pipeline-request-v1",
            "operation": operation,
            "stage": stage,
            "profile": profile,
            "scope": scope,
            "release_root": str(release_root),
            "control_root": str(control_root) if control_root else None,
            "run_root": str(run_root) if run_root else None,
            "report": str(report),
        },
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qa_record(question_id: str, group: str, split: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "source_task": "Task2",
        "question_raw": f"Câu hỏi {question_id}",
        "answer_raw": f"Câu trả lời {question_id}",
        "question_model": f"Câu hỏi {question_id}",
        "answer_model": f"Câu trả lời {question_id}",
        "question_group": group,
        "split": split,
        "source_sha256": "raw-sha",
        "transform_version": "task2-minimal-nfc-v1",
    }


def _refresh_manifest(release_root: Path) -> None:
    artifacts: dict[str, object] = {}
    for path in sorted(item for item in release_root.rglob("*") if item.is_file()):
        relative = path.relative_to(release_root).as_posix()
        if relative == "manifest.json":
            continue
        entry: dict[str, object] = {"sha256": _sha256(path)}
        if path.suffix == ".jsonl":
            entry["records"] = sum(
                1 for line in path.read_text(encoding="utf-8").splitlines() if line
            )
        artifacts[relative] = entry
    _write_json(
        release_root / "manifest.json",
        {
            "schema_version": "task2-data-release-v1",
            "task_id": "Task2",
            "release_id": "task2-data-v1",
            "status": "READY",
            "raw_sources": {},
            "artifacts": artifacts,
            "transform_version": "task2-minimal-nfc-v1",
            "split_version": "task2-group-split-v1",
            "parser_version": "task2-legal-parser-v1",
        },
    )


def _build_e0_release(release_root: Path) -> None:
    _write_json(
        release_root / "data_report.json",
        {
            "schema_version": "v1",
            "task_id": "Task2",
            "release_id": "task2-data-v1",
            "status": "PASS",
            "raw_counts": {},
            "processed_counts": {},
            "empty_counts": {},
            "duplicate_counts": {},
            "normalization": "NFC",
            "newline_preservation": True,
            "source_checksums": {},
            "transform_version": "task2-minimal-nfc-v1",
        },
    )
    _write_json(
        release_root / "leakage_report.json",
        {
            "schema_version": "v1",
            "task_id": "Task2",
            "release_id": "task2-data-v1",
            "status": "PASS",
            "exact_duplicate_groups": [],
            "near_duplicate_method": "fixture",
            "suspicious_pairs": [],
            "cross_task_hits": 0,
            "public_label_usage": 0,
            "private_label_usage": 0,
            "thresholds": {},
        },
    )
    _write_json(
        release_root / "split_manifest.json",
        {
            "schema_version": "v1",
            "task_id": "Task2",
            "release_id": "task2-data-v1",
            "status": "PASS",
            "seed": 2026,
            "method": "group-aware",
            "train_ids_sha256": "train-ids",
            "validation_ids_sha256": "validation-ids",
            "train_groups_sha256": "train-groups",
            "validation_groups_sha256": "validation-groups",
            "counts": {"train": 1, "validation": 1},
            "overlap": {"question_ids": 0, "question_groups": 0},
        },
    )
    _write_jsonl(release_root / "qa" / "train.jsonl", [_qa_record("q1", "g1", "train")])
    _write_jsonl(
        release_root / "qa" / "validation.jsonl",
        [_qa_record("q2", "g2", "validation")],
    )
    _refresh_manifest(release_root)


def _build_training_controls(control_root: Path, release_root: Path) -> None:
    payloads: dict[str, dict[str, object]] = {
        "approvals/architecture_approval.json": {
            "schema_version": "v1",
            "task_id": "Task2",
            "status": "APPROVED",
            "adr_id": "ADR-T2-0003",
            "approved_by": "fixture",
        },
        "approvals/preprocess_release_approval.json": {
            "schema_version": "v1",
            "task_id": "Task2",
            "status": "APPROVED",
            "release_id": "task2-data-v1",
            "release_manifest_sha256": _sha256(release_root / "manifest.json"),
            "preflight_report_sha256": "fixture",
            "approved_by": "fixture",
        },
        "approvals/environment_decision.json": {
            "schema_version": "v1",
            "task_id": "Task2",
            "status": "APPROVED",
            "environment": "local",
            "python_version": "3.12",
            "torch_build": "fixture",
            "dependency_lock_sha256": "fixture",
            "api_inference": False,
            "approved_by": "fixture",
        },
        "model/model_snapshot_manifest.json": {
            "schema_version": "v1",
            "task_id": "Task2",
            "status": "READY",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "model_revision": "fixture",
            "tokenizer_revision": "fixture",
            "parameter_count": 1540000000,
            "license": "fixture",
            "snapshot_path": "fixture",
            "snapshot_files": [],
        },
        "tokenizer/tokenizer_report_qwen.json": {
            "schema_version": "v1",
            "task_id": "Task2",
            "release_id": "task2-data-v1",
            "status": "PASS",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "model_revision": "fixture-revision",
            "tokenizer_revision": "fixture-revision",
            "tokenizer_files_sha256": "fixture-tokenizer-files",
            "prompt_version": "direct-v1",
            "question_tokens": {},
            "target_tokens": {},
            "retrieval_chunk_tokens": {},
            "packed_context_tokens": {},
            "total_sequence_tokens": {},
            "candidate_max_lengths": [],
            "truncation_counts": {},
            "candidate_max_new_tokens": [],
            "target_truncation_counts": {},
            "sample_ids": ["q1", "q2"],
        },
        "training/training_config.json": {
            "schema_version": "v1",
            "task_id": "Task2",
            "status": "APPROVED",
            "profile": "e0-direct",
            "seed": 2026,
            "answer_only_loss": True,
            "prompt_version": "direct-v1",
            "run_mode": "full",
            "max_sequence_length": 2048,
            "max_new_tokens": 1024,
        },
        "scorer/scorer_contract.json": {
            "schema_version": "v1",
            "task_id": "Task2",
            "status": "PROVISIONAL_APPROVED",
            "implementation": "fixture",
            "version": "fixture",
            "metric_order": ["METEOR", "ROUGE-L"],
            "parity_status": "provisional",
            "golden_cases_sha256": "fixture",
        },
    }
    for relative, payload in payloads.items():
        _write_json(control_root / relative, payload)


def _build_public_release_with_answer(release_root: Path) -> None:
    question_path = release_root / "qa" / "public.jsonl"
    record: dict[str, object] = {
        "question_id": "public-1",
        "source_task": "Task2",
        "question_raw": "Câu hỏi public",
        "question_model": "Câu hỏi public",
        "split": "public",
        "source_sha256": "raw-public",
        "transform_version": "task2-minimal-nfc-v1",
        "answer": "Không được xuất hiện",
    }
    _write_jsonl(question_path, [record])
    _write_json(
        release_root / "qa" / "public_manifest.json",
        {
            "schema_version": "v1",
            "task_id": "Task2",
            "release_id": "task2-data-v1",
            "status": "READY",
            "phase": "public",
            "source_sha256": "raw-public",
            "records": 1,
            "ids_sha256": "public-ids",
            "answer_visibility": "hidden",
            "transform_version": "task2-minimal-nfc-v1",
            "artifact_sha256": _sha256(question_path),
        },
    )
    _refresh_manifest(release_root)


class TrainingTestScaffoldTest(unittest.TestCase):
    """Prove contract inspection, rejection, acceptance, and stop behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        TEMP_PARENT.mkdir(parents=True, exist_ok=True)

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ENTRYPOINT), *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def test_contract_command_is_dependency_free(self) -> None:
        """The contract can be inspected without any ML dependency."""
        result = self._run("contract")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["task_id"], "Task2")

    def test_private_kaggle_contract_is_narrow_and_inspectable(self) -> None:
        """The remote entrypoint must expose only the approved private batch."""
        contract = subprocess.run(
            [sys.executable, str(SOURCE_API_ENTRYPOINT), "contract"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(contract.returncode, 0)
        payload = json.loads(contract.stdout)
        self.assertEqual(payload["status"], "APPROVED_PRIVATE_KAGGLE_BATCH")
        self.assertEqual(payload["task_id"], "Task2")
        self.assertFalse(payload["api_inference"])
        self.assertTrue(payload["private_kernel_required"])
        self.assertTrue(payload["private_dataset_required"])
        self.assertEqual(payload["kaggle_api_role"], "orchestration_only")

    def test_missing_release_fails_and_writes_report(self) -> None:
        """An empty handoff cannot pass E0 release preflight."""
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as temp_dir:
            root = Path(temp_dir)
            report = root / "report.json"
            request = root / "request.json"
            _write_request(
                request,
                operation="preflight",
                stage="training",
                release_root=root / "release",
                report=report,
            )
            result = self._run("run", "--request", str(request))
            self.assertEqual(result.returncode, 2)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "FAIL")
            self.assertTrue(any("manifest.json" in error for error in payload["errors"]))

    def test_valid_e0_release_passes(self) -> None:
        """A hashed, disjoint Task 2 QA release passes release-only preflight."""
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as temp_dir:
            root = Path(temp_dir)
            release = root / "release"
            report = root / "report.json"
            request = root / "request.json"
            _build_e0_release(release)
            _write_request(
                request,
                operation="preflight",
                stage="training",
                release_root=release,
                report=report,
            )
            result = self._run("run", "--request", str(request))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["status"], "PASS")

    def test_group_overlap_is_rejected(self) -> None:
        """Question-group leakage fails even when all files and hashes exist."""
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as temp_dir:
            root = Path(temp_dir)
            release = root / "release"
            report = root / "report.json"
            request = root / "request.json"
            _build_e0_release(release)
            _write_jsonl(
                release / "qa" / "validation.jsonl",
                [_qa_record("q2", "g1", "validation")],
            )
            _refresh_manifest(release)
            _write_request(
                request,
                operation="preflight",
                stage="training",
                release_root=release,
                report=report,
            )
            result = self._run("run", "--request", str(request))
            self.assertEqual(result.returncode, 2)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(any("question_group overlap" in error for error in payload["errors"]))

    def test_public_release_rejects_visible_answer(self) -> None:
        """Public prepared data cannot leak a non-null answer into inference."""
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as temp_dir:
            root = Path(temp_dir)
            release = root / "release"
            report = root / "report.json"
            request = root / "request.json"
            _build_public_release_with_answer(release)
            _write_request(
                request,
                operation="preflight",
                stage="public",
                release_root=release,
                report=report,
            )
            result = self._run("run", "--request", str(request))
            self.assertEqual(result.returncode, 2)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(any("forbidden non-null keys" in error for error in payload["errors"]))

    def test_full_training_preflight_passes(self) -> None:
        """A complete control fixture passes without importing model dependencies."""
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as temp_dir:
            root = Path(temp_dir)
            release = root / "release"
            control = root / "control"
            report = root / "report.json"
            request = root / "request.json"
            _build_e0_release(release)
            _build_training_controls(control, release)
            _write_request(
                request,
                operation="preflight",
                stage="training",
                release_root=release,
                control_root=control,
                report=report,
                scope="full",
            )
            result = self._run("run", "--request", str(request))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
