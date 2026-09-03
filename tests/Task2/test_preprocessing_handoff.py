"""Small replay tests for the imported Task 2 preprocessing handoff."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "Source" / "Task2"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from Offline.core.config import DataConfig, DiagnosticConfig, PipelineConfig  # noqa: E402
from Offline.core.io import (  # noqa: E402
    context_source_count,
    context_source_sha256,
    iter_context_payloads,
)
from Offline.release_audit import validate_data_release  # noqa: E402
from Offline.release_builder import build_data_release  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_context_directory_iteration_and_hash_are_deterministic(tmp_path: Path) -> None:
    """A local organizer directory must behave like the supported ZIP source."""
    contexts = tmp_path / "selected-contexts"
    _write_json(contexts / "context_2.json", {"id": 2, "passage": "Hai"})
    _write_json(contexts / "context_1.json", {"id": 1, "passage": "Một"})

    names = [name for name, _payload in iter_context_payloads(contexts)]

    assert names == ["context_1.json", "context_2.json"]
    assert context_source_count(contexts) == 2
    assert context_source_sha256(contexts) == context_source_sha256(contexts)


def test_small_release_build_and_deep_audit(tmp_path: Path) -> None:
    """The builder must emit the canonical aliases consumed by the E1 gate."""
    raw = tmp_path / "raw"
    _write_json(
        raw / "train.json",
        {
            "q1": {
                "question": "Điều kiện thứ nhất là gì?",
                "answer": "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP.\nNội dung một.",
            },
            "q2": {
                "question": "Điều kiện thứ hai là gì?",
                "answer": "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP.\nNội dung hai.",
            },
        },
    )
    _write_json(
        raw / "public-official.json",
        {"p1": {"question": "Câu hỏi Public", "answer": None}},
    )
    _write_json(
        raw / "selected-contexts" / "context_90.json",
        {
            "id": 90,
            "name": "Nghị định 90/2017/NĐ-CP",
            "link": "",
            "passage": "Nghị định số 90/2017/NĐ-CP\nĐiều 17. Quy định\n1. Nội dung.",
        },
    )
    release = tmp_path / "task2-data-v1"
    config = PipelineConfig(
        data=DataConfig(data_dir=str(raw)),
        diagnostic=DiagnosticConfig(manual_sample_size=0),
    )

    result = build_data_release(config, release, validation_size=1)
    report = json.loads((release / "corpus" / "corpus_report.json").read_text(encoding="utf-8"))
    audit = validate_data_release(
        release,
        profile="e1-bm25",
        stage="evaluation",
    )

    assert result["status"] == "READY"
    assert report["chunk_count"] == sum(report["chunk_level_counts"].values())
    assert report["source_span_failures"] == 0
    assert report["chunk_token_stats"]["status"] == "NOT_RUN"
    assert audit["status"] == "PASS"


def test_release_identity_must_match_directory(tmp_path: Path) -> None:
    """Versioned paths and manifest identities must never silently diverge."""
    release = tmp_path / "task2-data-v2"
    release.mkdir()
    (release / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "task2-data-release-v1",
                "task_id": "Task2",
                "release_id": "task2-data-v1",
                "status": "READY",
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )

    audit = validate_data_release(release, profile="e0-direct", stage="training")

    assert audit["status"] == "FAIL"
