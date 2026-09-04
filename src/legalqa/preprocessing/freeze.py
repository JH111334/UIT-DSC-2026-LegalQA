from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..core.config import PipelineConfig, artifact_dir
from ..core.io import write_json

_HARD_WEB_MARKERS = (
    b"TVPL Pro",
    b"function LawNote",
    b"cldivContentDocEn",
    b"isTCVNFree",
    "Bạn Chưa Đăng Nhập Thành Viên!".encode(),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_marker(path: Path) -> str | None:
    tail = b""
    max_marker = max(len(value) for value in _HARD_WEB_MARKERS)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value = tail + block
            for marker in _HARD_WEB_MARKERS:
                if marker.lower() in value.lower():
                    return marker.decode("utf-8")
            tail = value[-max_marker:]
    return None


def freeze_corpus(config: PipelineConfig) -> dict[str, object]:
    output = artifact_dir(config)
    required = (
        "documents.jsonl",
        "parents.jsonl",
        "chunks.jsonl",
        "audit_report.json",
        "sanitation_report.json",
        "sanitation_audit_report.json",
        "boilerplate_review_report.json",
        "diagnostic_report.json",
        "diagnostic_labels.jsonl",
        "citation_graph.json",
        "corpus.sqlite",
        "retrieval_index_report.json",
        "retrieval_diagnostic_metrics.json",
        "corpus_manifest.json",
    )
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze; missing artifacts: {missing}")

    build = json.loads((output / "build_report.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "audit_report.json").read_text(encoding="utf-8"))
    sanitation = json.loads((output / "sanitation_report.json").read_text(encoding="utf-8"))
    review = json.loads((output / "boilerplate_review_report.json").read_text(encoding="utf-8"))
    index = json.loads((output / "retrieval_index_report.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output / "diagnostic_report.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "retrieval_diagnostic_metrics.json").read_text(encoding="utf-8"))

    checks = {
        "zero_empty_passages": audit.get("empty_removed") == 20 and build.get("documents") == 8507,
        "exact_document_dedup_done": audit.get("exact_duplicates_removed") == 5,
        "zero_paywall_banners_after": sanitation.get("paywall_occurrences_after") == 0,
        "five_paywall_documents_audited": sanitation.get("paywall_documents") == 5,
        "high_df_candidates_reviewed": review.get("pending") == 0 and review.get("reviewed") == review.get("candidates"),
        "diagnostic_labels_regenerated": diagnostics.get("examples") == 7000,
        "bm25_rebenchmarked": metrics.get("evaluated", 0) > 0,
        "index_matches_build": index.get("chunks") == build.get("chunks") and index.get("parents") == build.get("parents"),
        "sanitation_version_matches": sanitation.get("version") == config.sanitation.version,
    }
    residual_markers = {
        name: _contains_marker(output / name) for name in ("parents.jsonl", "chunks.jsonl")
    }
    checks["zero_hard_web_markers"] = not any(residual_markers.values())
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Cannot freeze; failed checks: {failed}; residuals={residual_markers}")

    artifact_names = ("documents.jsonl", "parents.jsonl", "chunks.jsonl", "corpus.sqlite")
    fingerprints = {
        name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
        for name in artifact_names
    }
    manifest_path = output / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "frozen": True,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_checks": checks,
            "residual_web_markers": residual_markers,
            "artifact_fingerprints": fingerprints,
            "baseline_metrics": metrics,
            "review_policy_sha256": _sha256(Path(config.sanitation.review_policy_path)),
        }
    )
    write_json(manifest_path, manifest)
    report = {
        "frozen": True,
        "corpus_version": config.sanitation.version,
        "checks": checks,
        "artifact_fingerprints": fingerprints,
    }
    write_json(output / "freeze_report.json", report)
    return report
