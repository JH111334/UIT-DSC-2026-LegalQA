from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ..core.config import PipelineConfig, artifact_dir, data_path
from ..core.io import iter_context_zip, read_jsonl, write_json
from ..core.normalize import text_fingerprint
from .audit import StreamingAuditPlan, audit_context_stream
from .chunking import FixedChunker, LegalAwareChunker
from .sanitation import PAYWALL_RE, CorpusSanitizer

ProgressCallback = Callable[[str], None]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_sanitation_summary(config: PipelineConfig) -> dict[str, object]:
    return {
        "version": config.sanitation.version,
        "documents_scanned": 0,
        "paywall_documents": 0,
        "paywall_occurrences": 0,
        "paywall_occurrences_after": 0,
        "raw_chars": 0,
        "normalized_chars": 0,
        "generation_chars": 0,
        "retrieval_chars": 0,
        "exact_fragments_removed": 0,
        "exact_blocks_removed": 0,
        "overlap_tokens_removed": 0,
        "terminal_noi_nhan_blocks_removed": 0,
        "retrieval_boilerplate_lines_removed": 0,
        "web_artifact_lines_removed": 0,
        "web_artifact_chars_removed": 0,
        "review_candidate_matches": {},
        "paywall_document_details": [],
    }


def _accumulate_sanitation(summary: dict[str, object], document: object) -> None:
    metrics = document.metadata.get("sanitation", {})  # type: ignore[attr-defined]
    summary["documents_scanned"] = int(summary["documents_scanned"]) + 1
    for key in (
        "raw_chars",
        "normalized_chars",
        "generation_chars",
        "retrieval_chars",
        "paywall_count",
        "exact_fragments_removed",
        "exact_blocks_removed",
        "overlap_tokens_removed",
        "retrieval_boilerplate_lines_removed",
        "web_artifact_lines_removed",
        "web_artifact_chars_removed",
    ):
        target = "paywall_occurrences" if key == "paywall_count" else key
        summary[target] = int(summary[target]) + int(metrics.get(key, 0))
    summary["paywall_occurrences_after"] = int(summary["paywall_occurrences_after"]) + len(
        PAYWALL_RE.findall(document.generation_text)  # type: ignore[attr-defined]
    )
    if metrics.get("noi_nhan_removed"):
        summary["terminal_noi_nhan_blocks_removed"] = int(summary["terminal_noi_nhan_blocks_removed"]) + 1
    review_totals = summary["review_candidate_matches"]
    for pattern, count in metrics.get("review_candidate_matches", {}).items():
        review_totals[pattern] = int(review_totals.get(pattern, 0)) + int(count)
    if metrics.get("paywall_count"):
        summary["paywall_documents"] = int(summary["paywall_documents"]) + 1
        summary["paywall_document_details"].append(
            {
                "document_id": document.id,  # type: ignore[attr-defined]
                **{
                    key: metrics.get(key)
                    for key in (
                        "raw_chars",
                        "generation_chars",
                        "retrieval_chars",
                        "paywall_count",
                        "exact_fragments_removed",
                        "exact_blocks_removed",
                        "overlap_tokens_removed",
                        "web_artifact_lines_removed",
                        "web_artifact_chars_removed",
                        "repetition_ratio",
                        "post_repetition_ratio",
                        "max_block_frequency",
                        "post_max_block_frequency",
                    )
                },
            }
        )


def _finalize_sanitation_summary(summary: dict[str, object]) -> None:
    raw_chars = max(1, int(summary["raw_chars"]))
    summary["generation_char_reduction_ratio"] = round(
        1.0 - int(summary["generation_chars"]) / raw_chars, 6
    )
    summary["retrieval_char_reduction_ratio"] = round(
        1.0 - int(summary["retrieval_chars"]) / raw_chars, 6
    )


def deduplicate_jsonl_by_id(path: str | Path) -> dict[str, int]:
    """Mechanically collapse repeated stable IDs while preserving first occurrence."""

    source = Path(path)
    temporary = source.with_suffix(source.suffix + ".dedup.tmp")
    seen: set[str] = set()
    kept = 0
    removed = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for value in read_jsonl(source):
            record_id = str(value["id"])
            if record_id in seen:
                removed += 1
                continue
            seen.add(record_id)
            output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
    temporary.replace(source)
    return {"kept": kept, "duplicates_removed": removed}


def build_corpus(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    reuse_audit: bool = False,
) -> dict[str, object]:
    notify = progress or (lambda message: None)
    source = data_path(config, config.data.contexts_zip)
    output = artifact_dir(config)
    output.mkdir(parents=True, exist_ok=True)

    notify(f"Loading {source}")
    sanitizer = CorpusSanitizer(config.sanitation) if config.sanitation.enabled else None
    sanitation_report_path = output / "sanitation_report.json"
    sanitation_summary = _new_sanitation_summary(config)

    def iter_documents(*, track_sanitation: bool = False):
        for value in iter_context_zip(source):
            if sanitizer is not None:
                value = sanitizer.sanitize(value)
                if track_sanitation:
                    _accumulate_sanitation(sanitation_summary, value)
            yield value

    reusable_documents = output / "documents.jsonl"
    if reuse_audit and reusable_documents.exists():
        if config.sanitation.enabled:
            if not sanitation_report_path.exists():
                raise ValueError("Cannot reuse audit: sanitation_report.json is missing")
            previous_sanitation = json.loads(sanitation_report_path.read_text(encoding="utf-8"))
            if previous_sanitation.get("version") != config.sanitation.version:
                raise ValueError("Cannot reuse audit across different sanitation versions")
            sanitation_summary = _new_sanitation_summary(config)
        notify("Reusing verified audit metadata; hierarchy/chunks will be rebuilt")
        canonical_ids: set[str] = set()
        source_ids_by_canonical: dict[str, tuple[str, ...]] = {}
        flags_by_id: dict[str, tuple[str, ...]] = {}
        metadata_by_id: dict[str, dict[str, object]] = {}
        for value in read_jsonl(reusable_documents):
            document_id = str(value["id"])
            canonical_ids.add(document_id)
            source_ids_by_canonical[document_id] = tuple(str(x) for x in value.get("source_ids", [document_id]))
            flags_by_id[document_id] = tuple(str(x) for x in value.get("flags", []))
            metadata = dict(value.get("metadata") or {})
            # Sanitation is recomputed from source on every build pass so a
            # code-level fix cannot be overwritten by stale document metadata.
            metadata.pop("sanitation", None)
            metadata_by_id[document_id] = metadata
        audit = StreamingAuditPlan(
            canonical_ids=canonical_ids,
            source_ids_by_canonical=source_ids_by_canonical,
            flags_by_id=flags_by_id,
            metadata_by_id=metadata_by_id,
            report=json.loads((output / "audit_report.json").read_text(encoding="utf-8")),
            exact_duplicate_groups=json.loads((output / "exact_duplicate_groups.json").read_text(encoding="utf-8")),
            near_duplicate_groups=json.loads((output / "near_duplicate_groups.json").read_text(encoding="utf-8")),
        )
    else:
        notify("Audit pass 1/2: integrity, exact duplicate and near-duplicate signatures")
        audit = audit_context_stream(iter_documents(track_sanitation=True), config.audit)
        if sanitizer is not None:
            _finalize_sanitation_summary(sanitation_summary)
            write_json(sanitation_report_path, sanitation_summary)
    write_json(output / "audit_report.json", audit.report)
    write_json(output / "exact_duplicate_groups.json", audit.exact_duplicate_groups)
    write_json(output / "near_duplicate_groups.json", audit.near_duplicate_groups)

    chunker = LegalAwareChunker(config.chunking) if config.chunking.legal_aware else FixedChunker(config.chunking)
    chunk_fingerprints: set[str] = set()
    global_chunk_duplicates = 0
    global_parent_duplicates = 0
    within_document_exact_chunk_duplicates = 0
    within_document_near_chunk_duplicates = 0
    level_counts: Counter[str] = Counter()
    parent_level_counts: Counter[str] = Counter()
    chunk_count = 0
    parent_count = 0
    parsed_count = 0
    parent_ids: set[str] = set()
    notify("Build pass: normalized documents, hierarchy and parent-child chunks")
    with (
        (output / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as document_handle,
        (output / "parents.jsonl").open("w", encoding="utf-8", newline="\n") as parent_handle,
        (output / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as chunk_handle,
    ):
        for document in iter_documents(track_sanitation=bool(sanitizer is not None and reuse_audit)):
            if document.id not in audit.canonical_ids:
                continue
            document.source_ids = audit.source_ids_by_canonical[document.id]
            document.flags = audit.flags_by_id[document.id]
            document.metadata.update(audit.metadata_by_id[document.id])
            result = chunker.chunk_document(document)
            within_document_exact_chunk_duplicates += result.exact_duplicates_removed
            within_document_near_chunk_duplicates += result.near_duplicates_removed
            document_handle.write(json.dumps(document.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
            for parent in result.parents:
                if parent.id in parent_ids:
                    global_parent_duplicates += 1
                    continue
                parent_ids.add(parent.id)
                parent_handle.write(json.dumps(parent.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
                parent_level_counts[parent.level] += 1
                parent_count += 1
            for chunk in result.chunks:
                fingerprint = text_fingerprint(chunk.retrieval_text)
                if fingerprint in chunk_fingerprints:
                    global_chunk_duplicates += 1
                    continue
                chunk_fingerprints.add(fingerprint)
                chunk_handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
                level_counts[chunk.level] += 1
                chunk_count += 1
            parsed_count += 1
            if parsed_count % 500 == 0:
                notify(f"Parsed {parsed_count}/{len(audit.canonical_ids)} documents")
    if sanitizer is not None and reuse_audit:
        _finalize_sanitation_summary(sanitation_summary)
        write_json(sanitation_report_path, sanitation_summary)
    report: dict[str, object] = {
        "documents": len(audit.canonical_ids),
        "parents": parent_count,
        "chunks": chunk_count,
        "global_exact_chunk_duplicates_removed": global_chunk_duplicates,
        "global_exact_parent_duplicates_removed": global_parent_duplicates,
        "within_document_exact_chunk_duplicates_removed": within_document_exact_chunk_duplicates,
        "within_document_near_chunk_duplicates_removed": within_document_near_chunk_duplicates,
        "chunk_levels": dict(sorted(level_counts.items())),
        "parent_levels": dict(sorted(parent_level_counts.items())),
        "legal_aware": config.chunking.legal_aware,
        "chunk_max_chars": config.chunking.max_chars,
        "chunk_overlap_chars": config.chunking.overlap_chars,
        "corpus_source": str(source),
        "sanitation_enabled": config.sanitation.enabled,
        "sanitation_version": config.sanitation.version if config.sanitation.enabled else None,
    }
    write_json(output / "build_report.json", report)
    manifest = {
        "corpus_version": config.sanitation.version if config.sanitation.enabled else "v1",
        "frozen": False,
        "source": str(source),
        "source_sha256": _sha256(source),
        "config": config.to_dict(),
        "rules": {
            "boilerplate_blacklist_sha256": _sha256(config.sanitation.blacklist_path)
            if config.sanitation.enabled
            else None,
            "protected_patterns_sha256": _sha256(config.sanitation.protected_patterns_path)
            if config.sanitation.enabled
            else None,
        },
        "artifacts": {
            "documents": "documents.jsonl",
            "parents": "parents.jsonl",
            "chunks": "chunks.jsonl",
            "audit_report": "audit_report.json",
            "sanitation_report": "sanitation_report.json" if config.sanitation.enabled else None,
        },
    }
    write_json(output / "corpus_manifest.json", manifest)
    notify(f"Built {chunk_count} child chunks and {parent_count} parents")
    return report
