from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..core.io import read_jsonl, write_json
from .release import (
    CORPUS_TRANSFORM_VERSION,
    PARSER_VERSION,
    QA_SPLIT_VERSION,
    QA_TRAIN_VALIDATION_REQUIRED_FIELDS,
    QA_TRANSFORM_VERSION,
    RELEASE_SCHEMA,
    TASK_ID,
    _chunk_structure_bucket,
    _hash_values,
    _stable_json_bytes,
    _valid_logical_parent_id,
    sha256_bytes,
    sha256_file,
)

_BASE_REQUIRED = {
    "manifest.json",
    "data_report.json",
    "leakage_report.json",
    "split_manifest.json",
    "qa/train.jsonl",
    "qa/validation.jsonl",
    "tokenizer/tokenizer_report_qwen.json",
}
_CORPUS_REQUIRED = {
    "corpus/corpus_manifest.json",
    "corpus/raw_inventory.jsonl",
    "corpus/documents.jsonl",
    "corpus/chunks.jsonl",
    "corpus/quarantine.jsonl",
    "corpus/corpus_report.json",
    "diagnostics/citation_qrels.jsonl",
    "diagnostics/citation_match_report.json",
    "diagnostics/citation_manual_audit.jsonl",
}
_PUBLIC_REQUIRED = {"qa/public.jsonl", "qa/public_manifest.json"}


def release_contract() -> dict[str, Any]:
    return {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "profiles": {
            "e0-direct": sorted(_BASE_REQUIRED),
            "e1-bm25": sorted(_BASE_REQUIRED | _CORPUS_REQUIRED),
        },
        "public_additions": sorted(_PUBLIC_REQUIRED),
        "versions": {
            "qa_transform": QA_TRANSFORM_VERSION,
            "split": QA_SPLIT_VERSION,
            "corpus_transform": CORPUS_TRANSFORM_VERSION,
            "parser": PARSER_VERSION,
        },
        "exit_codes": {"pass": 0, "contract_failure": 2},
    }


def _load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"Invalid JSON {path}: {error}")
        return {}


def _required_paths(profile: str, stage: str) -> set[str]:
    if profile not in {"e0-direct", "e1-bm25"}:
        raise ValueError(f"Unsupported profile: {profile}")
    required = set(_BASE_REQUIRED)
    if profile == "e1-bm25":
        required.update(_CORPUS_REQUIRED)
    if stage == "public":
        required.update(_PUBLIC_REQUIRED)
    return required


def _validate_qa_records(
    root: Path,
    errors: list[str],
) -> tuple[set[str], set[str], set[str], set[str]]:
    split_values: dict[str, list[dict[str, Any]]] = {}
    required = set(QA_TRAIN_VALIDATION_REQUIRED_FIELDS)
    for split in ("train", "validation"):
        values = list(read_jsonl(root / "qa" / f"{split}.jsonl"))
        split_values[split] = values
        seen: set[str] = set()
        for line_number, value in enumerate(values, 1):
            missing = required - set(value)
            if missing:
                errors.append(f"qa/{split}.jsonl:{line_number} missing {sorted(missing)}")
                continue
            question_id = str(value["question_id"])
            if question_id in seen:
                errors.append(f"Duplicate {split} question_id={question_id}")
            seen.add(question_id)
            if value["source_task"] != TASK_ID:
                errors.append(f"Cross-task QA record {split}:{question_id}")
            if value["split"] != split:
                errors.append(f"Wrong split field {split}:{question_id}")
            if value["transform_version"] != QA_TRANSFORM_VERSION:
                errors.append(f"Wrong QA transform version {split}:{question_id}")
            if not isinstance(value["question_model"], str) or not value["question_model"].strip():
                errors.append(f"Empty question_model {split}:{question_id}")
            if not isinstance(value["answer_model"], str) or not value["answer_model"].strip():
                errors.append(f"Empty answer_model {split}:{question_id}")
    train_ids = {str(value.get("question_id")) for value in split_values["train"]}
    validation_ids = {
        str(value.get("question_id")) for value in split_values["validation"]
    }
    train_groups = {str(value.get("question_group")) for value in split_values["train"]}
    validation_groups = {
        str(value.get("question_group")) for value in split_values["validation"]
    }
    if train_ids & validation_ids:
        errors.append("Train/validation question IDs overlap")
    if train_groups & validation_groups:
        errors.append("Train/validation question groups overlap")
    return train_ids, validation_ids, train_groups, validation_groups


def _validate_public(root: Path, errors: list[str]) -> None:
    required = {
        "question_id",
        "source_task",
        "question_raw",
        "question_model",
        "split",
        "source_sha256",
        "transform_version",
    }
    seen: set[str] = set()
    for line_number, value in enumerate(read_jsonl(root / "qa" / "public.jsonl"), 1):
        missing = required - set(value)
        if missing:
            errors.append(f"qa/public.jsonl:{line_number} missing {sorted(missing)}")
            continue
        question_id = str(value["question_id"])
        if question_id in seen:
            errors.append(f"Duplicate public question_id={question_id}")
        seen.add(question_id)
        if value["source_task"] != TASK_ID or value["split"] != "public":
            errors.append(f"Invalid public provenance/split {question_id}")
        for key in ("answer", "answer_raw", "answer_model"):
            if key in value and value[key] is not None:
                errors.append(f"Public label visible at {question_id}:{key}")
        if not value["question_model"].strip():
            errors.append(f"Empty public question {question_id}")
    manifest = _load_json(root / "qa" / "public_manifest.json", errors)
    if manifest.get("answer_visibility") != "hidden":
        errors.append("public_manifest.answer_visibility must be hidden")
    if manifest.get("records") != len(seen):
        errors.append("public_manifest record count mismatch")
    if manifest.get("ids_sha256") != _hash_values(seen):
        errors.append("public_manifest IDs hash mismatch")


def _validate_corpus(
    root: Path,
    validation_ids: set[str],
    errors: list[str],
) -> tuple[int, int, int]:
    inventory_ids: set[str] = set()
    inventory_paths: set[str] = set()
    inventory_required = {
        "doc_id",
        "relative_path",
        "source_task",
        "sha256",
        "bytes",
        "json_valid",
    }
    for line_number, value in enumerate(
        read_jsonl(root / "corpus" / "raw_inventory.jsonl"), 1
    ):
        missing = inventory_required - set(value)
        if missing:
            errors.append(f"raw_inventory:{line_number} missing {sorted(missing)}")
            continue
        doc_id = str(value["doc_id"])
        relative_path = str(value["relative_path"])
        if doc_id in inventory_ids:
            errors.append(f"Duplicate inventory doc_id={doc_id}")
        if relative_path in inventory_paths:
            errors.append(f"Duplicate inventory relative_path={relative_path}")
        inventory_ids.add(doc_id)
        inventory_paths.add(relative_path)
        if value["source_task"] != TASK_ID:
            errors.append(f"Cross-task inventory record {doc_id}")

    documents: dict[str, tuple[str, str, bool]] = {}
    non_indexable_document_ids: set[str] = set()
    document_required = {
        "doc_id",
        "source_task",
        "name_raw",
        "name_derived",
        "name_source",
        "link_raw",
        "passage_raw",
        "source_sha256",
        "status",
        "indexable",
        "quality_flags",
        "transform_version",
    }
    for line_number, value in enumerate(read_jsonl(root / "corpus" / "documents.jsonl"), 1):
        missing = document_required - set(value)
        if missing:
            errors.append(f"documents:{line_number} missing {sorted(missing)}")
            continue
        doc_id = str(value["doc_id"])
        if doc_id in documents:
            errors.append(f"Duplicate document doc_id={doc_id}")
        documents[doc_id] = (
            str(value["passage_raw"]),
            str(value["source_sha256"]),
            bool(value["indexable"]),
        )
        if not value["indexable"]:
            non_indexable_document_ids.add(doc_id)
        if value["source_task"] != TASK_ID:
            errors.append(f"Cross-task document {doc_id}")
        if value["transform_version"] != CORPUS_TRANSFORM_VERSION:
            errors.append(f"Wrong document transform version {doc_id}")
    if set(documents) != inventory_ids:
        errors.append("Inventory/document ID coverage mismatch")

    chunk_ids: set[str] = set()
    chunk_total = 0
    indexable_chunks = 0
    chunk_levels: Counter[str] = Counter()
    chunk_structure_counts: Counter[str] = Counter()
    parent_doc_ids: dict[str, str] = {}
    orphan_parent_ids: set[str] = set()
    orphan_parent_references = 0
    empty_retrieval_text = 0
    source_offset_round_trip_failures = 0
    retrieval_fingerprints: set[str] = set()
    duplicate_retrieval_fingerprints: set[str] = set()
    duplicate_searchable_chunks = 0
    chunk_required = {
        "chunk_id",
        "doc_id",
        "source_task",
        "parent_chunk_id",
        "doc_type",
        "doc_number_surface",
        "doc_number_canonical",
        "chapter",
        "article",
        "clause",
        "point",
        "level",
        "raw_text",
        "canonical_text",
        "retrieval_text",
        "parent_text",
        "source_sha256",
        "source_start",
        "source_end",
        "parser_version",
        "transform_version",
        "indexable",
        "quality_flags",
    }
    for line_number, value in enumerate(read_jsonl(root / "corpus" / "chunks.jsonl"), 1):
        missing = chunk_required - set(value)
        if missing:
            errors.append(f"chunks:{line_number} missing {sorted(missing)}")
            continue
        chunk_total += 1
        chunk_id = str(value["chunk_id"])
        doc_id = str(value["doc_id"])
        if chunk_id in chunk_ids:
            errors.append(f"Duplicate chunk_id={chunk_id}")
        chunk_ids.add(chunk_id)
        document = documents.get(doc_id)
        if document is None:
            errors.append(f"Chunk references unknown doc_id={doc_id}")
            continue
        passage, source_sha, indexable = document
        level = str(value.get("level") or "")
        chunk_levels[level] += 1
        chunk_structure_counts[_chunk_structure_bucket(level)] += 1
        indexable_chunks += int(bool(value["indexable"]))
        retrieval_text = str(value["retrieval_text"])
        empty_retrieval_text += int(not retrieval_text.strip())
        fingerprint = sha256_bytes(retrieval_text.encode("utf-8"))
        if fingerprint in retrieval_fingerprints:
            duplicate_searchable_chunks += 1
            duplicate_retrieval_fingerprints.add(fingerprint)
        else:
            retrieval_fingerprints.add(fingerprint)
        parent_chunk_id = str(value["parent_chunk_id"])
        previous_parent_doc = parent_doc_ids.setdefault(parent_chunk_id, doc_id)
        if (
            previous_parent_doc != doc_id
            or not _valid_logical_parent_id(doc_id, parent_chunk_id)
        ):
            orphan_parent_ids.add(parent_chunk_id)
            orphan_parent_references += 1
        start = value["source_start"]
        end = value["source_end"]
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(passage)):
            errors.append(f"Invalid source span chunk_id={chunk_id}")
            source_offset_round_trip_failures += 1
        elif passage[start:end] != value["raw_text"]:
            errors.append(f"Raw span mismatch chunk_id={chunk_id}")
            source_offset_round_trip_failures += 1
        if source_sha != value["source_sha256"]:
            errors.append(f"Source checksum mismatch chunk_id={chunk_id}")
        if not indexable or not value["indexable"]:
            errors.append(f"Non-indexable document emitted chunk_id={chunk_id}")
        if value["source_task"] != TASK_ID:
            errors.append(f"Cross-task chunk {chunk_id}")
        if value["parser_version"] != PARSER_VERSION:
            errors.append(f"Wrong parser version chunk_id={chunk_id}")

    quarantine_count = 0
    quarantine_ids: set[str] = set()
    quarantine_reason_counts: Counter[str] = Counter()
    for line_number, value in enumerate(
        read_jsonl(root / "corpus" / "quarantine.jsonl"), 1
    ):
        quarantine_count += 1
        required = {"doc_id", "reason", "source_sha256", "included_in_index", "review_status"}
        missing = required - set(value)
        if missing:
            errors.append(f"quarantine:{line_number} missing {sorted(missing)}")
        if value.get("included_in_index") is not False:
            errors.append(f"Quarantine record included in index: {value.get('doc_id')}")
        doc_id = str(value.get("doc_id"))
        if doc_id in quarantine_ids:
            errors.append(f"Duplicate quarantine doc_id={doc_id}")
        quarantine_ids.add(doc_id)
        quarantine_reason_counts[str(value.get("reason") or "")] += 1
        if doc_id not in non_indexable_document_ids:
            errors.append(f"Quarantine record is not a non-indexable document: {doc_id}")
    if quarantine_ids != non_indexable_document_ids:
        missing = sorted(non_indexable_document_ids - quarantine_ids)
        extra = sorted(quarantine_ids - non_indexable_document_ids)
        errors.append(
            f"Quarantine/non-indexable coverage mismatch missing={missing} extra={extra}"
        )

    qrel_count = 0
    seen_qrels: set[tuple[str, str]] = set()
    matched_question_ids: set[str] = set()
    qrel_confidence_counts: Counter[str] = Counter()
    for line_number, value in enumerate(
        read_jsonl(root / "diagnostics" / "citation_qrels.jsonl"), 1
    ):
        qrel_count += 1
        required = {
            "question_id",
            "chunk_id",
            "relevance",
            "confidence",
            "match_reason",
            "qrels_status",
            "citation_parser_version",
            "matcher_version",
        }
        missing = required - set(value)
        if missing:
            errors.append(f"citation_qrels:{line_number} missing {sorted(missing)}")
            continue
        key = (str(value["question_id"]), str(value["chunk_id"]))
        if key in seen_qrels:
            errors.append(f"Duplicate qrel {key}")
        seen_qrels.add(key)
        if key[0] not in validation_ids:
            errors.append(f"Qrel outside validation split {key[0]}")
        if key[1] not in chunk_ids:
            errors.append(f"Qrel references unknown chunk {key[1]}")
        if value["qrels_status"] != "diagnostic_proxy":
            errors.append(f"Qrel is not diagnostic_proxy {key}")
        matched_question_ids.add(key[0])
        qrel_confidence_counts[str(value["confidence"])] += 1

    corpus_report = _load_json(root / "corpus" / "corpus_report.json", errors)
    if corpus_report.get("documents_total") != len(documents):
        errors.append("corpus_report document count mismatch")
    if corpus_report.get("chunk_count") != len(chunk_ids):
        errors.append("corpus_report chunk count mismatch")
    if corpus_report.get("quarantine_count") != quarantine_count:
        errors.append("corpus_report quarantine count mismatch")
    corpus_metric_expectations = {
        "documents_non_indexable": len(non_indexable_document_ids),
        "chunks_total": chunk_total,
        "chunks_indexable": indexable_chunks,
        "unique_chunk_id": len(chunk_ids),
        "duplicate_chunk_id": chunk_total - len(chunk_ids),
        "unique_parent_chunk_id": len(parent_doc_ids),
        "orphan_parent_chunk_ids": len(orphan_parent_ids),
        "orphan_parent_chunk_id_references": orphan_parent_references,
        "empty_retrieval_text": empty_retrieval_text,
        "source_offset_round_trip_failures": source_offset_round_trip_failures,
        "duplicate_searchable_chunks": duplicate_searchable_chunks,
        "duplicate_searchable_chunk_groups": len(duplicate_retrieval_fingerprints),
    }
    for field, expected in corpus_metric_expectations.items():
        if corpus_report.get(field) != expected:
            errors.append(f"corpus_report {field} mismatch")
    if corpus_report.get("chunk_levels") != dict(sorted(chunk_levels.items())):
        errors.append("corpus_report chunk_levels mismatch")
    if corpus_report.get("chunk_structure_counts") != dict(
        sorted(chunk_structure_counts.items())
    ):
        errors.append("corpus_report chunk_structure_counts mismatch")
    if corpus_report.get("quarantine_reason_counts") != dict(
        sorted(quarantine_reason_counts.items())
    ):
        errors.append("corpus_report quarantine_reason_counts mismatch")
    for name, expected in corpus_report.get("file_sizes_bytes", {}).items():
        path = root / "corpus" / name
        if not path.is_file() or path.stat().st_size != expected:
            errors.append(f"corpus_report file size mismatch: {name}")

    qrels_report = _load_json(
        root / "diagnostics" / "citation_match_report.json", errors
    )
    qrels_expectations = {
        "answers_scanned": len(validation_ids),
        "questions_with_at_least_one_match": len(matched_question_ids),
        "unique_question_chunk_pairs": len(seen_qrels),
        "qrel_records": qrel_count,
    }
    for field, expected in qrels_expectations.items():
        if qrels_report.get(field) != expected:
            errors.append(f"citation_match_report {field} mismatch")
    expected_confidence_counts = {
        confidence: qrel_confidence_counts[confidence]
        for confidence in ("HIGH", "MEDIUM", "LOW")
    }
    if qrels_report.get("qrel_pair_counts_by_confidence") != expected_confidence_counts:
        errors.append("citation_match_report confidence pair counts mismatch")
    coverage_basis = qrels_report.get("judged_coverage_basis", {})
    if (
        coverage_basis.get("numerator") != len(matched_question_ids)
        or coverage_basis.get("denominator") != len(validation_ids)
    ):
        errors.append("citation_match_report judged coverage basis mismatch")
    return len(documents), len(chunk_ids), qrel_count


def validate_data_release(
    release_root: str | Path,
    *,
    profile: str,
    stage: str,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(release_root)
    errors: list[str] = []
    warnings: list[str] = []
    required = _required_paths(profile, stage)
    missing_paths = sorted(path for path in required if not (root / path).is_file())
    errors.extend(f"Missing required artifact: {path}" for path in missing_paths)
    if missing_paths:
        report = {
            "schema_version": RELEASE_SCHEMA,
            "task_id": TASK_ID,
            "status": "FAIL",
            "profile": profile,
            "stage": stage,
            "errors": errors,
            "warnings": warnings,
        }
        if report_path:
            write_json(report_path, report)
        return report

    manifest = _load_json(root / "manifest.json", errors)
    if manifest.get("schema_version") != RELEASE_SCHEMA:
        errors.append("Invalid release schema_version")
    if manifest.get("task_id") != TASK_ID:
        errors.append("Invalid task_id")
    if manifest.get("status") != "READY":
        errors.append("Release manifest is not READY")
    artifacts = manifest.get("artifacts") or {}
    if "manifest.json" in artifacts:
        errors.append("Manifest must not hash itself")
    for relative, expected in artifacts.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"Manifest artifact missing: {relative}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != expected.get("sha256"):
            errors.append(f"Artifact SHA-256 mismatch: {relative}")
        if "bytes" in expected and path.stat().st_size != expected["bytes"]:
            errors.append(f"Artifact byte size mismatch: {relative}")
        if "records" in expected:
            with path.open("r", encoding="utf-8") as handle:
                actual_records = sum(1 for line in handle if line.strip())
            if actual_records != expected["records"]:
                errors.append(f"Artifact record count mismatch: {relative}")
    for relative in required - {"manifest.json"}:
        if relative not in artifacts:
            errors.append(f"Required artifact absent from manifest: {relative}")

    for report_name in ("data_report.json", "leakage_report.json", "split_manifest.json"):
        value = _load_json(root / report_name, errors)
        if value.get("status") != "PASS":
            errors.append(f"{report_name} does not PASS")
    leakage = _load_json(root / "leakage_report.json", errors)
    for key in ("cross_task_hits", "public_label_usage", "private_label_usage"):
        if leakage.get(key) != 0:
            errors.append(f"leakage_report.{key} must be zero")

    train_ids, validation_ids, train_groups, validation_groups = _validate_qa_records(
        root, errors
    )
    data_report = _load_json(root / "data_report.json", errors)
    qa_contract = data_report.get("train_validation_record_contract", {})
    if qa_contract.get("required_fields") != list(
        QA_TRAIN_VALIDATION_REQUIRED_FIELDS
    ):
        errors.append("data_report train/validation required-field contract mismatch")
    if qa_contract.get("records_checked") != len(train_ids) + len(validation_ids):
        errors.append("data_report train/validation contract count mismatch")
    if qa_contract.get("records_missing_required_fields") != 0:
        errors.append("data_report reports train/validation records with missing fields")
    split_manifest = _load_json(root / "split_manifest.json", errors)
    if split_manifest.get("train_ids_sha256") != _hash_values(train_ids):
        errors.append("Train ID hash mismatch")
    if split_manifest.get("validation_ids_sha256") != _hash_values(validation_ids):
        errors.append("Validation ID hash mismatch")
    if split_manifest.get("train_groups_sha256") != _hash_values(train_groups):
        errors.append("Train group hash mismatch")
    if split_manifest.get("validation_groups_sha256") != _hash_values(validation_groups):
        errors.append("Validation group hash mismatch")
    if split_manifest.get("split_version") != QA_SPLIT_VERSION:
        errors.append("Wrong split version")

    public_count = 0
    if stage == "public":
        _validate_public(root, errors)
        public_count = sum(1 for _ in read_jsonl(root / "qa" / "public.jsonl"))

    document_count = chunk_count = qrel_count = 0
    if profile == "e1-bm25":
        document_count, chunk_count, qrel_count = _validate_corpus(
            root, validation_ids, errors
        )
        for report_name in (
            "corpus/corpus_report.json",
            "diagnostics/citation_match_report.json",
        ):
            value = _load_json(root / report_name, errors)
            if value.get("status") != "PASS":
                errors.append(f"{report_name} does not PASS")
        audit_records = list(read_jsonl(root / "diagnostics" / "citation_manual_audit.jsonl"))
        pending = sum(value.get("review_status") != "APPROVED" for value in audit_records)
        if pending:
            message = f"Citation manual audit has {pending} pending records"
            if stage == "promotion":
                errors.append(message)
            else:
                warnings.append(message)

    tokenizer_report = _load_json(
        root / "tokenizer" / "tokenizer_report_qwen.json", errors
    )
    tokenizer_required = {
        "schema_version",
        "task_id",
        "release_id",
        "status",
        "model_id",
        "tokenizer_revision",
        "prompt_version",
        "question_tokens",
        "target_tokens",
        "retrieval_chunk_tokens",
        "packed_context_tokens",
        "total_sequence_tokens",
        "candidate_max_lengths",
        "truncation_counts",
        "sample_ids",
        "report_sha256",
    }
    missing_tokenizer = tokenizer_required - set(tokenizer_report)
    if missing_tokenizer:
        errors.append(f"Tokenizer report missing {sorted(missing_tokenizer)}")
    else:
        expected_report_hash = tokenizer_report.pop("report_sha256")
        if sha256_bytes(_stable_json_bytes(tokenizer_report)) != expected_report_hash:
            errors.append("Tokenizer report self-hash mismatch")
        if profile == "e1-bm25":
            corpus_report = _load_json(root / "corpus" / "corpus_report.json", errors)
            lengths = tokenizer_report["retrieval_chunk_tokens"]
            expected_lengths = {
                "tokenizer_model_id": tokenizer_report["model_id"],
                "tokenizer_revision": tokenizer_report["tokenizer_revision"],
                "count": lengths["count"],
                "p50": lengths["p50"],
                "p95": lengths["p95"],
                "max": lengths["max"],
            }
            if corpus_report.get("retrieval_text_token_lengths") != expected_lengths:
                errors.append("corpus_report retrieval token lengths mismatch")

    report = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": manifest.get("release_id"),
        "status": "PASS" if not errors else "FAIL",
        "profile": profile,
        "stage": stage,
        "checks": {
            "required_artifacts": len(required),
            "manifest_artifacts": len(artifacts),
            "train_records": len(train_ids),
            "validation_records": len(validation_ids),
            "public_records": public_count,
            "documents": document_count,
            "chunks": chunk_count,
            "citation_qrels": qrel_count,
        },
        "errors": errors,
        "warnings": warnings,
    }
    if report_path:
        write_json(report_path, report)
    return report
