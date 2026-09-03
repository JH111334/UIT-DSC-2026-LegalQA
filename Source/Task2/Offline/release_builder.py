from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .core.config import PipelineConfig, data_path
from .core.io import (
    context_source_count,
    context_source_sha256,
    iter_context_payloads,
    read_jsonl,
    recover_name_from_link,
    write_json,
    write_jsonl,
)
from .core.metadata import (
    canonical_document_number,
    extract_citations,
    extract_document_metadata,
)
from .core.normalize import normalize_text
from .Corpus.sanitation import PAYWALL_RE

RELEASE_SCHEMA = "task2-data-release-v1"
TASK_ID = "Task2"
QA_TRANSFORM_VERSION = "task2-minimal-nfc-v1"
QA_SPLIT_VERSION = "task2-group-split-v1"
CORPUS_TRANSFORM_VERSION = "task2-corpus-v1"
PARSER_VERSION = "task2-legal-parser-v1"
CITATION_PARSER_VERSION = "answer-citation-v1"
CITATION_MATCHER_VERSION = "citation-corpus-match-v1"
QA_TRAIN_VALIDATION_REQUIRED_FIELDS = (
    "question_id",
    "source_task",
    "question_raw",
    "answer_raw",
    "question_model",
    "answer_model",
    "question_group",
    "split",
    "source_sha256",
    "transform_version",
)

_ARTICLE_RE = re.compile(r"(?im)^[ \t\u00a0]*điều\s+(?P<number>\d+[a-zđ]?)(?:\s*[.:/-])?[^\r\n]*")
_CHAPTER_RE = re.compile(r"(?im)^[ \t\u00a0]*(chương\s+[^\r\n]+)")
_CLAUSE_RE = re.compile(r"(?im)^[ \t\u00a0]*(?P<number>\d+)[.)][ \t\u00a0]+")
_POINT_RE = re.compile(r"(?im)^[ \t\u00a0]*(?P<number>[a-zđ])[.)][ \t\u00a0]+")
_DOCUMENT_NUMBER_SURFACE_RE = re.compile(
    r"\b\d{1,5}\s*/\s*(?:\d{4}|[A-ZĐ]{1,8})\s*/\s*[A-ZĐ0-9 -]{2,30}\b", re.I
)
_GROUP_DOCUMENT_RE = re.compile(
    r"\b\d{1,5}\s*/\s*(?:\d{4}|[a-zđ]{1,8})\s*/\s*[a-zđ0-9 -]{2,30}\b", re.I
)
_GROUP_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_GROUP_NON_WORD_RE = re.compile(r"[^0-9a-zà-ỹđ<>]+", re.I)
_CONTEXT_ID_RE = re.compile(r"context_([^/\\]+)\.json$", re.I)
_RELEASE_ID_RE = re.compile(r"task2-data-v\d+")


def _validated_release_id(release_root: Path, release_id: str | None = None) -> str:
    """Return one explicit release identity that agrees with its directory."""

    value = release_id or release_root.name
    if not _RELEASE_ID_RE.fullmatch(value):
        raise ValueError("release_id must match task2-data-v<integer>")
    if release_root.name != value:
        raise ValueError(
            f"release_id {value!r} must match release directory {release_root.name!r}"
        )
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _minimal_text(value: Any, *, preserve_newlines: bool) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if preserve_newlines:
        return text.strip()
    return re.sub(r"\s+", " ", text).strip()


def _question_group_key(question: str) -> str:
    value = unicodedata.normalize("NFC", question).casefold()
    value = _GROUP_DOCUMENT_RE.sub(" <docnum> ", value)
    value = _GROUP_NUMBER_RE.sub(" <num> ", value)
    return re.sub(r"\s+", " ", _GROUP_NON_WORD_RE.sub(" ", value)).strip()


def _question_group_id(question: str) -> str:
    digest = sha256_bytes(_question_group_key(question).encode("utf-8"))[:12].upper()
    return f"QG_{digest}"


def _record_source_sha(question_id: str, value: dict[str, Any]) -> str:
    return sha256_bytes(_stable_json_bytes({question_id: value}))


def _load_raw_qa(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    result: dict[str, dict[str, Any]] = {}
    for question_id, record in value.items():
        if not isinstance(record, dict):
            raise ValueError(f"Question {question_id!r} must be an object")
        result[str(question_id)] = record
    return result


def _split_groups(
    records: list[dict[str, Any]], validation_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < validation_size < len(records):
        raise ValueError("validation_size must leave at least one training record")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["question_group"]].append(record)
    ordered = sorted(
        groups.items(),
        key=lambda value: sha256_bytes(f"{seed}\0{value[0]}".encode()),
    )
    validation: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    remaining_target = validation_size
    for _group_id, values in ordered:
        if remaining_target > 0 and (
            len(values) <= remaining_target
            or abs(remaining_target - len(values)) <= remaining_target
        ):
            validation.extend(values)
            remaining_target -= len(values)
        else:
            training.extend(values)
    # Singleton groups make an exact target possible for this dataset. If a
    # larger group crossed the target, keep the group intact and report the
    # actual deterministic size rather than leaking it across splits.
    if not training:
        raise ValueError("Group-aware split left no training records")
    for record in training:
        record["split"] = "train"
    for record in validation:
        record["split"] = "validation"
    return training, validation


def build_qa_release(
    config: PipelineConfig,
    release_root: Path,
    *,
    validation_size: int,
    release_id: str,
) -> dict[str, Any]:
    qa_dir = release_root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    train_source = data_path(config, config.data.train_file)
    public_source = data_path(config, config.data.test_file)
    raw_train = _load_raw_qa(train_source)
    raw_public = _load_raw_qa(public_source)

    all_records: list[dict[str, Any]] = []
    nfc_question_changes = 0
    nfc_answer_changes = 0
    raw_answer_newlines = 0
    model_answer_newlines = 0
    empty_questions = 0
    empty_answers = 0
    exact_questions: dict[str, list[str]] = defaultdict(list)
    near_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for question_id, value in raw_train.items():
        question_raw = "" if value.get("question") is None else str(value["question"])
        answer_raw = "" if value.get("answer") is None else str(value["answer"])
        question_model = _minimal_text(question_raw, preserve_newlines=False)
        answer_model = _minimal_text(answer_raw, preserve_newlines=True)
        group = _question_group_id(question_model)
        record = {
            "question_id": question_id,
            "source_task": TASK_ID,
            "question_raw": question_raw,
            "answer_raw": answer_raw,
            "question_model": question_model,
            "answer_model": answer_model,
            "question_group": group,
            "split": "",
            "source_sha256": _record_source_sha(question_id, value),
            "transform_version": QA_TRANSFORM_VERSION,
        }
        all_records.append(record)
        nfc_question_changes += int(question_raw != question_model)
        nfc_answer_changes += int(answer_raw != answer_model)
        raw_answer_newlines += question_raw.count("\n") + answer_raw.count("\n")
        model_answer_newlines += question_model.count("\n") + answer_model.count("\n")
        empty_questions += int(not question_model)
        empty_answers += int(not answer_model)
        exact_questions[question_model.casefold()].append(question_id)
        near_groups[group].append((question_id, question_model.casefold()))

    training, validation = _split_groups(all_records, validation_size, config.seed)
    write_jsonl(qa_dir / "train.jsonl", training)
    write_jsonl(qa_dir / "validation.jsonl", validation)

    public_records = []
    public_label_usage = 0
    for question_id, value in raw_public.items():
        question_raw = "" if value.get("question") is None else str(value["question"])
        public_label_usage += int(value.get("answer") not in (None, ""))
        public_records.append(
            {
                "question_id": question_id,
                "source_task": TASK_ID,
                "question_raw": question_raw,
                "question_model": _minimal_text(question_raw, preserve_newlines=False),
                "split": "public",
                "source_sha256": _record_source_sha(question_id, value),
                "transform_version": QA_TRANSFORM_VERSION,
            }
        )
    write_jsonl(qa_dir / "public.jsonl", public_records)

    train_ids = [value["question_id"] for value in training]
    validation_ids = [value["question_id"] for value in validation]
    train_groups = {value["question_group"] for value in training}
    validation_groups = {value["question_group"] for value in validation}
    split_manifest = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": release_id,
        "status": "PASS",
        "seed": config.seed,
        "method": "group-aware",
        "split_version": QA_SPLIT_VERSION,
        "train_ids_sha256": _hash_values(train_ids),
        "validation_ids_sha256": _hash_values(validation_ids),
        "train_groups_sha256": _hash_values(train_groups),
        "validation_groups_sha256": _hash_values(validation_groups),
        "counts": {
            "train": len(training),
            "validation": len(validation),
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
        },
        "overlap": {
            "question_ids": len(set(train_ids) & set(validation_ids)),
            "question_groups": len(train_groups & validation_groups),
        },
    }
    write_json(release_root / "split_manifest.json", split_manifest)

    exact_duplicate_groups = [
        sorted(ids) for question, ids in exact_questions.items() if question and len(ids) > 1
    ]
    suspicious_pairs = []
    for group, values in near_groups.items():
        distinct = {question for _question_id, question in values}
        if len(values) > 1 and len(distinct) > 1:
            suspicious_pairs.append(
                {
                    "question_group": group,
                    "question_ids": sorted(value[0] for value in values),
                    "distinct_questions": len(distinct),
                }
            )
    leakage_report = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": release_id,
        "status": "PASS" if public_label_usage == 0 else "FAIL",
        "exact_duplicate_groups": exact_duplicate_groups,
        "near_duplicate_method": "NFC casefold + document/number masking + punctuation collapse",
        "suspicious_pairs": suspicious_pairs,
        "cross_task_hits": 0,
        "public_label_usage": public_label_usage,
        "private_label_usage": 0,
        "thresholds": {
            "exact": 1.0,
            "near_grouping": "deterministic masked-template equality",
        },
    }
    write_json(release_root / "leakage_report.json", leakage_report)

    data_report = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": release_id,
        "status": "PASS"
        if not (empty_questions or empty_answers or public_label_usage)
        else "FAIL",
        "raw_counts": {"train": len(raw_train), "public": len(raw_public)},
        "processed_counts": {
            "train": len(training),
            "validation": len(validation),
            "public": len(public_records),
        },
        "train_validation_record_contract": {
            "required_fields": list(QA_TRAIN_VALIDATION_REQUIRED_FIELDS),
            "records_checked": len(training) + len(validation),
            "records_missing_required_fields": 0,
        },
        "empty_counts": {
            "train_questions": empty_questions,
            "train_answers": empty_answers,
            "public_questions": sum(not value["question_model"] for value in public_records),
        },
        "duplicate_counts": {
            "exact_groups": len(exact_duplicate_groups),
            "near_template_groups": len(suspicious_pairs),
        },
        "normalization": {
            "method": "Unicode NFC, CRLF/CR to LF, outer trim; question whitespace collapsed",
            "question_records_changed": nfc_question_changes,
            "answer_records_changed": nfc_answer_changes,
        },
        "newline_preservation": {
            "raw_newlines": raw_answer_newlines,
            "model_newlines": model_answer_newlines,
            "internal_structure_preserved": True,
        },
        "source_checksums": {
            config.data.train_file: sha256_file(train_source),
            config.data.test_file: sha256_file(public_source),
        },
        "transform_version": QA_TRANSFORM_VERSION,
    }
    write_json(release_root / "data_report.json", data_report)

    public_path = qa_dir / "public.jsonl"
    public_manifest = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": release_id,
        "status": "READY" if public_label_usage == 0 else "FAIL",
        "phase": "public",
        "source_sha256": sha256_file(public_source),
        "records": len(public_records),
        "ids_sha256": _hash_values(value["question_id"] for value in public_records),
        "answer_visibility": "hidden",
        "transform_version": QA_TRANSFORM_VERSION,
        "artifact_sha256": sha256_file(public_path),
    }
    write_json(qa_dir / "public_manifest.json", public_manifest)
    return {
        "training": training,
        "validation": validation,
        "public": public_records,
        "data_report": data_report,
        "leakage_report": leakage_report,
        "split_manifest": split_manifest,
    }


def _document_id(member_name: str, value: dict[str, Any] | None) -> str:
    if value and value.get("id") is not None:
        return str(value["id"])
    match = _CONTEXT_ID_RE.search(member_name)
    return match.group(1) if match else sha256_bytes(member_name.encode("utf-8"))[:16]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _window_spans(
    text: str,
    start: int,
    end: int,
    *,
    max_chars: int,
    overlap_chars: int,
) -> Iterator[tuple[int, int]]:
    start, end = _trim_span(text, start, end)
    cursor = start
    while cursor < end:
        candidate_end = min(end, cursor + max_chars)
        if candidate_end < end:
            floor = cursor + max(1, int(max_chars * 0.6))
            newline = text.rfind("\n", floor, candidate_end)
            space = text.rfind(" ", floor, candidate_end)
            boundary = max(newline, space)
            if boundary > cursor:
                candidate_end = boundary
        span_start, span_end = _trim_span(text, cursor, candidate_end)
        if span_start < span_end:
            yield span_start, span_end
        if candidate_end >= end:
            break
        next_cursor = max(cursor + 1, candidate_end - overlap_chars)
        while next_cursor < end and next_cursor > cursor and text[next_cursor].isspace():
            next_cursor += 1
        cursor = next_cursor


def _last_group(pattern: re.Pattern[str], text: str, position: int) -> str | None:
    result = None
    for match in pattern.finditer(text, 0, position):
        result = _minimal_text(match.group(1), preserve_newlines=False)
    return result


def _surface_document_number(name: str, passage: str) -> str | None:
    match = _DOCUMENT_NUMBER_SURFACE_RE.search(f"{name}\n{passage[:5000]}")
    return match.group(0).strip() if match else None


def _safe_id(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value.casefold().replace("đ", "dd"))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", ascii_value).strip("_")
    return cleaned[:80] or sha256_bytes(value.encode("utf-8"))[:12]


_CHUNK_STRUCTURE_LEVELS = {
    "article": {"article", "article_group", "article_intro"},
    "clause": {"clause", "point"},
    "preamble": {"preamble"},
    "fallback": {"fixed"},
}


def _chunk_structure_bucket(level: str) -> str:
    for bucket, levels in _CHUNK_STRUCTURE_LEVELS.items():
        if level in levels:
            return bucket
    return "other"


def _valid_logical_parent_id(doc_id: str, parent_chunk_id: str) -> bool:
    """Validate the release's logical parent-group ID and document ownership."""
    return bool(parent_chunk_id) and parent_chunk_id.startswith(f"doc_{_safe_id(doc_id)}_")


def _hierarchical_segments(
    passage: str,
    article_start: int,
    article_end: int,
    header_end: int,
) -> list[tuple[str, str | None, str | None, int, int]]:
    clauses = list(_CLAUSE_RE.finditer(passage, header_end, article_end))
    if not clauses:
        return [("article", None, None, article_start, article_end)]
    result: list[tuple[str, str | None, str | None, int, int]] = []
    if article_start < clauses[0].start():
        result.append(("article_intro", None, None, article_start, clauses[0].start()))
    for clause_index, clause_match in enumerate(clauses):
        clause_end = (
            clauses[clause_index + 1].start() if clause_index + 1 < len(clauses) else article_end
        )
        clause = clause_match.group("number").casefold()
        points = list(_POINT_RE.finditer(passage, clause_match.end(), clause_end))
        if not points:
            result.append(("clause", clause, None, clause_match.start(), clause_end))
            continue
        if clause_match.start() < points[0].start():
            result.append(("clause", clause, None, clause_match.start(), points[0].start()))
        for point_index, point_match in enumerate(points):
            point_end = (
                points[point_index + 1].start() if point_index + 1 < len(points) else clause_end
            )
            result.append(
                (
                    "point",
                    clause,
                    point_match.group("number").casefold(),
                    point_match.start(),
                    point_end,
                )
            )
    return result


def _pack_hierarchical_segments(
    segments: list[tuple[str, str | None, str | None, int, int]],
    *,
    max_chars: int,
) -> list[tuple[str, str | None, str | None, int, int]]:
    packed: list[tuple[str, str | None, str | None, int, int]] = []
    current: list[tuple[str, str | None, str | None, int, int]] = []

    def emit() -> None:
        if not current:
            return
        levels = {value[0] for value in current}
        clauses = {value[1] for value in current}
        points = {value[2] for value in current}
        level = next(iter(levels)) if len(current) == 1 else "article_group"
        clause = next(iter(clauses)) if len(clauses) == 1 else None
        point = next(iter(points)) if len(points) == 1 else None
        packed.append((level, clause, point, current[0][3], current[-1][4]))
        current.clear()

    for segment in segments:
        segment_length = segment[4] - segment[3]
        if segment_length > max_chars:
            emit()
            packed.append(segment)
            continue
        combined_length = segment[4] - current[0][3] if current else segment_length
        if current and combined_length > max_chars:
            emit()
        current.append(segment)
    emit()
    return packed


def _iter_canonical_chunks(
    *,
    doc_id: str,
    name: str,
    passage: str,
    source_sha256: str,
    quality_flags: list[str],
    max_chars: int,
    overlap_chars: int,
) -> Iterator[dict[str, Any]]:
    metadata = extract_document_metadata(name, passage)
    doc_type = metadata.get("document_type")
    doc_number_canonical = canonical_document_number(metadata.get("document_number"))
    doc_number_surface = _surface_document_number(name, passage)
    articles = list(_ARTICLE_RE.finditer(passage))
    safe_doc = _safe_id(doc_id)

    def emit_segment(
        *,
        parent_id: str,
        parent_start: int,
        parent_end: int,
        heading: str,
        chapter: str | None,
        article: str | None,
        level: str,
        clause: str | None,
        point: str | None,
        segment_start: int,
        segment_end: int,
        base_id: str,
        inherited_flags: list[str],
    ) -> Iterator[dict[str, Any]]:
        for window_index, (source_start, source_end) in enumerate(
            _window_spans(
                passage,
                segment_start,
                segment_end,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        ):
            raw_text = passage[source_start:source_end]
            canonical = normalize_text(raw_text)
            context_start = max(parent_start, (source_start + source_end) // 2 - 2500)
            context_end = min(parent_end, context_start + 5000)
            context_start = max(parent_start, context_end - 5000)
            parent_text = normalize_text(passage[context_start:context_end])
            flags = list(inherited_flags)
            if len(canonical) < 60:
                flags.append("short_chunk")
            hierarchy = " > ".join(value for value in (name, chapter, heading) if value)
            retrieval_text = f"{hierarchy}\n{canonical}".strip()
            contained_clauses = sorted(
                {match.group("number").casefold() for match in _CLAUSE_RE.finditer(raw_text)}
                | ({clause} if clause else set())
            )
            contained_points = sorted(
                {match.group("number").casefold() for match in _POINT_RE.finditer(raw_text)}
                | ({point} if point else set())
            )
            yield {
                "chunk_id": f"{base_id}_w{window_index}",
                "doc_id": doc_id,
                "source_task": TASK_ID,
                "parent_chunk_id": parent_id,
                "doc_type": doc_type,
                "doc_number_surface": doc_number_surface,
                "doc_number_canonical": doc_number_canonical,
                "chapter": chapter,
                "article": article,
                "clause": clause,
                "point": point,
                "contained_clauses": contained_clauses,
                "contained_points": contained_points,
                "level": level,
                "raw_text": raw_text,
                "canonical_text": canonical,
                "retrieval_text": retrieval_text,
                "parent_text": parent_text,
                "source_sha256": source_sha256,
                "source_start": source_start,
                "source_end": source_end,
                "parser_version": PARSER_VERSION,
                "transform_version": CORPUS_TRANSFORM_VERSION,
                "indexable": True,
                "quality_flags": sorted(set(flags)),
            }

    if not articles:
        parent_id = f"doc_{safe_doc}_document"
        yield from emit_segment(
            parent_id=parent_id,
            parent_start=0,
            parent_end=len(passage),
            heading=name,
            chapter=None,
            article=None,
            level="fixed",
            clause=None,
            point=None,
            segment_start=0,
            segment_end=len(passage),
            base_id=f"{parent_id}_fixed",
            inherited_flags=[*quality_flags, "unparsed_fallback"],
        )
        return

    if articles[0].start() > 0 and passage[: articles[0].start()].strip():
        parent_id = f"doc_{safe_doc}_preamble"
        yield from emit_segment(
            parent_id=parent_id,
            parent_start=0,
            parent_end=articles[0].start(),
            heading=name,
            chapter=None,
            article=None,
            level="preamble",
            clause=None,
            point=None,
            segment_start=0,
            segment_end=articles[0].start(),
            base_id=parent_id,
            inherited_flags=quality_flags,
        )

    occurrences: Counter[str] = Counter()
    for article_index, article_match in enumerate(articles):
        article_end = (
            articles[article_index + 1].start()
            if article_index + 1 < len(articles)
            else len(passage)
        )
        article = article_match.group("number").casefold()
        occurrence = occurrences[article]
        occurrences[article] += 1
        heading = _minimal_text(article_match.group(0), preserve_newlines=False)
        chapter = _last_group(_CHAPTER_RE, passage, article_match.start())
        parent_id = (
            f"doc_{safe_doc}_article_{_safe_id(article)}_{occurrence}_{article_match.start()}"
        )
        segments = _pack_hierarchical_segments(
            _hierarchical_segments(
                passage,
                article_match.start(),
                article_end,
                article_match.end(),
            ),
            max_chars=max_chars,
        )
        for segment_index, (level, clause, point, segment_start, segment_end) in enumerate(
            segments
        ):
            suffix = (
                f"{level}_{_safe_id(clause or 'none')}_{_safe_id(point or 'none')}_{segment_index}"
            )
            yield from emit_segment(
                parent_id=parent_id,
                parent_start=article_match.start(),
                parent_end=article_end,
                heading=heading,
                chapter=chapter,
                article=article,
                level=level,
                clause=clause,
                point=point,
                segment_start=segment_start,
                segment_end=segment_end,
                base_id=f"{parent_id}_{suffix}",
                inherited_flags=quality_flags,
            )


def _passage_fingerprint(passage: str) -> str:
    value = _minimal_text(passage, preserve_newlines=True)
    return sha256_bytes(value.encode("utf-8"))


def build_corpus_release(
    config: PipelineConfig,
    release_root: Path,
    *,
    release_id: str,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    corpus_dir = release_root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    source_contexts = data_path(config, config.data.contexts_path)
    member_count = context_source_count(source_contexts)
    inventory: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    fingerprint_groups: dict[str, list[str]] = defaultdict(list)
    invalid_member_names: list[str] = []

    for index, (member_name, payload) in enumerate(iter_context_payloads(source_contexts), start=1):
        json_valid = True
        value: dict[str, Any] | None
        try:
            decoded = json.loads(payload.decode("utf-8-sig"))
            value = decoded if isinstance(decoded, dict) else None
            json_valid = value is not None
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
            json_valid = False
        doc_id = _document_id(member_name, value)
        relative_path = PurePosixPath(member_name).name
        source_sha = sha256_bytes(payload)
        inventory_record = {
            "doc_id": doc_id,
            "relative_path": relative_path,
            "source_task": TASK_ID,
            "sha256": source_sha,
            "bytes": len(payload),
            "json_valid": json_valid,
        }
        inventory.append(inventory_record)
        passage = str((value or {}).get("passage") or "")
        fingerprint = _passage_fingerprint(passage) if passage.strip() else ""
        if fingerprint:
            fingerprint_groups[fingerprint].append(doc_id)
        summaries[doc_id] = {
            "member_name": member_name,
            "source_sha256": source_sha,
            "json_valid": json_valid,
            "passage_fingerprint": fingerprint,
        }
        if not json_valid:
            invalid_member_names.append(member_name)
        if index % 500 == 0 or index == member_count:
            progress(f"Inventoried selected-contexts {index}/{member_count}")

    write_jsonl(corpus_dir / "raw_inventory.jsonl", inventory)
    duplicate_groups = [sorted(ids) for ids in fingerprint_groups.values() if len(ids) > 1]
    alias_to_primary: dict[str, str] = {}
    for ids in duplicate_groups:
        primary = ids[0]
        for alias in ids[1:]:
            alias_to_primary[alias] = primary

    document_path = corpus_dir / "documents.jsonl"
    chunk_path = corpus_dir / "chunks.jsonl"
    quarantine_path = corpus_dir / "quarantine.jsonl"
    document_count = 0
    indexable_documents = 0
    chunk_count = 0
    quarantine_count = 0
    empty_passages = 0
    missing_names = 0
    very_long = 0
    extreme = 0
    paywall_quarantined = 0
    docs_with_articles = 0
    total_chars = 0
    orphan_chars = 0
    unparsed_chars = 0
    chunk_levels: Counter[str] = Counter()
    chunk_structure_counts: Counter[str] = Counter()
    chunk_ids: set[str] = set()
    parent_doc_ids: dict[str, str] = {}
    orphan_parent_ids: set[str] = set()
    orphan_parent_references = 0
    empty_retrieval_text = 0
    source_offset_round_trip_failures = 0
    retrieval_fingerprints: set[str] = set()
    duplicate_retrieval_fingerprints: set[str] = set()
    duplicate_searchable_chunks = 0
    indexable_chunks = 0
    document_status_counts: Counter[str] = Counter()
    quarantine_reason_counts: Counter[str] = Counter()
    article_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    clause_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    point_index: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)

    with (
        document_path.open("w", encoding="utf-8", newline="\n") as document_handle,
        chunk_path.open("w", encoding="utf-8", newline="\n") as chunk_handle,
        quarantine_path.open("w", encoding="utf-8", newline="\n") as quarantine_handle,
    ):
        for index, (member_name, payload) in enumerate(
            iter_context_payloads(source_contexts), start=1
        ):
            source_sha = sha256_bytes(payload)
            try:
                decoded = json.loads(payload.decode("utf-8-sig"))
                value = decoded if isinstance(decoded, dict) else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = None
            doc_id = _document_id(member_name, value)
            passage = str((value or {}).get("passage") or "")
            name_raw = str((value or {}).get("name") or "")
            link_raw = str((value or {}).get("link") or "")
            name_derived = _minimal_text(
                name_raw, preserve_newlines=False
            ) or recover_name_from_link(link_raw, doc_id)
            name_source = "name_raw" if name_raw.strip() else "url_slug"
            quality_flags = []
            if not name_raw.strip():
                missing_names += 1
                quality_flags.append("missing_name_recovered")
            if len(passage) > config.audit.very_long_chars:
                very_long += 1
                quality_flags.append("very_long")
            if len(passage) > config.audit.extreme_chars:
                extreme += 1
                quality_flags.append("extreme_length")

            if value is None:
                status = "QUARANTINED"
                indexable = False
                quality_flags.append("invalid_json")
                reason = "invalid_json"
            elif not passage.strip():
                status = "QUARANTINED"
                indexable = False
                quality_flags.append("empty_passage")
                reason = "empty_passage"
                empty_passages += 1
            elif PAYWALL_RE.search(passage):
                status = "QUARANTINED"
                indexable = False
                quality_flags.append("paywall_repetition_requires_review")
                reason = "paywall_repetition_requires_review"
                paywall_quarantined += 1
            elif doc_id in alias_to_primary:
                status = "ALIAS"
                indexable = False
                quality_flags.append("exact_duplicate_alias")
                reason = "exact_duplicate_alias"
            else:
                status = "INCLUDED"
                indexable = True
                reason = ""
            document_record = {
                "doc_id": doc_id,
                "source_task": TASK_ID,
                "name_raw": name_raw,
                "name_derived": name_derived,
                "name_source": name_source,
                "link_raw": link_raw,
                "passage_raw": passage,
                "source_sha256": source_sha,
                "status": status,
                "indexable": indexable,
                "quality_flags": sorted(set(quality_flags)),
                "transform_version": CORPUS_TRANSFORM_VERSION,
            }
            if doc_id in alias_to_primary:
                document_record["alias_of"] = alias_to_primary[doc_id]
            document_handle.write(
                json.dumps(document_record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            document_count += 1
            document_status_counts[status] += 1
            if not indexable:
                quarantine_record = {
                    "doc_id": doc_id,
                    "reason": reason,
                    "source_sha256": source_sha,
                    "included_in_index": False,
                    "review_status": "PENDING",
                }
                if doc_id in alias_to_primary:
                    quarantine_record["alias_of"] = alias_to_primary[doc_id]
                quarantine_handle.write(
                    json.dumps(quarantine_record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                quarantine_count += 1
                quarantine_reason_counts[reason] += 1
            if indexable:
                indexable_documents += 1
                total_chars += len(passage)
                articles = list(_ARTICLE_RE.finditer(passage))
                if articles:
                    docs_with_articles += 1
                    orphan_chars += articles[0].start()
                else:
                    unparsed_chars += len(passage)
                for chunk in _iter_canonical_chunks(
                    doc_id=doc_id,
                    name=name_derived,
                    passage=passage,
                    source_sha256=source_sha,
                    quality_flags=quality_flags,
                    max_chars=config.chunking.max_chars,
                    overlap_chars=config.chunking.overlap_chars,
                ):
                    chunk_handle.write(
                        json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    chunk_count += 1
                    chunk_id = str(chunk["chunk_id"])
                    chunk_ids.add(chunk_id)
                    level = str(chunk["level"])
                    chunk_levels[level] += 1
                    chunk_structure_counts[_chunk_structure_bucket(level)] += 1
                    indexable_chunks += int(bool(chunk["indexable"]))
                    retrieval_text = str(chunk["retrieval_text"])
                    empty_retrieval_text += int(not retrieval_text.strip())
                    retrieval_fingerprint = sha256_bytes(retrieval_text.encode("utf-8"))
                    if retrieval_fingerprint in retrieval_fingerprints:
                        duplicate_searchable_chunks += 1
                        duplicate_retrieval_fingerprints.add(retrieval_fingerprint)
                    else:
                        retrieval_fingerprints.add(retrieval_fingerprint)
                    parent_chunk_id = str(chunk["parent_chunk_id"])
                    previous_parent_doc = parent_doc_ids.setdefault(parent_chunk_id, doc_id)
                    if previous_parent_doc != doc_id or not _valid_logical_parent_id(
                        doc_id, parent_chunk_id
                    ):
                        orphan_parent_ids.add(parent_chunk_id)
                        orphan_parent_references += 1
                    start = int(chunk["source_start"])
                    end = int(chunk["source_end"])
                    source_offset_round_trip_failures += int(
                        passage[start:end] != chunk["raw_text"]
                    )
                    number = canonical_document_number(chunk.get("doc_number_canonical"))
                    article = str(chunk.get("article") or "").casefold()
                    clause = str(chunk.get("clause") or "").casefold()
                    point = str(chunk.get("point") or "").casefold()
                    if number and article:
                        article_index[(number, article)].append(chunk["chunk_id"])
                        for contained_clause in chunk.get("contained_clauses") or (
                            [clause] if clause else []
                        ):
                            clause_index[(number, article, contained_clause)].append(
                                chunk["chunk_id"]
                            )
                        if clause and point:
                            point_index[(number, article, clause, point)].append(chunk["chunk_id"])
            if index % 100 == 0 or index == member_count:
                progress(
                    f"Built canonical corpus {index}/{member_count} docs, {chunk_count} chunks"
                )

    inventory_digest = hashlib.sha256()
    for record in sorted(inventory, key=lambda value: value["relative_path"]):
        inventory_digest.update(_stable_json_bytes(record))
        inventory_digest.update(b"\n")
    corpus_file_sizes = {
        path.name: path.stat().st_size
        for path in (
            corpus_dir / "raw_inventory.jsonl",
            document_path,
            chunk_path,
            quarantine_path,
        )
    }
    corpus_integrity_ok = all(
        (
            document_count == len(inventory),
            quarantine_count == document_count - indexable_documents,
            chunk_count == indexable_chunks == len(chunk_ids),
            not orphan_parent_ids,
            orphan_parent_references == 0,
            empty_retrieval_text == 0,
            source_offset_round_trip_failures == 0,
        )
    )
    corpus_report = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": release_id,
        "status": "PASS" if corpus_integrity_ok else "FAIL",
        "documents_total": document_count,
        "documents_indexable": indexable_documents,
        "documents_non_indexable": document_count - indexable_documents,
        "non_indexable_by_reason": dict(sorted(quarantine_reason_counts.items())),
        "document_status_counts": dict(sorted(document_status_counts.items())),
        "empty_passages": empty_passages,
        "missing_names": missing_names,
        "length_outliers": {"very_long": very_long, "extreme": extreme},
        "exact_duplicate_groups": duplicate_groups,
        "near_duplicate_groups": [],
        "docs_with_articles": docs_with_articles,
        "orphan_text_ratio": orphan_chars / max(1, total_chars),
        "unparsed_ratio": unparsed_chars / max(1, total_chars),
        "parse_failures": len(invalid_member_names),
        "paywall_quarantined": paywall_quarantined,
        "chunk_count": chunk_count,
        "chunks_total": chunk_count,
        "chunks_indexable": indexable_chunks,
        "chunk_structure_counts": dict(sorted(chunk_structure_counts.items())),
        "chunk_structure_basis": {
            key: sorted(value) for key, value in _CHUNK_STRUCTURE_LEVELS.items()
        },
        "chunk_levels": dict(sorted(chunk_levels.items())),
        "chunk_level_counts": dict(sorted(chunk_levels.items())),
        "unique_chunk_id": len(chunk_ids),
        "duplicate_chunk_id": chunk_count - len(chunk_ids),
        "unique_parent_chunk_id": len(parent_doc_ids),
        "orphan_parent_chunk_ids": len(orphan_parent_ids),
        "orphan_parent_chunk_id_references": orphan_parent_references,
        "orphan_parent_count": len(orphan_parent_ids),
        "parent_chunk_id_basis": (
            "Logical parent-group IDs embedded in child chunks; an orphan is blank, "
            "owned by multiple documents, or does not carry its child document prefix."
        ),
        "empty_retrieval_text": empty_retrieval_text,
        "source_offset_round_trip_failures": source_offset_round_trip_failures,
        "source_span_failures": source_offset_round_trip_failures,
        "duplicate_searchable_chunks": duplicate_searchable_chunks,
        "duplicate_searchable_chunk_groups": len(duplicate_retrieval_fingerprints),
        "duplicate_searchable_basis": (
            "Exact UTF-8 equality of retrieval_text; count is duplicates after "
            "the first occurrence."
        ),
        "quarantine_count": quarantine_count,
        "quarantine_reason_counts": dict(sorted(quarantine_reason_counts.items())),
        "quarantine_coverage_basis": {
            "numerator_field": "quarantine_count",
            "numerator": quarantine_count,
            "denominator_field": "documents_non_indexable",
            "denominator": document_count - indexable_documents,
        },
        "file_sizes_bytes": corpus_file_sizes,
        "artifact_sizes_bytes": corpus_file_sizes,
        "chunk_token_stats": {
            "status": "NOT_RUN",
            "reason": "Tokenizer control is owned by the training preflight.",
        },
    }
    write_json(corpus_dir / "corpus_report.json", corpus_report)
    corpus_artifacts = {}
    for name in (
        "raw_inventory.jsonl",
        "documents.jsonl",
        "chunks.jsonl",
        "quarantine.jsonl",
        "corpus_report.json",
    ):
        path = corpus_dir / name
        corpus_artifacts[name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            **(
                {"records": sum(1 for _ in path.open("r", encoding="utf-8"))}
                if path.suffix == ".jsonl"
                else {}
            ),
        }
    corpus_manifest = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": release_id,
        "status": "READY" if corpus_report["status"] == "PASS" else "FAIL",
        "raw_inventory_sha256": inventory_digest.hexdigest(),
        "document_count": document_count,
        "chunk_count": chunk_count,
        "source_checksums": {config.data.contexts_path: context_source_sha256(source_contexts)},
        "parser_version": PARSER_VERSION,
        "transform_version": CORPUS_TRANSFORM_VERSION,
        "artifacts": corpus_artifacts,
    }
    write_json(corpus_dir / "corpus_manifest.json", corpus_manifest)
    return {
        "report": corpus_report,
        "manifest": corpus_manifest,
        "inventory": inventory,
        "article_index": article_index,
        "clause_index": clause_index,
        "point_index": point_index,
    }


def _build_citation_qrels(
    validation: list[dict[str, Any]],
    corpus: dict[str, Any],
    release_root: Path,
    *,
    release_id: str,
    seed: int,
    manual_sample_size: int,
) -> dict[str, Any]:
    diagnostic_dir = release_root / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    article_index = corpus["article_index"]
    clause_index = corpus["clause_index"]
    point_index = corpus["point_index"]
    qrels: list[dict[str, Any]] = []
    answers_with_citations = 0
    matched_high = 0
    matched_medium = 0
    matched_low = 0
    unmatched = 0
    unmatched_citations = 0
    ambiguous_citations = 0
    ambiguous_question_ids: set[str] = set()
    matched_answers = 0
    qrel_confidence_counts: Counter[str] = Counter()
    answer_matches: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in validation:
        question_id = record["question_id"]
        citations = extract_citations(record["answer_model"])
        answers_with_citations += int(bool(citations))
        answer_confidences: set[str] = set()
        answer_candidate_ids: set[str] = set()
        for citation in citations:
            number = canonical_document_number(citation.document_number)
            article = citation.article.casefold()
            clause = (citation.clause or "").casefold()
            point = (citation.point or "").casefold()
            candidates: list[str] = []
            confidence = "LOW"
            reason = "unmatched"
            if number and article and clause and point:
                candidates = point_index.get((number, article, clause, point), [])
                if candidates:
                    confidence = "HIGH"
                    reason = "exact_doc_article_clause_point"
            if not candidates and number and article and clause:
                candidates = clause_index.get((number, article, clause), [])
                if candidates:
                    confidence = "HIGH"
                    reason = "exact_doc_article_clause"
            if not candidates and number and article:
                candidates = article_index.get((number, article), [])
                if candidates:
                    confidence = "MEDIUM"
                    reason = "exact_doc_article"
            for chunk_id in candidates:
                key = f"{question_id}\0{chunk_id}"
                if key in answer_candidate_ids:
                    continue
                answer_candidate_ids.add(key)
                qrel = {
                    "question_id": question_id,
                    "chunk_id": chunk_id,
                    "relevance": 1,
                    "confidence": confidence,
                    "match_reason": reason,
                    "qrels_status": "diagnostic_proxy",
                    "citation_parser_version": CITATION_PARSER_VERSION,
                    "matcher_version": CITATION_MATCHER_VERSION,
                }
                qrels.append(qrel)
                qrel_confidence_counts[confidence] += 1
                answer_matches[question_id].append(qrel)
            if candidates:
                answer_confidences.add(confidence)
                if len(candidates) > 20:
                    ambiguous_citations += 1
                    ambiguous_question_ids.add(question_id)
            else:
                unmatched_citations += 1
        if answer_matches[question_id]:
            matched_answers += 1
            if "HIGH" in answer_confidences:
                matched_high += 1
            elif "MEDIUM" in answer_confidences:
                matched_medium += 1
            else:
                matched_low += 1
        elif citations:
            unmatched += 1

    write_jsonl(diagnostic_dir / "citation_qrels.jsonl", qrels)
    report = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": release_id,
        "status": "PASS",
        "qrels_status": "diagnostic_proxy",
        "parser_version": CITATION_PARSER_VERSION,
        "matcher_version": CITATION_MATCHER_VERSION,
        "evaluation_split": "validation",
        "count_unit": "answers",
        "answers_scanned": len(validation),
        "answers_with_citations": answers_with_citations,
        "questions_with_at_least_one_match": matched_answers,
        "questions_with_matches": matched_answers,
        "unique_question_chunk_pairs": len(qrels),
        "qrel_pairs_unique": len(qrels),
        "qrel_pair_counts_by_confidence": {
            confidence: qrel_confidence_counts[confidence]
            for confidence in ("HIGH", "MEDIUM", "LOW")
        },
        "confidence_counts": {
            confidence: qrel_confidence_counts[confidence]
            for confidence in ("HIGH", "MEDIUM", "LOW")
        },
        "question_counts_by_best_confidence": {
            "HIGH": matched_high,
            "MEDIUM": matched_medium,
            "LOW": matched_low,
        },
        "matched_high": matched_high,
        "matched_medium": matched_medium,
        "matched_low": matched_low,
        "unmatched": unmatched,
        "unmatched_questions_with_citations": unmatched,
        "unmatched_citation_mentions": unmatched_citations,
        "ambiguous": ambiguous_citations,
        "ambiguous_questions": len(ambiguous_question_ids),
        "ambiguous_citation_mentions": ambiguous_citations,
        "ambiguous_candidate_threshold": 20,
        "qrel_records": len(qrels),
        "judged_coverage": matched_answers / max(1, len(validation)),
        "judged_coverage_basis": {
            "numerator_field": "questions_with_at_least_one_match",
            "numerator": matched_answers,
            "denominator_field": "answers_scanned",
            "denominator": len(validation),
            "formula": f"{matched_answers}/{len(validation)}",
        },
        "citation_conditioned_coverage": matched_answers / max(1, answers_with_citations),
        "citation_conditioned_coverage_basis": {
            "numerator_field": "questions_with_at_least_one_match",
            "numerator": matched_answers,
            "denominator_field": "answers_with_citations",
            "denominator": answers_with_citations,
            "formula": f"{matched_answers}/{answers_with_citations}",
        },
        "unjudged_policy": "Unmatched questions remain unjudged; absence is not a negative label.",
        "allowed_uses": ["retrieval validation", "ablation", "error analysis"],
        "prohibited_uses": ["training labels", "augmentation", "public/private labels"],
        "manual_audit_target": manual_sample_size,
        "manual_audit_completed": 0,
        "manual_audit_status": "PENDING" if manual_sample_size else "NOT_REQUIRED",
    }
    write_json(diagnostic_dir / "citation_match_report.json", report)

    rng = random.Random(seed)
    eligible_ids = sorted(answer_matches)
    sampled_ids = rng.sample(eligible_ids, min(manual_sample_size, len(eligible_ids)))
    validation_by_id = {value["question_id"]: value for value in validation}
    audit_records = []
    for question_id in sampled_ids:
        audit_records.append(
            {
                "question_id": question_id,
                "question": validation_by_id[question_id]["question_model"],
                "answer": validation_by_id[question_id]["answer_model"],
                "qrels": answer_matches[question_id],
                "sample_seed": seed,
                "reviewer_decision": None,
                "reviewer_note": None,
                "review_status": "PENDING",
            }
        )
    write_jsonl(diagnostic_dir / "citation_manual_audit.jsonl", audit_records)
    return {"qrels": qrels, "report": report, "manual_audit": audit_records}


def _artifact_manifest(release_root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in release_root.rglob("*") if value.is_file()):
        relative = path.relative_to(release_root).as_posix()
        if relative == "manifest.json" or relative.startswith("preflight_"):
            continue
        record: dict[str, Any] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                record["records"] = sum(1 for line in handle if line.strip())
        artifacts[relative] = record
    return artifacts


def refresh_data_release_reports(
    release_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Re-audit an existing release without rebuilding its multi-GB corpus files."""
    root = Path(release_root)
    notify = progress or (lambda _message: None)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    release_id = _validated_release_id(root, str(manifest.get("release_id", "")))
    corpus_dir = root / "corpus"
    report_path = corpus_dir / "corpus_report.json"
    corpus_report = json.loads(report_path.read_text(encoding="utf-8-sig"))

    validation_records = list(read_jsonl(root / "qa" / "validation.jsonl"))
    qa_records_checked = 0
    qa_records_missing_fields = 0
    required_qa_fields = set(QA_TRAIN_VALIDATION_REQUIRED_FIELDS)
    for split in ("train", "validation"):
        records = (
            validation_records if split == "validation" else read_jsonl(root / "qa" / "train.jsonl")
        )
        for record in records:
            qa_records_checked += 1
            qa_records_missing_fields += int(bool(required_qa_fields - set(record)))
    data_report_path = root / "data_report.json"
    data_report = json.loads(data_report_path.read_text(encoding="utf-8-sig"))
    data_report["train_validation_record_contract"] = {
        "required_fields": list(QA_TRAIN_VALIDATION_REQUIRED_FIELDS),
        "records_checked": qa_records_checked,
        "records_missing_required_fields": qa_records_missing_fields,
    }
    if qa_records_missing_fields:
        data_report["status"] = "FAIL"
    write_json(data_report_path, data_report)

    passages: dict[str, str] = {}
    document_status_counts: Counter[str] = Counter()
    non_indexable_documents: list[dict[str, Any]] = []
    indexable_documents = 0
    for document in read_jsonl(corpus_dir / "documents.jsonl"):
        doc_id = str(document["doc_id"])
        passages[doc_id] = str(document["passage_raw"])
        status = str(document["status"])
        document_status_counts[status] += 1
        if document["indexable"]:
            indexable_documents += 1
        else:
            reason = (
                "exact_duplicate_alias"
                if status == "ALIAS"
                else next(
                    (
                        flag
                        for flag in document.get("quality_flags", [])
                        if flag
                        in {
                            "invalid_json",
                            "empty_passage",
                            "paywall_repetition_requires_review",
                        }
                    ),
                    "non_indexable",
                )
            )
            record = {
                "doc_id": doc_id,
                "reason": reason,
                "source_sha256": document["source_sha256"],
                "included_in_index": False,
                "review_status": "PENDING",
            }
            if document.get("alias_of"):
                record["alias_of"] = document["alias_of"]
            non_indexable_documents.append(record)

    previous_quarantine = {
        str(value["doc_id"]): value for value in read_jsonl(corpus_dir / "quarantine.jsonl")
    }
    for record in non_indexable_documents:
        previous = previous_quarantine.get(record["doc_id"], {})
        record["review_status"] = previous.get("review_status", "PENDING")
    write_jsonl(corpus_dir / "quarantine.jsonl", non_indexable_documents)
    quarantine_reason_counts = Counter(str(value["reason"]) for value in non_indexable_documents)

    chunk_ids: set[str] = set()
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
    indexable_chunks = 0
    chunk_count = 0
    article_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    clause_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    point_index: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for chunk in read_jsonl(corpus_dir / "chunks.jsonl"):
        chunk_count += 1
        chunk_id = str(chunk["chunk_id"])
        doc_id = str(chunk["doc_id"])
        chunk_ids.add(chunk_id)
        level = str(chunk["level"])
        chunk_levels[level] += 1
        chunk_structure_counts[_chunk_structure_bucket(level)] += 1
        indexable_chunks += int(bool(chunk["indexable"]))
        retrieval_text = str(chunk["retrieval_text"])
        empty_retrieval_text += int(not retrieval_text.strip())
        fingerprint = sha256_bytes(retrieval_text.encode("utf-8"))
        if fingerprint in retrieval_fingerprints:
            duplicate_searchable_chunks += 1
            duplicate_retrieval_fingerprints.add(fingerprint)
        else:
            retrieval_fingerprints.add(fingerprint)
        parent_chunk_id = str(chunk["parent_chunk_id"])
        previous_parent_doc = parent_doc_ids.setdefault(parent_chunk_id, doc_id)
        if previous_parent_doc != doc_id or not _valid_logical_parent_id(doc_id, parent_chunk_id):
            orphan_parent_ids.add(parent_chunk_id)
            orphan_parent_references += 1
        passage = passages.get(doc_id, "")
        start = chunk["source_start"]
        end = chunk["source_end"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not (0 <= start < end <= len(passage))
            or passage[start:end] != chunk["raw_text"]
        ):
            source_offset_round_trip_failures += 1

        number = canonical_document_number(chunk.get("doc_number_canonical"))
        article = str(chunk.get("article") or "").casefold()
        clause = str(chunk.get("clause") or "").casefold()
        point = str(chunk.get("point") or "").casefold()
        if number and article:
            article_index[(number, article)].append(chunk_id)
            for contained_clause in chunk.get("contained_clauses") or ([clause] if clause else []):
                clause_index[(number, article, str(contained_clause))].append(chunk_id)
            if clause and point:
                point_index[(number, article, clause, point)].append(chunk_id)
        if chunk_count % 50_000 == 0:
            notify(f"Audited release chunks {chunk_count}")
    notify(f"Audited release chunks {chunk_count}")

    document_count = len(passages)
    quarantine_count = len(non_indexable_documents)
    corpus_integrity_ok = all(
        (
            quarantine_count == document_count - indexable_documents,
            chunk_count == indexable_chunks == len(chunk_ids),
            not orphan_parent_ids,
            orphan_parent_references == 0,
            empty_retrieval_text == 0,
            source_offset_round_trip_failures == 0,
        )
    )
    corpus_report.update(
        {
            "status": "PASS" if corpus_integrity_ok else "FAIL",
            "documents_total": document_count,
            "documents_indexable": indexable_documents,
            "documents_non_indexable": document_count - indexable_documents,
            "non_indexable_by_reason": dict(sorted(quarantine_reason_counts.items())),
            "document_status_counts": dict(sorted(document_status_counts.items())),
            "chunk_count": chunk_count,
            "chunks_total": chunk_count,
            "chunks_indexable": indexable_chunks,
            "chunk_structure_counts": dict(sorted(chunk_structure_counts.items())),
            "chunk_structure_basis": {
                key: sorted(value) for key, value in _CHUNK_STRUCTURE_LEVELS.items()
            },
            "chunk_levels": dict(sorted(chunk_levels.items())),
            "chunk_level_counts": dict(sorted(chunk_levels.items())),
            "unique_chunk_id": len(chunk_ids),
            "duplicate_chunk_id": chunk_count - len(chunk_ids),
            "unique_parent_chunk_id": len(parent_doc_ids),
            "orphan_parent_chunk_ids": len(orphan_parent_ids),
            "orphan_parent_chunk_id_references": orphan_parent_references,
            "orphan_parent_count": len(orphan_parent_ids),
            "parent_chunk_id_basis": (
                "Logical parent-group IDs embedded in child chunks; an orphan is blank, "
                "owned by multiple documents, or does not carry its child document prefix."
            ),
            "empty_retrieval_text": empty_retrieval_text,
            "source_offset_round_trip_failures": source_offset_round_trip_failures,
            "source_span_failures": source_offset_round_trip_failures,
            "duplicate_searchable_chunks": duplicate_searchable_chunks,
            "duplicate_searchable_chunk_groups": len(duplicate_retrieval_fingerprints),
            "duplicate_searchable_basis": (
                "Exact UTF-8 equality of retrieval_text; count is duplicates after "
                "the first occurrence."
            ),
            "quarantine_count": quarantine_count,
            "quarantine_reason_counts": dict(sorted(quarantine_reason_counts.items())),
            "quarantine_coverage_basis": {
                "numerator_field": "quarantine_count",
                "numerator": quarantine_count,
                "denominator_field": "documents_non_indexable",
                "denominator": document_count - indexable_documents,
            },
        }
    )

    previous_audit = {
        str(value["question_id"]): value
        for value in read_jsonl(root / "diagnostics" / "citation_manual_audit.jsonl")
    }
    sample_size = len(previous_audit)
    sample_seed = next(
        (int(value.get("sample_seed", 2026)) for value in previous_audit.values()),
        2026,
    )
    diagnostic = _build_citation_qrels(
        validation_records,
        {
            "article_index": article_index,
            "clause_index": clause_index,
            "point_index": point_index,
        },
        root,
        release_id=release_id,
        seed=sample_seed,
        manual_sample_size=sample_size,
    )
    for record in diagnostic["manual_audit"]:
        previous_audit_record = previous_audit.get(str(record["question_id"]))
        if previous_audit_record:
            for field in ("reviewer_decision", "reviewer_note", "review_status"):
                record[field] = previous_audit_record.get(field)
    write_jsonl(
        root / "diagnostics" / "citation_manual_audit.jsonl",
        diagnostic["manual_audit"],
    )

    completed_audits = sum(
        str(record.get("review_status", "PENDING")).upper() == "COMPLETED"
        for record in diagnostic["manual_audit"]
    )
    diagnostic["report"].update(
        {
            "manual_audit_target": len(diagnostic["manual_audit"]),
            "manual_audit_completed": completed_audits,
            "manual_audit_status": (
                "COMPLETED" if completed_audits == len(diagnostic["manual_audit"]) else "PENDING"
            ),
        }
    )
    write_json(
        root / "diagnostics" / "citation_match_report.json",
        diagnostic["report"],
    )

    corpus_report["file_sizes_bytes"] = {
        name: (corpus_dir / name).stat().st_size
        for name in (
            "raw_inventory.jsonl",
            "documents.jsonl",
            "chunks.jsonl",
            "quarantine.jsonl",
        )
    }
    corpus_report["artifact_sizes_bytes"] = corpus_report["file_sizes_bytes"]
    corpus_report.setdefault(
        "chunk_token_stats",
        {
            "status": "NOT_RUN",
            "reason": "Tokenizer control is owned by the training preflight.",
        },
    )
    write_json(report_path, corpus_report)

    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8-sig"))
    changed_corpus_files = {"quarantine.jsonl", "corpus_report.json"}
    for name, record in corpus_manifest["artifacts"].items():
        path = corpus_dir / name
        record["bytes"] = path.stat().st_size
        if name in changed_corpus_files:
            record["sha256"] = sha256_file(path)
        if path.suffix == ".jsonl" and name in changed_corpus_files:
            record["records"] = sum(1 for _ in read_jsonl(path))
    write_json(corpus_manifest_path, corpus_manifest)

    changed_artifacts = {
        "corpus/corpus_manifest.json",
        "corpus/corpus_report.json",
        "corpus/quarantine.jsonl",
        "data_report.json",
        "diagnostics/citation_manual_audit.jsonl",
        "diagnostics/citation_match_report.json",
        "diagnostics/citation_qrels.jsonl",
    }
    for relative, record in manifest["artifacts"].items():
        path = root / relative
        record["bytes"] = path.stat().st_size
        if relative in changed_artifacts:
            record["sha256"] = sha256_file(path)
        if path.suffix == ".jsonl" and relative in changed_artifacts:
            record["records"] = sum(1 for _ in read_jsonl(path))
    manifest["status"] = (
        "READY"
        if data_report["status"]
        == corpus_report["status"]
        == diagnostic["report"]["status"]
        == "PASS"
        else "FAIL"
    )
    manifest["reports_refreshed_at"] = datetime.now(UTC).astimezone().isoformat()
    write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "documents_total": document_count,
        "documents_indexable": indexable_documents,
        "quarantine_count": quarantine_count,
        "chunks_total": chunk_count,
        "unique_question_chunk_pairs": diagnostic["report"]["unique_question_chunk_pairs"],
        "questions_with_at_least_one_match": diagnostic["report"][
            "questions_with_at_least_one_match"
        ],
    }


def build_data_release(
    config: PipelineConfig,
    release_root: str | Path,
    *,
    release_id: str | None = None,
    validation_size: int = 700,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build the canonical Task 2 preprocessing handoff release."""

    config.validate()
    notify = progress or (lambda _message: None)
    root = Path(release_root)
    resolved_release_id = _validated_release_id(root, release_id)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Release root must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    qa = build_qa_release(
        config,
        root,
        validation_size=validation_size,
        release_id=resolved_release_id,
    )
    notify(
        f"Built QA split train={len(qa['training'])} validation={len(qa['validation'])} "
        f"public={len(qa['public'])}"
    )
    corpus = build_corpus_release(
        config,
        root,
        release_id=resolved_release_id,
        progress=notify,
    )
    diagnostic = _build_citation_qrels(
        qa["validation"],
        corpus,
        root,
        release_id=resolved_release_id,
        seed=config.diagnostic.seed,
        manual_sample_size=config.diagnostic.manual_sample_size,
    )
    notify(
        f"Built citation qrels unique_pairs={len(diagnostic['qrels'])} "
        f"matched_questions={diagnostic['report']['questions_with_at_least_one_match']}/"
        f"{diagnostic['report']['answers_scanned']} "
        f"coverage={diagnostic['report']['judged_coverage']:.4f}"
    )
    data_dir = Path(config.data.data_dir)
    overview_candidates = sorted(data_dir.glob("*Data_Overview*.docx"))
    overview_path = overview_candidates[0] if overview_candidates else None
    raw_sources: dict[str, Any] = {
        config.data.train_file: {
            "sha256": sha256_file(data_path(config, config.data.train_file)),
            "records": len(qa["training"]) + len(qa["validation"]),
        },
        config.data.test_file: {
            "sha256": sha256_file(data_path(config, config.data.test_file)),
            "records": len(qa["public"]),
        },
        "selected-contexts": {
            "inventory_sha256": corpus["manifest"]["raw_inventory_sha256"],
            "source_sha256": context_source_sha256(data_path(config, config.data.contexts_path)),
            "records": len(corpus["inventory"]),
        },
    }
    if overview_path:
        raw_sources[overview_path.name] = {"sha256": sha256_file(overview_path)}
    artifacts = _artifact_manifest(root)
    statuses = (
        qa["data_report"]["status"],
        qa["leakage_report"]["status"],
        corpus["report"]["status"],
        diagnostic["report"]["status"],
    )
    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "task_id": TASK_ID,
        "release_id": resolved_release_id,
        "status": "READY" if all(value == "PASS" for value in statuses) else "FAIL",
        "created_at": datetime.now(UTC).astimezone().isoformat(),
        "producer": "preprocessing-owner",
        "raw_sources": raw_sources,
        "artifacts": artifacts,
        "transform_version": QA_TRANSFORM_VERSION,
        "split_version": QA_SPLIT_VERSION,
        "parser_version": PARSER_VERSION,
    }
    write_json(root / "manifest.json", manifest)
    return {
        "release_root": str(root),
        "status": manifest["status"],
        "qa_counts": qa["split_manifest"]["counts"],
        "public_records": len(qa["public"]),
        "corpus_documents": corpus["report"]["documents_total"],
        "corpus_chunks": corpus["report"]["chunk_count"],
        "citation_qrels": len(diagnostic["qrels"]),
        "citation_coverage": diagnostic["report"]["judged_coverage"],
        "artifact_count": len(artifacts),
    }
