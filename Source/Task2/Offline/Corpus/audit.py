from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ..core.config import AuditConfig
from ..core.normalize import canonical_text, text_fingerprint
from ..core.schema import LegalDocument
from ..core.stats import percentiles

_STRUCTURE_RE = re.compile(r"(?im)^\s*(?:chương|mục|điều\s+\d+|\d+[.)]\s+)")


@dataclass(slots=True)
class AuditResult:
    documents: list[LegalDocument]
    report: dict[str, object]
    exact_duplicate_groups: dict[str, list[str]]
    near_duplicate_groups: list[list[str]]


@dataclass(slots=True)
class StreamingAuditPlan:
    canonical_ids: set[str]
    source_ids_by_canonical: dict[str, tuple[str, ...]]
    flags_by_id: dict[str, tuple[str, ...]]
    metadata_by_id: dict[str, dict[str, object]]
    report: dict[str, object]
    exact_duplicate_groups: dict[str, list[str]]
    near_duplicate_groups: list[list[str]]


def _structure_density(text: str) -> float:
    if not text:
        return 0.0
    return len(_STRUCTURE_RE.findall(text)) * 10_000.0 / len(text)


def _simhash(text: str, max_blocks: int = 256, block_chars: int = 700) -> int:
    compact = canonical_text(text)
    block_count = max(1, math.ceil(len(compact) / block_chars))
    step = max(1, math.ceil(block_count / max_blocks))
    weights = [0] * 64
    for block_index in range(0, block_count, step):
        block = compact[block_index * block_chars : (block_index + 1) * block_chars]
        if not block:
            continue
        value = int.from_bytes(
            hashlib.blake2b(block.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _near_groups_from_signatures(
    document_ids: list[str], signatures: list[int], threshold: float
) -> list[list[str]]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for index, signature in enumerate(signatures):
        for band in range(4):
            key = (band, (signature >> (band * 16)) & 0xFFFF)
            for other in buckets[key]:
                candidates.add((other, index))
            buckets[key].append(index)

    parent = list(range(len(document_ids)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    max_distance = math.floor((1.0 - threshold) * 64)
    for left, right in candidates:
        if (signatures[left] ^ signatures[right]).bit_count() <= max_distance:
            union(left, right)
    groups: dict[int, list[str]] = defaultdict(list)
    for index, document_id in enumerate(document_ids):
        groups[find(index)].append(document_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def _near_duplicate_groups(documents: list[LegalDocument], threshold: float) -> list[list[str]]:
    return _near_groups_from_signatures(
        [document.id for document in documents],
        [_simhash(document.passage) for document in documents],
        threshold,
    )


def audit_context_stream(
    documents: Iterable[LegalDocument],
    config: AuditConfig,
) -> StreamingAuditPlan:
    """First-pass audit that retains metadata/signatures, never all passages."""

    fingerprints: dict[str, str] = {}
    source_ids: dict[str, list[str]] = {}
    exact_groups: dict[str, list[str]] = {}
    flags_by_id: dict[str, tuple[str, ...]] = {}
    metadata_by_id: dict[str, dict[str, object]] = {}
    ids: list[str] = []
    signatures: list[int] = []
    empty_ids: list[str] = []
    lengths: list[int] = []
    input_count = 0
    flag_counts: Counter[str] = Counter()
    for document in documents:
        input_count += 1
        if not document.passage.strip():
            empty_ids.append(document.id)
            continue
        fingerprint = text_fingerprint(document.passage)
        canonical_id = fingerprints.get(fingerprint)
        if canonical_id is not None:
            source_ids[canonical_id].append(document.id)
            exact_groups.setdefault(canonical_id, [canonical_id]).append(document.id)
            continue
        fingerprints[fingerprint] = document.id
        source_ids[document.id] = [document.id]
        length = len(document.passage)
        density = _structure_density(document.passage)
        flags = list(document.flags)
        if length >= config.very_long_chars:
            flags.append("very_long")
        if length >= config.extreme_chars:
            flags.append("extreme_length")
        if length >= config.very_long_chars and density < config.min_structure_density_per_10k:
            flags.append("suspicious_structure")
        flags_by_id[document.id] = tuple(dict.fromkeys(flags))
        metadata_by_id[document.id] = {
            "char_length": length,
            "structure_density_per_10k": round(density, 4),
        }
        ids.append(document.id)
        signatures.append(_simhash(document.passage) if config.near_duplicate_enabled else 0)
        lengths.append(length)
        flag_counts.update(flags)

    near_groups = (
        _near_groups_from_signatures(ids, signatures, config.near_duplicate_threshold)
        if config.near_duplicate_enabled
        else []
    )
    near_members = {document_id for group in near_groups for document_id in group}
    for document_id in near_members:
        flags_by_id[document_id] = tuple(
            dict.fromkeys((*flags_by_id.get(document_id, ()), "near_duplicate"))
        )
    flag_counts["near_duplicate"] = len(near_members)
    points = (0, 25, 50, 75, 90, 95, 99, 100)
    report: dict[str, object] = {
        "input_documents": input_count,
        "kept_documents": len(ids),
        "empty_removed": len(empty_ids),
        "empty_ids": empty_ids,
        "exact_duplicates_removed": sum(len(group) - 1 for group in exact_groups.values()),
        "exact_duplicate_group_count": len(exact_groups),
        "near_duplicate_group_count": len(near_groups),
        "near_duplicates_removed": 0,
        "flag_counts": dict(sorted(flag_counts.items())),
        "passage_char_percentiles": percentiles(lengths, points),
        "policy": {
            "empty": "removed",
            "exact_duplicate": "collapsed_with_source_ids_preserved",
            "near_duplicate": "flagged_only",
            "very_long": "kept_and_structurally_parsed",
        },
    }
    return StreamingAuditPlan(
        canonical_ids=set(ids),
        source_ids_by_canonical={key: tuple(value) for key, value in source_ids.items()},
        flags_by_id=flags_by_id,
        metadata_by_id=metadata_by_id,
        report=report,
        exact_duplicate_groups=exact_groups,
        near_duplicate_groups=near_groups,
    )


def audit_corpus(documents: list[LegalDocument], config: AuditConfig) -> AuditResult:
    unique: dict[str, LegalDocument] = {}
    duplicate_groups: dict[str, list[str]] = {}
    empty_ids: list[str] = []
    lengths: list[int] = []
    flag_counts: Counter[str] = Counter()

    for document in documents:
        if not document.passage.strip():
            empty_ids.append(document.id)
            continue
        fingerprint = text_fingerprint(document.passage)
        existing = unique.get(fingerprint)
        if existing is not None:
            existing.source_ids = tuple(dict.fromkeys((*existing.source_ids, document.id)))
            duplicate_groups.setdefault(existing.id, [existing.id]).append(document.id)
            continue

        flags = list(document.flags)
        length = len(document.passage)
        density = _structure_density(document.passage)
        if length >= config.very_long_chars:
            flags.append("very_long")
        if length >= config.extreme_chars:
            flags.append("extreme_length")
        if length >= config.very_long_chars and density < config.min_structure_density_per_10k:
            flags.append("suspicious_structure")
        document.flags = tuple(dict.fromkeys(flags))
        document.metadata["char_length"] = length
        document.metadata["structure_density_per_10k"] = round(density, 4)
        unique[fingerprint] = document
        lengths.append(length)
        flag_counts.update(document.flags)

    kept = list(unique.values())
    near_groups: list[list[str]] = []
    if config.near_duplicate_enabled:
        near_groups = _near_duplicate_groups(kept, config.near_duplicate_threshold)
        membership = {document_id for group in near_groups for document_id in group}
        for document in kept:
            if document.id in membership:
                document.flags = tuple(dict.fromkeys((*document.flags, "near_duplicate")))
        flag_counts["near_duplicate"] = len(membership)

    length_percentiles = percentiles(lengths, (0, 25, 50, 75, 90, 95, 99, 100))
    report: dict[str, object] = {
        "input_documents": len(documents),
        "kept_documents": len(kept),
        "empty_removed": len(empty_ids),
        "empty_ids": empty_ids,
        "exact_duplicates_removed": sum(len(ids) - 1 for ids in duplicate_groups.values()),
        "exact_duplicate_group_count": len(duplicate_groups),
        "near_duplicate_group_count": len(near_groups),
        "near_duplicates_removed": 0,
        "flag_counts": dict(sorted(flag_counts.items())),
        "passage_char_percentiles": length_percentiles,
        "policy": {
            "empty": "removed",
            "exact_duplicate": "collapsed_with_source_ids_preserved",
            "near_duplicate": "flagged_only",
            "very_long": "kept_and_structurally_parsed",
        },
    }
    return AuditResult(kept, report, duplicate_groups, near_groups)
