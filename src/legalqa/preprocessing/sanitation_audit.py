from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import yaml

from ..core.config import PipelineConfig, artifact_dir, data_path
from ..core.io import iter_context_zip, write_json, write_jsonl
from ..core.normalize import normalize_text
from .sanitation import (
    PAYWALL_RE,
    CorpusSanitizer,
    _informative_block,
    _normalized_block,
    local_repetition_metrics,
)

ProgressCallback = Callable[[str], None]


def _line_hash(line: str) -> int:
    return int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "big")


def _normalized_lines(text: str, minimum: int, maximum: int) -> list[str]:
    values: list[str] = []
    for raw_line in normalize_text(text).splitlines():
        line = _normalized_block(raw_line)
        if minimum <= len(line) <= maximum:
            values.append(line)
    return values


def _top_repeated_blocks(text: str, minimum: int, limit: int = 5) -> list[dict[str, object]]:
    blocks = [
        _normalized_block(block)
        for block in re.split(r"\n\s*\n+", normalize_text(text))
    ]
    counts = Counter(block for block in blocks if _informative_block(block, minimum))
    repeated = sorted(
        ((block, count) for block, count in counts.items() if count > 1),
        key=lambda item: (-(len(item[0]) * (item[1] - 1)), -item[1], item[0]),
    )[:limit]
    return [
        {
            "frequency": count,
            "block_chars": len(block),
            "duplicate_chars": len(block) * (count - 1),
            "preview": block[:500],
        }
        for block, count in repeated
    ]


def audit_sanitation(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Discover global boilerplate and local crawler corruption without deletion."""

    notify = progress or (lambda message: None)
    source = data_path(config, config.data.contexts_zip)
    output = artifact_dir(config)
    output.mkdir(parents=True, exist_ok=True)
    sanitizer = CorpusSanitizer(config.sanitation)
    document_frequency: Counter[int] = Counter()
    local_rows: list[dict[str, object]] = []
    review_signal_totals: Counter[str] = Counter()
    document_count = 0

    notify("Sanitation audit pass 1/2: global DF hashes and local block TF")
    for document in iter_context_zip(source):
        document_count += 1
        base = normalize_text(document.raw_text or document.passage)
        lines = _normalized_lines(
            base,
            config.sanitation.df_line_min_chars,
            config.sanitation.df_line_max_chars,
        )
        document_frequency.update(set(_line_hash(line) for line in lines))
        repetition = local_repetition_metrics(base, config.sanitation.block_min_chars)
        paywall_count = len(PAYWALL_RE.findall(base))
        for signal in sanitizer.review_candidates:
            count = base.casefold().count(signal)
            if count:
                review_signal_totals[signal] += count
        suspicious = (
            paywall_count > 0
            or (
                repetition["repetition_ratio"] >= config.sanitation.local_repetition_flag_ratio
                and repetition["max_block_frequency"] >= config.sanitation.local_max_tf_flag
            )
        )
        severity = "CRITICAL" if paywall_count or repetition["max_block_frequency"] >= 20 else (
            "REVIEW" if suspicious else "OK"
        )
        local_rows.append(
            {
                "document_id": document.id,
                "document_length": len(base),
                "paywall_count": paywall_count,
                **repetition,
                "top_repeated_blocks": _top_repeated_blocks(
                    base, config.sanitation.block_min_chars
                )
                if suspicious
                else [],
                "flag": severity,
            }
        )
        if document_count % 500 == 0:
            notify(f"Scanned {document_count} documents")

    min_df = max(2, math.ceil(document_count * config.sanitation.df_min_ratio))
    candidate_counts = {
        line_hash: count for line_hash, count in document_frequency.items() if count >= min_df
    }
    if len(candidate_counts) > config.sanitation.df_max_candidates:
        candidate_counts = dict(
            sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))[
                : config.sanitation.df_max_candidates
            ]
        )

    notify(f"Sanitation audit pass 2/2: recovering {len(candidate_counts)} DF candidate lines")
    representatives: dict[int, str] = {}
    total_frequency: Counter[int] = Counter()
    max_document_frequency: Counter[int] = Counter()
    for document in iter_context_zip(source):
        per_document: Counter[int] = Counter()
        for line in _normalized_lines(
            document.raw_text or document.passage,
            config.sanitation.df_line_min_chars,
            config.sanitation.df_line_max_chars,
        ):
            line_hash = _line_hash(line)
            if line_hash not in candidate_counts:
                continue
            representatives.setdefault(line_hash, line)
            per_document[line_hash] += 1
        total_frequency.update(per_document)
        for line_hash, count in per_document.items():
            max_document_frequency[line_hash] = max(max_document_frequency[line_hash], count)

    candidates = []
    for line_hash, count in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0])):
        line = representatives.get(line_hash, "")
        classification = sanitizer.classify_line(line)
        candidates.append(
            {
                "text": line,
                "document_frequency": count,
                "document_frequency_ratio": round(count / max(1, document_count), 6),
                "total_frequency": total_frequency[line_hash],
                "max_frequency_in_one_document": max_document_frequency[line_hash],
                "classification": classification,
                "action": "review_only" if classification == "manual_review" else classification,
            }
        )

    local_rows.sort(
        key=lambda value: (
            0 if value["flag"] == "CRITICAL" else 1 if value["flag"] == "REVIEW" else 2,
            -int(value["paywall_count"]),
            -float(value["duplicate_char_ratio"]),
            -int(value["document_length"]),
        )
    )
    write_json(output / "boilerplate_candidates.json", candidates)
    write_jsonl(output / "local_corruption_report.jsonl", local_rows)
    summary: dict[str, object] = {
        "documents": document_count,
        "line_df_minimum": min_df,
        "df_candidates": len(candidates),
        "candidate_class_counts": dict(Counter(value["classification"] for value in candidates)),
        "critical_documents": sum(value["flag"] == "CRITICAL" for value in local_rows),
        "review_documents": sum(value["flag"] == "REVIEW" for value in local_rows),
        "paywall_documents": sum(int(value["paywall_count"]) > 0 for value in local_rows),
        "paywall_occurrences": sum(int(value["paywall_count"]) for value in local_rows),
        "review_signal_totals": dict(review_signal_totals.most_common()),
        "policy": {
            "document_frequency": "candidate_discovery_only_never_auto_delete",
            "local_block_frequency": "corruption_flag_only",
            "short_blocks_below_chars": config.sanitation.block_min_chars,
        },
        "outputs": {
            "boilerplate_candidates": str(output / "boilerplate_candidates.json"),
            "local_corruption_report": str(output / "local_corruption_report.jsonl"),
        },
    }
    write_json(output / "sanitation_audit_report.json", summary)
    return summary


def review_boilerplate_candidates(config: PipelineConfig) -> dict[str, object]:
    """Apply the manually reviewed v2.1 decision policy without deleting text."""

    output = artifact_dir(config)
    candidates_path = output / "boilerplate_candidates.json"
    candidates = __import__("json").loads(candidates_path.read_text(encoding="utf-8"))
    policy = yaml.safe_load(Path(config.sanitation.review_policy_path).read_text(encoding="utf-8")) or {}
    headers = {str(value).casefold() for value in policy.get("header_fragments_keep", [])}
    site_labels = {str(value).casefold() for value in policy.get("site_labels_keep_conservative", [])}
    patterns = {
        key: re.compile(str(value), re.IGNORECASE) for key, value in (policy.get("patterns") or {}).items()
    }
    counts: Counter[str] = Counter()
    reviewed = []
    for candidate in candidates:
        classification = candidate["classification"]
        text = str(candidate["text"])
        canonical = text.casefold().strip()
        if classification == "blacklist":
            decision = "remove_retrieval_reviewed_blacklist"
        elif classification == "protected":
            decision = "keep_protected_legal"
        elif canonical in headers:
            decision = "keep_conservative_header_fragment"
        elif canonical in site_labels:
            decision = "keep_conservative_site_label"
        elif patterns.get("form_placeholder") and patterns["form_placeholder"].fullmatch(canonical):
            decision = "keep_form_placeholder"
        elif patterns.get("form_signature") and patterns["form_signature"].search(canonical):
            decision = "keep_form_signature"
        elif patterns.get("terminal_routing_candidate") and patterns["terminal_routing_candidate"].search(canonical):
            decision = "terminal_routing_candidate_structural_only"
        else:
            decision = str(policy.get("default_decision") or "keep_legal_or_ambiguous")
        counts[decision] += 1
        reviewed.append({**candidate, "reviewed": True, "decision": decision})
    write_json(output / "boilerplate_reviewed_candidates.json", reviewed)
    report: dict[str, object] = {
        "version": policy.get("version"),
        "status": policy.get("status"),
        "candidates": len(reviewed),
        "reviewed": sum(bool(value["reviewed"]) for value in reviewed),
        "pending": sum(not bool(value["reviewed"]) for value in reviewed),
        "decision_counts": dict(sorted(counts.items())),
        "policy": "high_df_is_review_signal_never_auto_delete",
        "policy_path": str(config.sanitation.review_policy_path),
    }
    write_json(output / "boilerplate_review_report.json", report)
    return report
