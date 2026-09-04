from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from typing import Any

from ..core.config import PipelineConfig, artifact_dir, data_path
from ..core.io import load_qa, read_jsonl, write_json, write_jsonl
from ..core.metadata import LegalCitation, canonical_document_number, extract_citations


def _citation_dict(citation: LegalCitation) -> dict[str, str | None]:
    return {
        "raw": citation.raw,
        "document_number": citation.document_number,
        "document_type": citation.document_type,
        "article": citation.article,
        "clause": citation.clause,
        "point": citation.point,
        "key": citation.key,
    }


def build_diagnostic_labels(config: PipelineConfig) -> dict[str, object]:
    if config.diagnostic.allow_training_use:
        raise ValueError("Citation-derived labels are diagnostic-only")
    output = artifact_dir(config)
    examples = load_qa(data_path(config, config.data.train_file))

    parents_by_doc_article: dict[tuple[str, str], list[str]] = defaultdict(list)
    chunks_by_specificity: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for value in read_jsonl(output / "parents.jsonl"):
        number = canonical_document_number(value.get("document_number"))
        article = value.get("article")
        if number and article:
            parents_by_doc_article[(number, str(article).casefold())].append(str(value["id"]))
    for value in read_jsonl(output / "chunks.jsonl"):
        number = canonical_document_number(value.get("document_number"))
        article = value.get("article")
        if number and article:
            key = (
                number,
                str(article).casefold(),
                str(value.get("clause") or "").casefold(),
                str(value.get("point") or "").casefold(),
            )
            chunks_by_specificity[key].append(str(value["id"]))

    records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for example in examples:
        citations = extract_citations(example.answer or "")
        matched_parents: set[str] = set()
        matched_chunks: set[str] = set()
        matched_citations = 0
        for citation in citations:
            if not citation.document_number:
                continue
            parent_ids = parents_by_doc_article.get((citation.document_number, citation.article), [])
            if parent_ids:
                matched_citations += 1
                matched_parents.update(parent_ids)
            exact_key = (
                citation.document_number,
                citation.article,
                citation.clause or "",
                citation.point or "",
            )
            specific = chunks_by_specificity.get(exact_key, [])
            if not specific and citation.clause:
                specific = chunks_by_specificity.get(
                    (citation.document_number, citation.article, citation.clause, ""), []
                )
            matched_chunks.update(specific)
        if not citations:
            status = "no_citation"
        elif matched_parents:
            status = "matched"
        elif any(citation.document_number for citation in citations):
            status = "document_or_article_not_found"
        else:
            status = "citation_without_document_number"
        status_counts[status] += 1
        records.append(
            {
                "example_id": example.id,
                "question": example.question,
                "citations": [_citation_dict(value) for value in citations],
                "matched_parent_ids": sorted(matched_parents),
                "matched_chunk_ids": sorted(matched_chunks),
                "matched_citation_count": matched_citations,
                "status": status,
                "diagnostic_only": True,
                "training_use_prohibited": True,
            }
        )

    write_jsonl(output / "diagnostic_labels.jsonl", records)
    eligible = [record for record in records if record["status"] == "matched"]
    rng = random.Random(config.diagnostic.seed)
    sample = rng.sample(eligible, min(config.diagnostic.manual_sample_size, len(eligible)))
    manual_path = output / "diagnostic_manual_validation.csv"
    with manual_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "example_id",
                "question",
                "citations",
                "matched_parent_ids",
                "is_correct",
                "reviewer_note",
            ),
        )
        writer.writeheader()
        for record in sample:
            writer.writerow(
                {
                    "example_id": record["example_id"],
                    "question": record["question"],
                    "citations": record["citations"],
                    "matched_parent_ids": "|".join(record["matched_parent_ids"]),
                    "is_correct": "",
                    "reviewer_note": "",
                }
            )
    report: dict[str, object] = {
        "examples": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "matched_coverage": round(len(eligible) / max(1, len(records)), 6),
        "manual_validation_samples": len(sample),
        "diagnostic_only": True,
        "training_use_prohibited": True,
        "allowed_uses": ["retrieval evaluation", "ablation", "error analysis"],
        "disallowed_uses": ["retriever fine-tuning", "reranker fine-tuning", "generator fine-tuning"],
    }
    write_json(output / "diagnostic_report.json", report)
    return report
