from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..core.io import iter_contexts, load_qa, write_json
from ..core.normalize import text_fingerprint
from ..core.stats import percentiles


def _percentiles(values: list[int]) -> dict[str, int]:
    return percentiles(values, (0, 25, 50, 75, 90, 95, 99, 100))


def analyze_dataset(
    data_dir: str | Path, output_path: str | Path | None = None
) -> dict[str, object]:
    root = Path(data_dir)
    train = load_qa(root / "train.json")
    test = load_qa(root / "public-official.json")
    question_fingerprints = Counter(text_fingerprint(item.question) for item in train)
    answer_fingerprints = Counter(text_fingerprint(item.answer) for item in train)
    context_fingerprints: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    context_count = 0
    empty_passages = 0
    recovered_names = 0
    passage_lengths: list[int] = []
    contexts = root / "selected-contexts"
    if not contexts.exists():
        contexts = root / "selected-contexts.zip"
    for document in iter_contexts(contexts):
        context_count += 1
        empty_passages += int(not document.passage)
        recovered_names += int("recovered_name" in document.flags)
        passage_lengths.append(len(document.passage))
        if document.passage:
            context_fingerprints[text_fingerprint(document.passage)] += 1
        document_type = document.metadata.get("document_type")
        if document_type:
            type_counts[str(document_type)] += 1
    report: dict[str, object] = {
        "train": {
            "examples": len(train),
            "empty_questions": sum(not item.question for item in train),
            "empty_answers": sum(not item.answer for item in train),
            "duplicate_question_extras": sum(
                value - 1 for value in question_fingerprints.values() if value > 1
            ),
            "duplicate_answer_extras": sum(
                value - 1 for value in answer_fingerprints.values() if value > 1
            ),
            "question_char_percentiles": _percentiles([len(item.question) for item in train]),
            "question_word_percentiles": _percentiles(
                [len(item.question.split()) for item in train]
            ),
            "answer_char_percentiles": _percentiles([len(item.answer or "") for item in train]),
            "answer_word_percentiles": _percentiles(
                [len((item.answer or "").split()) for item in train]
            ),
            "article_mentions": sum(
                len(re.findall(r"\bđiều\s+\d+", item.answer or "", re.I)) for item in train
            ),
        },
        "test": {
            "examples": len(test),
            "null_answers": sum(item.answer is None for item in test),
            "train_id_overlap": len({item.id for item in train} & {item.id for item in test}),
            "question_char_percentiles": _percentiles([len(item.question) for item in test]),
        },
        "contexts": {
            "documents": context_count,
            "empty_passages": empty_passages,
            "recovered_names": recovered_names,
            "exact_duplicate_extras": sum(
                value - 1 for value in context_fingerprints.values() if value > 1
            ),
            "passage_char_percentiles": _percentiles(passage_lengths),
            "document_types": dict(type_counts.most_common()),
        },
    }
    if output_path:
        write_json(output_path, report)
    return report
