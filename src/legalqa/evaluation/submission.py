from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.io import load_qa


def validate_prediction_file(
    input_path: str | Path,
    predictions_path: str | Path,
) -> dict[str, Any]:
    examples = {value.id: value for value in load_qa(input_path)}
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8-sig"))
    if not isinstance(predictions, dict):
        raise ValueError("Prediction file must contain a JSON object")

    expected_ids = set(examples)
    prediction_ids = {str(value) for value in predictions}
    missing_ids = sorted(expected_ids - prediction_ids)
    extra_ids = sorted(prediction_ids - expected_ids)
    invalid_record_ids = []
    question_mismatch_ids = []
    blank_answer_ids = []
    answer_lengths = []
    for example_id in sorted(expected_ids & prediction_ids):
        value = predictions[example_id]
        if not isinstance(value, dict):
            invalid_record_ids.append(example_id)
            continue
        if value.get("question") != examples[example_id].question:
            question_mismatch_ids.append(example_id)
        answer = value.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            blank_answer_ids.append(example_id)
        else:
            answer_lengths.append(len(answer.split()))

    valid = not any(
        (missing_ids, extra_ids, invalid_record_ids, question_mismatch_ids, blank_answer_ids)
    )
    return {
        "valid": valid,
        "expected": len(expected_ids),
        "predictions": len(prediction_ids),
        "missing_count": len(missing_ids),
        "extra_count": len(extra_ids),
        "invalid_record_count": len(invalid_record_ids),
        "question_mismatch_count": len(question_mismatch_ids),
        "blank_answer_count": len(blank_answer_ids),
        "answer_words_min": min(answer_lengths, default=0),
        "answer_words_max": max(answer_lengths, default=0),
        "answer_words_average": sum(answer_lengths) / max(1, len(answer_lengths)),
        "error_samples": {
            "missing": missing_ids[:10],
            "extra": extra_ids[:10],
            "invalid_record": invalid_record_ids[:10],
            "question_mismatch": question_mismatch_ids[:10],
            "blank_answer": blank_answer_ids[:10],
        },
    }
