"""Test independent Task 2 answer contracts."""

import pytest
from legal_qa import LegalQAPrediction


def test_answer_serializes_to_codabench_item() -> None:
    prediction = LegalQAPrediction("q1", "Câu trả lời.")
    assert prediction.as_submission_item() == {"answer": "Câu trả lời."}


def test_empty_answer_is_rejected() -> None:
    with pytest.raises(ValueError):
        LegalQAPrediction("q1", " ")


def test_query_id_with_whitespace_is_rejected() -> None:
    with pytest.raises(ValueError):
        LegalQAPrediction("q 2", "Câu trả lời.")


def test_metrics_identical_strings() -> None:
    from legal_qa import evaluate_qa, meteor_score, rouge_l_score

    text = "Căn cứ Điều 10 Luật Đầu tư năm 2020."
    assert rouge_l_score(text, text) == 1.0
    assert meteor_score(text, text) > 0.99
    eval_res = evaluate_qa({"q1": text}, {"q1": text})
    assert eval_res["rouge_l"] == 1.0
    assert eval_res["meteor"] > 0.99
    assert eval_res["evaluated_pairs"] == 1.0


def test_metrics_empty_or_disjoint_strings() -> None:
    from legal_qa import evaluate_qa, meteor_score, rouge_l_score

    assert rouge_l_score("", "Văn bản") == 0.0
    assert meteor_score("Khác biệt hoàn toàn", "Văn bản mẫu") == 0.0
    eval_res = evaluate_qa({}, {"q1": "Đáp án"})
    assert eval_res["evaluated_pairs"] == 0.0


def test_rouge_l_invalid_beta() -> None:
    from legal_qa import rouge_l_score

    with pytest.raises(ValueError):
        rouge_l_score("a", "b", beta=0.0)
