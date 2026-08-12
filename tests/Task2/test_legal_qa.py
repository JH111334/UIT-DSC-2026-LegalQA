"""Test evidence requirements for Task 2 predictions."""

import pytest

from legal_qa import LegalQAPrediction


def test_answer_requires_bounded_evidence() -> None:
    prediction = LegalQAPrediction("q1", "Câu trả lời.", ("doc::c001",))
    assert not prediction.abstained


def test_answer_without_evidence_is_rejected() -> None:
    with pytest.raises(ValueError):
        LegalQAPrediction("q1", "Câu trả lời.", ())


def test_abstention_has_no_evidence() -> None:
    prediction = LegalQAPrediction("q2", "Không đủ bằng chứng.", (), abstained=True)
    assert prediction.abstained
