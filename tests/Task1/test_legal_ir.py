"""Test Task 1 submission contracts and local diagnostics."""

import pytest

from legal_ir import LegalIRPrediction, macro_fbeta


def test_prediction_preserves_ranked_document_ids() -> None:
    prediction = LegalIRPrediction("q1", ("d2", "d1"))
    assert prediction.document_ids == ("d2", "d1")


def test_prediction_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        LegalIRPrediction("q1", ("d1", "d1"))


def test_macro_f2_is_query_averaged() -> None:
    score = macro_fbeta(
        {"q1": ("d1",), "q2": ("wrong",)},
        {"q1": ("d1",), "q2": ("d2",)},
    )
    assert score == 0.5
