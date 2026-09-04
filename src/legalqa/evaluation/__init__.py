"""Diagnostics, metrics, evaluation workflows, and submission validation."""

from .diagnostics import build_diagnostic_labels
from .evaluator import (
    build_diagnostic_candidate_cache,
    evaluate_cached_qa,
    evaluate_diagnostic_evidence,
    evaluate_diagnostic_retrieval,
    evaluate_qa_files,
    evaluate_retrieval_complementarity,
    tune_dynamic_k_rules,
)
from .submission import validate_prediction_file

__all__ = [
    "build_diagnostic_candidate_cache",
    "build_diagnostic_labels",
    "evaluate_cached_qa",
    "evaluate_diagnostic_evidence",
    "evaluate_diagnostic_retrieval",
    "evaluate_qa_files",
    "evaluate_retrieval_complementarity",
    "tune_dynamic_k_rules",
    "validate_prediction_file",
]
