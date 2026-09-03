"""Expose dependency-light Task 2 retrieval and answer contracts."""

from .batch import AnswerResult, BatchInferenceRunner, QuestionRecord
from .bm25 import BM25Hit, BM25Index, build_bm25_index, evaluate_bm25
from .engine import BM25Evidence, NoEvidence, TransformersAnswerEngine
from .fusion import FusedCandidate, reciprocal_rank_fusion

__all__ = [
    "AnswerResult",
    "BM25Evidence",
    "BM25Hit",
    "BM25Index",
    "BatchInferenceRunner",
    "FusedCandidate",
    "NoEvidence",
    "QuestionRecord",
    "TransformersAnswerEngine",
    "build_bm25_index",
    "evaluate_bm25",
    "reciprocal_rank_fusion",
]
