"""Task 2 Legal Question Answering contracts."""

from legal_qa.contracts import LegalQAPrediction
from legal_qa.metrics import evaluate_qa, meteor_score, rouge_l_score

__all__ = ["LegalQAPrediction", "evaluate_qa", "meteor_score", "rouge_l_score"]
