"""Provide transparent local diagnostics; organizer scoring remains authoritative."""

from collections.abc import Mapping, Sequence


def macro_fbeta(
    predictions: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Sequence[str]],
    *,
    beta: float = 2.0,
) -> float:
    """Compute query-level macro F-beta over document-ID sets."""
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if not relevant:
        return 0.0
    scores: list[float] = []
    beta_squared = beta * beta
    for query_id, gold_items in relevant.items():
        gold = set(gold_items)
        predicted = set(predictions.get(query_id, ()))
        true_positive = len(gold & predicted)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(gold) if gold else 0.0
        denominator = beta_squared * precision + recall
        score = (1 + beta_squared) * precision * recall / denominator if denominator else 0.0
        scores.append(score)
    return sum(scores) / len(scores)
