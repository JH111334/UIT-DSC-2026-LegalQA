from __future__ import annotations

from ..core.normalize import normalize_text


def _tokens(text: str) -> list[str]:
    return normalize_text(text, preserve_newlines=False).casefold().split()


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted, gold = _tokens(prediction), _tokens(reference)
    if not predicted or not gold:
        return 0.0
    previous = [0] * (len(gold) + 1)
    for predicted_token in predicted:
        current = [0]
        for index, gold_token in enumerate(gold, 1):
            if predicted_token == gold_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(predicted), lcs / len(gold)
    return 2 * precision * recall / max(1e-12, precision + recall)


def meteor_exact(prediction: str, reference: str) -> float:
    """Language-neutral exact-token METEOR (no English WordNet/stemmer)."""

    predicted, gold = _tokens(prediction), _tokens(reference)
    if not predicted or not gold:
        return 0.0
    available: dict[str, list[int]] = {}
    for index, token in enumerate(gold):
        available.setdefault(token, []).append(index)
    matched_positions: list[int] = []
    for token in predicted:
        positions = available.get(token)
        if positions:
            matched_positions.append(positions.pop(0))
    matches = len(matched_positions)
    if not matches:
        return 0.0
    precision, recall = matches / len(predicted), matches / len(gold)
    harmonic = 10 * precision * recall / max(1e-12, recall + 9 * precision)
    chunks = 1 + sum(
        current != previous + 1
        for previous, current in zip(matched_positions, matched_positions[1:])
    )
    penalty = 0.5 * (chunks / matches) ** 3
    return harmonic * (1.0 - penalty)
