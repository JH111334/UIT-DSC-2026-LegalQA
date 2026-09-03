"""Provide transparent local diagnostics for Task 2 LegalQA (METEOR and ROUGE-L)."""

from __future__ import annotations

import re
from collections.abc import Mapping

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text into normalized lowercase word tokens."""
    return TOKEN_PATTERN.findall(text.casefold())


def lcs_length(a: list[str], b: list[str]) -> int:
    """Compute Longest Common Subsequence length using space-optimized DP."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            curr = dp[j + 1]
            if x == y:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = curr
    return dp[len(b)]


def rouge_l_score(hypothesis: str, reference: str, *, beta: float = 1.2) -> float:
    """Compute sentence-level ROUGE-L F-score."""
    if beta <= 0:
        raise ValueError("beta must be positive.")
    h_tokens = tokenize(hypothesis)
    r_tokens = tokenize(reference)
    if not h_tokens or not r_tokens:
        return 0.0
    lcs = lcs_length(h_tokens, r_tokens)
    precision = lcs / len(h_tokens)
    recall = lcs / len(r_tokens)
    beta_sq = beta * beta
    denominator = recall + beta_sq * precision
    if denominator == 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denominator


def meteor_score(
    hypothesis: str,
    reference: str,
    *,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """Compute METEOR score with unigram precision/recall and chunk fragmentation penalty."""
    h_tokens = tokenize(hypothesis)
    r_tokens = tokenize(reference)
    if not h_tokens or not r_tokens:
        return 0.0

    matched_r: set[int] = set()
    matches: list[tuple[int, int]] = []
    for i, ht in enumerate(h_tokens):
        for j, rt in enumerate(r_tokens):
            if j not in matched_r and ht == rt:
                matched_r.add(j)
                matches.append((i, j))
                break

    m = len(matches)
    if m == 0:
        return 0.0

    precision = m / len(h_tokens)
    recall = m / len(r_tokens)
    denom = alpha * precision + (1.0 - alpha) * recall
    if denom == 0.0:
        return 0.0
    f_mean = (precision * recall) / denom

    matches.sort(key=lambda item: item[0])
    chunks = 1
    for k in range(1, len(matches)):
        if matches[k][1] != matches[k - 1][1] + 1:
            chunks += 1

    penalty = float(gamma * ((chunks / m) ** beta))
    return float(max(0.0, f_mean * (1.0 - penalty)))


def evaluate_qa(
    predictions: Mapping[str, str],
    references: Mapping[str, str],
) -> dict[str, float]:
    """Compute macro-averaged METEOR and ROUGE-L across paired question IDs."""
    common_ids = sorted(set(predictions) & set(references))
    if not common_ids:
        return {"meteor": 0.0, "rouge_l": 0.0, "evaluated_pairs": 0.0}

    meteor_total = 0.0
    rouge_l_total = 0.0
    for qid in common_ids:
        hyp = predictions[qid]
        ref = references[qid]
        meteor_total += meteor_score(hyp, ref)
        rouge_l_total += rouge_l_score(hyp, ref)

    count = float(len(common_ids))
    return {
        "meteor": meteor_total / count,
        "rouge_l": rouge_l_total / count,
        "evaluated_pairs": count,
    }
