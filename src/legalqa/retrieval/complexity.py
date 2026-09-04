from __future__ import annotations

import math
import re

from ..core.config import ComplexityConfig
from ..core.metadata import extract_citations
from ..core.normalize import normalize_text
from ..core.schema import QueryComplexity, RetrievalConfidence

_ENTITY_RE = re.compile(
    r"\b(người\s+lao\s+động|người\s+sử\s+dụng\s+lao\s+động|người|cá\s+nhân|"
    r"tổ\s+chức|doanh\s+nghiệp|cơ\s+quan|đơn\s+vị|bên\s+mua|bên\s+bán|"
    r"người\s+vi\s+phạm|người\s+nộp\s+thuế)\b",
    re.IGNORECASE,
)
_CONDITION_RE = re.compile(
    r"\b(nếu|khi|trong\s+trường\s+hợp|trừ\s+trường\s+hợp|với\s+điều\s+kiện|"
    r"tùy\s+theo|đồng\s+thời|ngoại\s+trừ)\b",
    re.IGNORECASE,
)
_CONJUNCTION_RE = re.compile(r"\b(và|hoặc|nhưng|cũng\s+như|hay)\b", re.IGNORECASE)


def analyze_query_complexity(question: str, config: ComplexityConfig) -> QueryComplexity:
    text = normalize_text(question, preserve_newlines=False)
    entities = len({match.group(0).casefold() for match in _ENTITY_RE.finditer(text)})
    conditions = len(_CONDITION_RE.findall(text))
    citations = len(extract_citations(text))
    conjunctions = len(_CONJUNCTION_RE.findall(text))
    tokens = len(text.split())
    length_feature = min(3.0, max(0.0, (tokens - 12) / 20.0))
    score = (
        config.entity_weight * min(entities, 4)
        + config.condition_weight * min(conditions, 4)
        + config.citation_weight * min(citations, 3)
        + config.conjunction_weight * min(conjunctions, 5)
        + config.length_weight * length_feature
    )
    if score < config.low_threshold:
        top_k = config.complexity_top_k[0]
    elif score < config.high_threshold:
        top_k = config.complexity_top_k[1]
    else:
        top_k = config.complexity_top_k[2]
    return QueryComplexity(
        score=round(score, 6),
        top_k=top_k,
        features={
            "entities": float(entities),
            "conditions": float(conditions),
            "citations": float(citations),
            "conjunctions": float(conjunctions),
            "tokens": float(tokens),
            "length_feature": round(length_feature, 6),
        },
    )


def analyze_retrieval_confidence(scores: list[float], max_rank: int = 8) -> RetrievalConfidence:
    if not scores:
        return RetrievalConfidence(0.0, 1.0, 0.0, 1.0, 0)
    values = scores[:max_rank]
    if len(values) == 1:
        return RetrievalConfidence(1.0, 0.0, 1.0, 0.0, 1)
    if all(0.0 <= value <= 1.0 for value in values):
        normalized = values
    else:
        normalized = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value)))) for value in values]
    gaps = [max(0.0, normalized[i] - normalized[i + 1]) for i in range(len(normalized) - 1)]
    largest_gap = max(gaps)
    elbow = gaps.index(largest_gap) + 1
    total = sum(max(value, 0.0) for value in normalized) or 1.0
    tail_mass = sum(max(value, 0.0) for value in normalized[3:]) / total
    top_gap = gaps[0] if gaps else 0.0
    return RetrievalConfidence(
        concentration=round(largest_gap, 6),
        ambiguity=round(1.0 - largest_gap, 6),
        top_gap=round(top_gap, 6),
        tail_mass=round(tail_mass, 6),
        elbow_position=elbow,
    )


def choose_dynamic_top_k(
    complexity: QueryComplexity,
    confidence: RetrievalConfidence,
    config: ComplexityConfig,
) -> int:
    if not config.enabled:
        return config.complexity_top_k[1]
    selected = complexity.top_k
    if not config.confidence_enabled or confidence.elbow_position == 0:
        return selected
    if confidence.concentration >= config.confident_gap and confidence.elbow_position <= config.confident_max_k:
        return max(config.confident_min_k, min(selected, confidence.elbow_position))
    if confidence.concentration <= config.ambiguous_gap:
        return max(selected, config.ambiguous_min_k)
    return selected
