"""Provide bilingual normalization, BM25, and TF-IDF scoring."""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "cach",
    "cho",
    "cua",
    "how",
    "is",
    "la",
    "must",
    "of",
    "the",
    "to",
    "what",
    "when",
    "why",
}


def normalize_text(text: str) -> str:
    """Normalize Vietnamese and English text for exact sparse retrieval."""
    folded = unicodedata.normalize("NFKD", text.casefold()).replace("đ", "d")
    return "".join(character for character in folded if not unicodedata.combining(character))


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize normalized alphanumeric terms."""
    return tuple(
        token for token in TOKEN_PATTERN.findall(normalize_text(text)) if token not in STOP_WORDS
    )


def _inverse_document_frequency(documents: Sequence[tuple[str, ...]]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    count = len(documents)
    return {
        term: math.log((1 + count) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }


class BM25Index:
    """Rank a small corpus with exact in-memory BM25."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        """Build term frequencies and BM25 inverse document frequencies."""
        if not documents:
            raise ValueError("documents must not be empty.")
        self._k1 = k1
        self._b = b
        self._tokens = [tokenize(document) for document in documents]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        count = len(self._tokens)
        frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            frequency.update(set(tokens))
        self._idf = {
            term: math.log(1 + (count - value + 0.5) / (value + 0.5))
            for term, value in frequency.items()
        }

    def scores(self, query: str) -> tuple[float, ...]:
        """Score every indexed document for one query."""
        terms = tokenize(query)
        scores: list[float] = []
        for frequencies, length in zip(self._frequencies, self._lengths, strict=True):
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self._k1 + 1)
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / max(self._average_length, 1)
                )
                score += self._idf.get(term, 0.0) * numerator / denominator
            scores.append(score)
        return tuple(scores)


class TfidfIndex:
    """Rank a small corpus by exact sparse TF-IDF cosine similarity."""

    def __init__(self, documents: Sequence[str]) -> None:
        """Build normalized sparse document vectors."""
        if not documents:
            raise ValueError("documents must not be empty.")
        tokenized = [tokenize(document) for document in documents]
        self._idf = _inverse_document_frequency(tokenized)
        self._vectors = [self._vector(tokens) for tokens in tokenized]

    def _vector(self, tokens: Iterable[str]) -> dict[str, float]:
        frequencies = Counter(tokens)
        vector = {term: count * self._idf.get(term, 0.0) for term, count in frequencies.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {term: value / norm for term, value in vector.items()} if norm else {}

    def scores(self, query: str) -> tuple[float, ...]:
        """Score every indexed document by sparse cosine similarity."""
        query_vector = self._vector(tokenize(query))
        return tuple(
            sum(query_vector.get(term, 0.0) * value for term, value in vector.items())
            for vector in self._vectors
        )
