"""Shared data contracts used across pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QAExample:
    id: str
    question: str
    answer: str | None = None


@dataclass(slots=True)
class LegalDocument:
    id: str
    name: str
    passage: str
    link: str = ""
    original_name: str = ""
    source_ids: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    retrieval_text: str = ""
    generation_text: str = ""

    def __post_init__(self) -> None:
        if not self.raw_text:
            self.raw_text = self.passage
        if not self.retrieval_text:
            self.retrieval_text = self.passage
        if not self.generation_text:
            self.generation_text = self.passage

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # ``passage`` is a backwards-compatible in-memory alias. Persist only
        # the three explicit v2 representations to avoid a fourth full copy.
        value.pop("passage", None)
        value["source_ids"] = list(self.source_ids)
        value["flags"] = list(self.flags)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LegalDocument:
        return cls(
            id=str(value["id"]),
            name=str(value.get("name") or ""),
            passage=str(value.get("generation_text") or value.get("passage") or value.get("raw_text") or ""),
            link=str(value.get("link") or ""),
            original_name=str(value.get("original_name") or ""),
            source_ids=tuple(str(x) for x in value.get("source_ids", ())),
            flags=tuple(str(x) for x in value.get("flags", ())),
            metadata=dict(value.get("metadata") or {}),
            raw_text=str(value.get("raw_text") or value.get("passage") or ""),
            retrieval_text=str(value.get("retrieval_text") or value.get("passage") or ""),
            generation_text=str(value.get("generation_text") or value.get("passage") or ""),
        )


@dataclass(slots=True)
class ParentSection:
    id: str
    document_id: str
    text: str
    level: str
    heading: str
    document_type: str | None = None
    document_number: str | None = None
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    path: tuple[str, ...] = ()
    source_name: str = ""
    references: tuple[str, ...] = ()
    amendments: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("path", "references", "amendments"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ParentSection:
        return cls(
            id=str(value["id"]),
            document_id=str(value["document_id"]),
            text=str(value["text"]),
            level=str(value["level"]),
            heading=str(value.get("heading") or ""),
            document_type=value.get("document_type"),
            document_number=value.get("document_number"),
            chapter=value.get("chapter"),
            section=value.get("section"),
            article=value.get("article"),
            path=tuple(str(x) for x in value.get("path", ())),
            source_name=str(value.get("source_name") or ""),
            references=tuple(str(x) for x in value.get("references", ())),
            amendments=tuple(str(x) for x in value.get("amendments", ())),
        )


@dataclass(slots=True)
class LegalChunk:
    id: str
    document_id: str
    parent_id: str
    retrieval_text: str
    level: str
    heading: str
    document_type: str | None = None
    document_number: str | None = None
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    clause: str | None = None
    point: str | None = None
    path: tuple[str, ...] = ()
    source_name: str = ""
    references: tuple[str, ...] = ()
    amendments: tuple[str, ...] = ()
    ordinal: int = 0

    @property
    def text(self) -> str:
        return self.retrieval_text

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("path", "references", "amendments"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LegalChunk:
        return cls(
            id=str(value["id"]),
            document_id=str(value["document_id"]),
            parent_id=str(value["parent_id"]),
            retrieval_text=str(value.get("retrieval_text") or value.get("text") or ""),
            level=str(value["level"]),
            heading=str(value.get("heading") or ""),
            document_type=value.get("document_type"),
            document_number=value.get("document_number"),
            chapter=value.get("chapter"),
            section=value.get("section"),
            article=value.get("article"),
            clause=value.get("clause"),
            point=value.get("point"),
            path=tuple(str(x) for x in value.get("path", ())),
            source_name=str(value.get("source_name") or ""),
            references=tuple(str(x) for x in value.get("references", ())),
            amendments=tuple(str(x) for x in value.get("amendments", ())),
            ordinal=int(value.get("ordinal", 0)),
        )


@dataclass(slots=True)
class Candidate:
    chunk_id: str
    score: float = 0.0
    channel_scores: dict[str, float] = field(default_factory=dict)
    channel_ranks: dict[str, int] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()


@dataclass(slots=True)
class QueryComplexity:
    score: float
    top_k: int
    features: dict[str, float]


@dataclass(slots=True)
class RetrievalConfidence:
    concentration: float
    ambiguity: float
    top_gap: float
    tail_mass: float
    elbow_position: int


@dataclass(slots=True)
class Evidence:
    id: str
    text: str
    score: float
    document_id: str
    source_name: str
    provenance: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
