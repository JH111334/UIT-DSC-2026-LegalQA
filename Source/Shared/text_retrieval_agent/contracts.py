"""Define validated corpus, retrieval, citation, and agent records."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

RetrieverName = Literal["bm25", "tfidf"]
RunState = Literal["ok", "skipped", "failed"]
AgentRoute = Literal["lookup", "compare", "abstain"]


class ContractError(ValueError):
    """Report invalid persisted or runtime text-retrieval data."""


def _require_identifier(name: str, value: str) -> None:
    if not value or value.strip() != value or any(character.isspace() for character in value):
        raise ContractError(f"{name} must be a non-empty identifier without whitespace.")


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Represent one governed source document."""

    document_id: str
    title: str
    body: str
    language: str
    source_uri: str
    access_scope: str
    corpus_version: str
    metadata: Mapping[str, str]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate identity, source, scope, and provenance."""
        _require_identifier("document_id", self.document_id)
        _require_identifier("corpus_version", self.corpus_version)
        if not self.title or not self.body or not self.source_uri or not self.access_scope:
            raise ContractError("title, body, source_uri, and access_scope are required.")
        if not self.provenance.get("source"):
            raise ContractError("provenance.source is required.")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """Represent one deterministic, source-backed document chunk."""

    chunk_id: str
    document_id: str
    position: int
    text: str
    title: str
    language: str
    source_uri: str
    access_scope: str
    corpus_version: str


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Describe one access-scoped text query."""

    query_id: str
    text: str
    allowed_scopes: tuple[str, ...] = ("public",)
    filters: Mapping[str, str] | None = None
    top_k: int = 5

    def __post_init__(self) -> None:
        """Validate query identity, scope, content, and budget."""
        _require_identifier("query_id", self.query_id)
        if not self.text.strip():
            raise ContractError("text must not be empty.")
        if not self.allowed_scopes:
            raise ContractError("allowed_scopes must not be empty.")
        if self.top_k < 1 or self.top_k > 100:
            raise ContractError("top_k must be between 1 and 100.")


@dataclass(frozen=True, slots=True)
class RetrieverCandidate:
    """Carry one retriever score without fusion overwrite."""

    chunk_id: str
    retriever: RetrieverName
    raw_score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RetrieverStatus:
    """Expose success, skip, or failure for one retriever."""

    retriever: RetrieverName
    state: RunState
    candidate_count: int
    detail: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Return one fused chunk with source and score evidence."""

    chunk_id: str
    document_id: str
    title: str
    excerpt: str
    source_uri: str
    access_scope: str
    rank: int
    final_score: float
    component_scores: Mapping[str, float]
    component_ranks: Mapping[str, int]
    corpus_version: str
    retrieval_config_version: str


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Return ranked chunks, retriever health, and observed latency."""

    query_id: str
    results: tuple[RetrievedChunk, ...]
    retriever_status: tuple[RetrieverStatus, ...]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class Citation:
    """Bind answer evidence to one retrieved source chunk."""

    document_id: str
    chunk_id: str
    source_uri: str
    quote: str


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """Record one bounded agent tool execution."""

    step: int
    tool: str
    state: RunState
    output_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """Return an extractive answer, citations, route, and trace."""

    query_id: str
    answer: str
    citations: tuple[Citation, ...]
    route: AgentRoute
    abstained: bool
    tool_events: tuple[ToolEvent, ...]
