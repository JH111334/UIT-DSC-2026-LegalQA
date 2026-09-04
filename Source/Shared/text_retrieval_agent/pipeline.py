"""Orchestrate scoped sparse retrieval and deterministic fusion."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from text_retrieval_agent.contracts import (
    RetrievalQuery,
    RetrieverCandidate,
    RetrieverStatus,
    SearchResponse,
)
from text_retrieval_agent.corpus import chunk_documents, load_documents
from text_retrieval_agent.fusion import reciprocal_rank_fusion
from text_retrieval_agent.retrievers import BM25Retriever, TfidfRetriever


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Control chunking, retrieval, and agent budgets."""

    version: str
    rrf_k: int
    default_top_k: int
    max_sentences: int
    max_query_characters: int
    max_candidates_per_retriever: int
    max_agent_steps: int
    max_evidence_chunks: int


def load_config(path: Path) -> PipelineConfig:
    """Load one versioned TOML configuration."""
    with path.open("rb") as source:
        payload: dict[str, Any] = tomllib.load(source)
    return PipelineConfig(
        version=str(payload["config_version"]),
        rrf_k=int(payload["rrf_k"]),
        default_top_k=int(payload["default_top_k"]),
        max_sentences=int(payload["chunking"]["max_sentences"]),
        max_query_characters=int(payload["limits"]["max_query_characters"]),
        max_candidates_per_retriever=int(payload["limits"]["max_candidates_per_retriever"]),
        max_agent_steps=int(payload["limits"]["max_agent_steps"]),
        max_evidence_chunks=int(payload["limits"]["max_evidence_chunks"]),
    )


class RetrievalPipeline:
    """Run access-scoped BM25 and TF-IDF before rank fusion."""

    def __init__(self, corpus_path: Path, config_path: Path) -> None:
        """Load one immutable corpus and configuration snapshot."""
        self.config = load_config(config_path)
        documents = load_documents(corpus_path)
        self.chunks = chunk_documents(documents, max_sentences=self.config.max_sentences)
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._bm25 = BM25Retriever(self.chunks)
        self._tfidf = TfidfRetriever(self.chunks)

    def _run_retriever(
        self,
        name: str,
        query: RetrievalQuery,
    ) -> tuple[tuple[RetrieverCandidate, ...], RetrieverStatus]:
        try:
            retriever = self._bm25 if name == "bm25" else self._tfidf
            candidates = retriever.search(query, self.config.max_candidates_per_retriever)
            state = "ok" if candidates else "skipped"
            detail = "candidates generated" if candidates else "no authorized lexical match"
            return candidates, RetrieverStatus(name, state, len(candidates), detail)  # type: ignore[arg-type]
        except (OSError, ValueError) as error:
            return (), RetrieverStatus(name, "failed", 0, str(error))  # type: ignore[arg-type]

    def search(self, query: RetrievalQuery) -> SearchResponse:
        """Search authorized chunks and return evidence-bearing results."""
        if len(query.text) > self.config.max_query_characters:
            raise ValueError("Query exceeds max_query_characters.")
        started = time.perf_counter()
        bm25, bm25_status = self._run_retriever("bm25", query)
        tfidf, tfidf_status = self._run_retriever("tfidf", query)
        results = reciprocal_rank_fusion(
            {"bm25": bm25, "tfidf": tfidf},
            self._chunks_by_id,
            rrf_k=self.config.rrf_k,
            top_k=query.top_k or self.config.default_top_k,
            config_version=self.config.version,
        )
        return SearchResponse(
            query_id=query.query_id,
            results=results,
            retriever_status=(bm25_status, tfidf_status),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
