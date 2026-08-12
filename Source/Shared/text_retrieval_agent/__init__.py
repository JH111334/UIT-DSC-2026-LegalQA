"""Expose the text retrieval agent public API."""

from text_retrieval_agent.agent import EvidenceAgent
from text_retrieval_agent.contracts import AgentAnswer, RetrievalQuery, SearchResponse
from text_retrieval_agent.pipeline import RetrievalPipeline

__all__ = [
    "AgentAnswer",
    "EvidenceAgent",
    "RetrievalPipeline",
    "RetrievalQuery",
    "SearchResponse",
]
