"""Apply a bounded evidence policy after retrieval."""

from text_retrieval_agent.contracts import (
    AgentAnswer,
    Citation,
    RetrievalQuery,
    ToolEvent,
)
from text_retrieval_agent.pipeline import RetrievalPipeline
from text_retrieval_agent.text import normalize_text

COMPARE_TERMS = ("compare", "difference", "so sanh", "khac nhau")


class EvidenceAgent:
    """Route retrieval, cite authorized chunks, and abstain safely."""

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        """Bind the agent to one retrieval snapshot."""
        self._pipeline = pipeline

    @staticmethod
    def _route(query: RetrievalQuery, result_count: int) -> str:
        normalized = normalize_text(query.text)
        if not result_count:
            return "abstain"
        if any(term in normalized for term in COMPARE_TERMS) and result_count > 1:
            return "compare"
        return "lookup"

    def ask(self, query: RetrievalQuery) -> AgentAnswer:
        """Run one bounded retrieval step and assemble extractive evidence."""
        response = self._pipeline.search(query)
        selected = response.results[: self._pipeline.config.max_evidence_chunks]
        event = ToolEvent(
            step=1,
            tool="hybrid_search",
            state="ok" if selected else "skipped",
            output_ids=tuple(result.chunk_id for result in selected),
        )
        route = self._route(query, len(selected))
        if route == "abstain":
            return AgentAnswer(
                query_id=query.query_id,
                answer="Insufficient authorized evidence to answer.",
                citations=(),
                route="abstain",
                abstained=True,
                tool_events=(event,),
            )
        citations = tuple(
            Citation(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                source_uri=result.source_uri,
                quote=result.excerpt,
            )
            for result in selected
        )
        if route == "compare":
            answer = "Comparison evidence:\n" + "\n".join(
                f"- {citation.quote} [{citation.chunk_id}]" for citation in citations[:2]
            )
        else:
            answer = f"{citations[0].quote} [{citations[0].chunk_id}]"
        return AgentAnswer(
            query_id=query.query_id,
            answer=answer,
            citations=citations,
            route=route,  # type: ignore[arg-type]
            abstained=False,
            tool_events=(event,),
        )
