"""Replay retrieval, access, citation, and abstention behavior."""

from pathlib import Path

from text_retrieval_agent.agent import EvidenceAgent
from text_retrieval_agent.contracts import RetrievalQuery
from text_retrieval_agent.evaluation import evaluate
from text_retrieval_agent.pipeline import RetrievalPipeline

ROOT = Path(__file__).parents[2]


def _pipeline() -> RetrievalPipeline:
    return RetrievalPipeline(
        ROOT / "Data/Shared/fixtures/documents.jsonl",
        ROOT / "configs/Shared/smoke.toml",
    )


def test_vietnamese_query_retrieves_release_evidence() -> None:
    """Retrieve the expected Vietnamese runbook with both sparse scores."""
    response = _pipeline().search(
        RetrievalQuery(
            query_id="q1",
            text="Cách quay lại phiên bản ổn định khi triển khai lỗi?",
            top_k=3,
        )
    )
    assert response.results[0].document_id == "doc_release_rollback"
    assert set(response.results[0].component_scores) == {"bm25", "tfidf"}


def test_public_scope_excludes_internal_document() -> None:
    """Prevent restricted content from entering public candidates."""
    pipeline = _pipeline()
    public = pipeline.search(
        RetrievalQuery(query_id="q2", text="Aurora acquisition budget", top_k=5)
    )
    internal = pipeline.search(
        RetrievalQuery(
            query_id="q3",
            text="Aurora acquisition budget",
            allowed_scopes=("internal",),
            top_k=5,
        )
    )
    assert all(result.document_id != "doc_aurora_budget" for result in public.results)
    assert internal.results[0].document_id == "doc_aurora_budget"


def test_agent_cites_only_retrieved_chunks() -> None:
    """Assemble citations from the bounded authorized result set."""
    query = RetrievalQuery(
        query_id="q4",
        text="Why must the agent cite retrieved chunks and abstain?",
        top_k=3,
    )
    pipeline = _pipeline()
    retrieved_ids = {result.chunk_id for result in pipeline.search(query).results}
    answer = EvidenceAgent(pipeline).ask(query)
    assert not answer.abstained
    assert answer.citations
    assert all(citation.chunk_id in retrieved_ids for citation in answer.citations)
    assert len(answer.tool_events) == 1


def test_agent_abstains_without_evidence() -> None:
    """Refuse an answer when no authorized chunk matches."""
    answer = EvidenceAgent(_pipeline()).ask(
        RetrievalQuery(query_id="q5", text="What is the office WiFi password?")
    )
    assert answer.abstained
    assert answer.route == "abstain"
    assert not answer.citations


def test_fixture_evaluation_is_reproducible() -> None:
    """Meet retrieval, abstention, and citation smoke thresholds."""
    metrics = evaluate(
        _pipeline(),
        ROOT / "Data/Shared/fixtures/queries.jsonl",
        ROOT / "Data/Shared/fixtures/qrels.jsonl",
        top_k=3,
    )
    assert metrics["query_count"] == 4
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["citation_integrity"] == 1.0
