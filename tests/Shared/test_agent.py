"""Test bounded agent route selection."""

from pathlib import Path

from text_retrieval_agent.agent import EvidenceAgent
from text_retrieval_agent.contracts import RetrievalQuery
from text_retrieval_agent.pipeline import RetrievalPipeline

ROOT = Path(__file__).parents[2]


def test_compare_route_uses_at_most_two_evidence_items() -> None:
    """Use the constrained comparison route for an explicit comparison."""
    pipeline = RetrievalPipeline(
        ROOT / "Data/Shared/fixtures/documents.jsonl",
        ROOT / "configs/Shared/smoke.toml",
    )
    answer = EvidenceAgent(pipeline).ask(
        RetrievalQuery(
            query_id="q_compare",
            text="Compare retrieval evaluation and agent retrieval contracts",
            top_k=5,
        )
    )
    assert answer.route == "compare"
    assert answer.answer.count("\n- ") == 2
