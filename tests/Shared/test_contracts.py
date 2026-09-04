"""Test query and corpus contract rejection."""

import pytest
from text_retrieval_agent.contracts import ContractError, RetrievalQuery


def test_query_rejects_empty_text() -> None:
    """Reject an empty text query."""
    with pytest.raises(ContractError):
        RetrievalQuery(query_id="q1", text=" ")


def test_query_requires_access_scope() -> None:
    """Reject a query without an authorized scope."""
    with pytest.raises(ContractError):
        RetrievalQuery(query_id="q1", text="rollback", allowed_scopes=())
