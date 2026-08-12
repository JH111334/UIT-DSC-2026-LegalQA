"""Test deterministic corpus ingestion and chunk identity."""

from pathlib import Path

from text_retrieval_agent.corpus import chunk_documents, load_documents

ROOT = Path(__file__).parents[2]


def test_chunks_are_stable_and_source_backed() -> None:
    """Create unique chunks that retain document source fields."""
    documents = load_documents(ROOT / "Data/Shared/fixtures/documents.jsonl")
    chunks = chunk_documents(documents, max_sentences=2)
    assert len(documents) == 6
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.source_uri and chunk.document_id in chunk.chunk_id for chunk in chunks)
