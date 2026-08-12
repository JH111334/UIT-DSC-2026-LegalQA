"""Generate independent, access-scoped sparse candidates."""

from collections.abc import Callable

from text_retrieval_agent.contracts import ChunkRecord, RetrievalQuery, RetrieverCandidate
from text_retrieval_agent.text import BM25Index, TfidfIndex


def _eligible(chunk: ChunkRecord, query: RetrievalQuery) -> bool:
    if chunk.access_scope not in query.allowed_scopes:
        return False
    if not query.filters:
        return True
    fields = {"language": chunk.language, "document_id": chunk.document_id}
    return all(fields.get(key) == value for key, value in query.filters.items())


def _search(
    chunks: tuple[ChunkRecord, ...],
    query: RetrievalQuery,
    limit: int,
    retriever: str,
    index_factory: Callable[[list[str]], BM25Index | TfidfIndex],
) -> tuple[RetrieverCandidate, ...]:
    eligible = [chunk for chunk in chunks if _eligible(chunk, query)]
    if not eligible:
        return ()
    index = index_factory([f"{chunk.title} {chunk.text}" for chunk in eligible])
    scored = [
        (chunk, score)
        for chunk, score in zip(eligible, index.scores(query.text), strict=True)
        if score > 0
    ]
    scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
    return tuple(
        RetrieverCandidate(
            chunk_id=chunk.chunk_id,
            retriever=retriever,  # type: ignore[arg-type]
            raw_score=score,
            rank=rank,
        )
        for rank, (chunk, score) in enumerate(scored[:limit], start=1)
    )


class BM25Retriever:
    """Retrieve authorized chunks with BM25."""

    def __init__(self, chunks: tuple[ChunkRecord, ...]) -> None:
        """Keep the immutable active chunk snapshot."""
        self._chunks = chunks

    def search(self, query: RetrievalQuery, limit: int) -> tuple[RetrieverCandidate, ...]:
        """Return positive BM25 candidates."""
        return _search(self._chunks, query, limit, "bm25", BM25Index)


class TfidfRetriever:
    """Retrieve authorized chunks with sparse TF-IDF cosine."""

    def __init__(self, chunks: tuple[ChunkRecord, ...]) -> None:
        """Keep the immutable active chunk snapshot."""
        self._chunks = chunks

    def search(self, query: RetrievalQuery, limit: int) -> tuple[RetrieverCandidate, ...]:
        """Return positive TF-IDF candidates."""
        return _search(self._chunks, query, limit, "tfidf", TfidfIndex)
