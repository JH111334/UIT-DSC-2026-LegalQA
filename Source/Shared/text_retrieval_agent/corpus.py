"""Load governed JSONL documents and create deterministic chunks."""

import json
import re
from pathlib import Path
from typing import Any

from text_retrieval_agent.contracts import ChunkRecord, ContractError, DocumentRecord

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _load_document(payload: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        document_id=str(payload["document_id"]),
        title=str(payload["title"]),
        body=str(payload["body"]),
        language=str(payload["language"]),
        source_uri=str(payload["source_uri"]),
        access_scope=str(payload["access_scope"]),
        corpus_version=str(payload["corpus_version"]),
        metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
        provenance={str(key): str(value) for key, value in payload.get("provenance", {}).items()},
    )


def load_documents(path: Path) -> tuple[DocumentRecord, ...]:
    """Load unique, contract-valid documents from a JSONL corpus."""
    documents: list[DocumentRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                document = _load_document(json.loads(line))
            except (KeyError, TypeError, json.JSONDecodeError, ContractError) as error:
                raise ContractError(f"Invalid corpus line {line_number}: {error}") from error
            if document.document_id in seen:
                raise ContractError(f"Duplicate document_id: {document.document_id}")
            seen.add(document.document_id)
            documents.append(document)
    if not documents:
        raise ContractError("Corpus must contain at least one document.")
    return tuple(documents)


def chunk_documents(
    documents: tuple[DocumentRecord, ...],
    *,
    max_sentences: int,
) -> tuple[ChunkRecord, ...]:
    """Split documents into stable, source-backed sentence groups."""
    if max_sentences < 1:
        raise ValueError("max_sentences must be positive.")
    chunks: list[ChunkRecord] = []
    for document in documents:
        sentences = tuple(
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY.split(document.body)
            if sentence.strip()
        )
        for position, start in enumerate(range(0, len(sentences), max_sentences)):
            text = " ".join(sentences[start : start + max_sentences])
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.document_id}::c{position:03d}",
                    document_id=document.document_id,
                    position=position,
                    text=text,
                    title=document.title,
                    language=document.language,
                    source_uri=document.source_uri,
                    access_scope=document.access_scope,
                    corpus_version=document.corpus_version,
                )
            )
    return tuple(chunks)
