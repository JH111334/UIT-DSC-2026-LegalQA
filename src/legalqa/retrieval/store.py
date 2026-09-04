from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from ..core.io import read_jsonl
from ..core.normalize import normalize_text
from ..core.schema import LegalChunk, ParentSection

_TOKEN_RE = re.compile(r"\d{1,5}/\d{4}/[A-ZĐ0-9-]+|[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(normalize_text(text, preserve_newlines=False))]


class CorpusStore:
    def __init__(self, path: str | Path, *, read_only: bool = True) -> None:
        self.path = Path(path)
        if read_only:
            uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True)
        else:
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    @classmethod
    def build(
        cls,
        path: str | Path,
        chunks_path: str | Path,
        parents_path: str | Path,
    ) -> dict[str, int | str]:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript(
            """
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                id TEXT NOT NULL UNIQUE,
                document_id TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                retrieval_text TEXT NOT NULL,
                level TEXT NOT NULL,
                heading TEXT,
                document_type TEXT,
                document_number TEXT,
                chapter TEXT,
                section_name TEXT,
                article TEXT,
                clause TEXT,
                point TEXT,
                path_json TEXT,
                source_name TEXT,
                references_json TEXT,
                amendments_json TEXT,
                ordinal INTEGER
            );
            CREATE INDEX chunks_parent_idx ON chunks(parent_id);
            CREATE INDEX chunks_document_idx ON chunks(document_id);
            CREATE INDEX chunks_citation_idx ON chunks(document_number, article, clause, point);
            CREATE TABLE parents (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                level TEXT NOT NULL,
                heading TEXT,
                document_type TEXT,
                document_number TEXT,
                chapter TEXT,
                section_name TEXT,
                article TEXT,
                path_json TEXT,
                source_name TEXT,
                references_json TEXT,
                amendments_json TEXT
            );
            CREATE INDEX parents_document_idx ON parents(document_id);
            CREATE INDEX parents_citation_idx ON parents(document_number, article);
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                retrieval_text,
                content='chunks',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 0'
            );
            """
        )
        chunk_count = 0
        batch = []
        insert_chunk = """INSERT INTO chunks VALUES (
            NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )"""
        for value in read_jsonl(chunks_path):
            batch.append(
                (
                    value["id"], value["document_id"], value["parent_id"], value["retrieval_text"],
                    value["level"], value.get("heading"), value.get("document_type"),
                    value.get("document_number"), value.get("chapter"), value.get("section"),
                    value.get("article"), value.get("clause"), value.get("point"),
                    json.dumps(value.get("path", []), ensure_ascii=False), value.get("source_name"),
                    json.dumps(value.get("references", []), ensure_ascii=False),
                    json.dumps(value.get("amendments", []), ensure_ascii=False), value.get("ordinal", 0),
                )
            )
            if len(batch) >= 2000:
                connection.executemany(insert_chunk, batch)
                chunk_count += len(batch)
                batch.clear()
        if batch:
            connection.executemany(insert_chunk, batch)
            chunk_count += len(batch)

        parent_count = 0
        batch = []
        insert_parent = "INSERT OR IGNORE INTO parents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        for value in read_jsonl(parents_path):
            batch.append(
                (
                    value["id"], value["document_id"], value["text"], value["level"],
                    value.get("heading"), value.get("document_type"), value.get("document_number"),
                    value.get("chapter"), value.get("section"), value.get("article"),
                    json.dumps(value.get("path", []), ensure_ascii=False), value.get("source_name"),
                    json.dumps(value.get("references", []), ensure_ascii=False),
                    json.dumps(value.get("amendments", []), ensure_ascii=False),
                )
            )
            if len(batch) >= 2000:
                cursor = connection.executemany(insert_parent, batch)
                parent_count += cursor.rowcount
                batch.clear()
        if batch:
            cursor = connection.executemany(insert_parent, batch)
            parent_count += cursor.rowcount
        connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
        connection.commit()
        connection.close()
        temporary.replace(output)
        return {"backend": "sqlite_fts5_bm25", "chunks": chunk_count, "parents": parent_count}

    @classmethod
    def build_release(
        cls,
        path: str | Path,
        chunks_path: str | Path,
    ) -> dict[str, int | str]:
        """Build the retrieval store directly from data-release chunk records."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript(
            """
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                id TEXT NOT NULL UNIQUE,
                document_id TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                retrieval_text TEXT NOT NULL,
                level TEXT NOT NULL,
                heading TEXT,
                document_type TEXT,
                document_number TEXT,
                chapter TEXT,
                section_name TEXT,
                article TEXT,
                clause TEXT,
                point TEXT,
                path_json TEXT,
                source_name TEXT,
                references_json TEXT,
                amendments_json TEXT,
                ordinal INTEGER
            );
            CREATE INDEX chunks_parent_idx ON chunks(parent_id);
            CREATE INDEX chunks_document_idx ON chunks(document_id);
            CREATE INDEX chunks_citation_idx ON chunks(document_number, article, clause, point);
            CREATE TABLE parents (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                level TEXT NOT NULL,
                heading TEXT,
                document_type TEXT,
                document_number TEXT,
                chapter TEXT,
                section_name TEXT,
                article TEXT,
                path_json TEXT,
                source_name TEXT,
                references_json TEXT,
                amendments_json TEXT
            );
            CREATE INDEX parents_document_idx ON parents(document_id);
            CREATE INDEX parents_citation_idx ON parents(document_number, article);
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                retrieval_text,
                content='chunks',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 0'
            );
            """
        )
        insert_chunk = """INSERT INTO chunks VALUES (
            NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )"""
        insert_parent = "INSERT OR IGNORE INTO parents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        chunk_count = 0
        parent_count = 0
        chunk_batch = []
        parent_batch = []

        def flush() -> None:
            nonlocal chunk_count, parent_count
            if not chunk_batch:
                return
            connection.executemany(insert_chunk, chunk_batch)
            chunk_count += len(chunk_batch)
            cursor = connection.executemany(insert_parent, parent_batch)
            parent_count += cursor.rowcount
            chunk_batch.clear()
            parent_batch.clear()

        for value in read_jsonl(chunks_path):
            retrieval_text = str(value["retrieval_text"])
            hierarchy = retrieval_text.split("\n", 1)[0]
            source_name = hierarchy.split(" > ", 1)[0]
            heading = hierarchy.rsplit(" > ", 1)[-1]
            document_id = str(value["doc_id"])
            parent_id = str(value["parent_chunk_id"])
            document_type = value.get("doc_type")
            document_number = value.get("doc_number_canonical")
            chapter = value.get("chapter")
            article = value.get("article")
            chunk_batch.append(
                (
                    value["chunk_id"],
                    document_id,
                    parent_id,
                    retrieval_text,
                    value["level"],
                    heading,
                    document_type,
                    document_number,
                    chapter,
                    None,
                    article,
                    value.get("clause"),
                    value.get("point"),
                    "[]",
                    source_name,
                    "[]",
                    "[]",
                    value.get("source_start", 0),
                )
            )
            parent_batch.append(
                (
                    parent_id,
                    document_id,
                    value["parent_text"],
                    "article" if article else "document",
                    heading,
                    document_type,
                    document_number,
                    chapter,
                    None,
                    article,
                    "[]",
                    source_name,
                    "[]",
                    "[]",
                )
            )
            if len(chunk_batch) >= 2000:
                flush()
        flush()
        connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
        connection.commit()
        connection.close()
        temporary.replace(output)
        return {
            "backend": "sqlite_fts5_bm25",
            "chunks": chunk_count,
            "parents": parent_count,
        }

    @staticmethod
    def _chunk(row: sqlite3.Row) -> LegalChunk:
        return LegalChunk(
            id=row["id"], document_id=row["document_id"], parent_id=row["parent_id"],
            retrieval_text=row["retrieval_text"], level=row["level"], heading=row["heading"] or "",
            document_type=row["document_type"], document_number=row["document_number"],
            chapter=row["chapter"], section=row["section_name"], article=row["article"],
            clause=row["clause"], point=row["point"], path=tuple(json.loads(row["path_json"] or "[]")),
            source_name=row["source_name"] or "", references=tuple(json.loads(row["references_json"] or "[]")),
            amendments=tuple(json.loads(row["amendments_json"] or "[]")), ordinal=row["ordinal"] or 0,
        )

    @staticmethod
    def _parent(row: sqlite3.Row) -> ParentSection:
        return ParentSection(
            id=row["id"], document_id=row["document_id"], text=row["text"], level=row["level"],
            heading=row["heading"] or "", document_type=row["document_type"],
            document_number=row["document_number"], chapter=row["chapter"], section=row["section_name"],
            article=row["article"], path=tuple(json.loads(row["path_json"] or "[]")),
            source_name=row["source_name"] or "", references=tuple(json.loads(row["references_json"] or "[]")),
            amendments=tuple(json.loads(row["amendments_json"] or "[]")),
        )

    @property
    def _is_unified_fts(self) -> bool:
        has_chunk_fts = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name IN ('chunk_fts', 'chunks_fts')"
        ).fetchone() is not None
        return not has_chunk_fts

    @property
    def _fts_table_name(self) -> str:
        row = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunk_fts', 'chunks_fts')"
        ).fetchone()
        return str(row[0]) if row else "chunks"

    def get_chunk(self, chunk_id: str) -> LegalChunk:
        if self._is_unified_fts:
            row = self.connection.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
            if row is None:
                raise KeyError(chunk_id)
            return LegalChunk(
                id=str(row["chunk_id"]),
                document_id=str(row["doc_id"]),
                parent_id=str(row["parent_chunk_id"]),
                retrieval_text=str(row["retrieval_text"]),
                level="clause" if row["article"] else "document",
                heading="",
                document_number=str(row["doc_number"]) if row["doc_number"] else None,
                article=str(row["article"]) if row["article"] else None,
            )
        row = self.connection.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if row is None:
            raise KeyError(chunk_id)
        return self._chunk(row)

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, LegalChunk]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return {}
        result: dict[str, LegalChunk] = {}
        if self._is_unified_fts:
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = self.connection.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", batch
                )
                for row in rows:
                    chunk = LegalChunk(
                        id=str(row["chunk_id"]),
                        document_id=str(row["doc_id"]),
                        parent_id=str(row["parent_chunk_id"]),
                        retrieval_text=str(row["retrieval_text"]),
                        level="clause" if row["article"] else "document",
                        heading="",
                        document_number=str(row["doc_number"]) if row["doc_number"] else None,
                        article=str(row["article"]) if row["article"] else None,
                    )
                    result[chunk.id] = chunk
            return result
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", batch)
            for row in rows:
                chunk = self._chunk(row)
                result[chunk.id] = chunk
        return result

    def get_parent(self, parent_id: str) -> ParentSection:
        if self._is_unified_fts:
            row = self.connection.execute(
                "SELECT * FROM chunks WHERE parent_chunk_id=? LIMIT 1", (parent_id,)
            ).fetchone()
            if row is None:
                raise KeyError(parent_id)
            return ParentSection(
                id=str(row["parent_chunk_id"]),
                document_id=str(row["doc_id"]),
                text=str(row["parent_text"]),
                level="article" if row["article"] else "document",
                heading="",
                document_number=str(row["doc_number"]) if row["doc_number"] else None,
                article=str(row["article"]) if row["article"] else None,
            )
        row = self.connection.execute("SELECT * FROM parents WHERE id=?", (parent_id,)).fetchone()
        if row is None:
            raise KeyError(parent_id)
        return self._parent(row)

    def has_parent(self, parent_id: str) -> bool:
        if self._is_unified_fts:
            return self.connection.execute(
                "SELECT 1 FROM chunks WHERE parent_chunk_id=? LIMIT 1", (parent_id,)
            ).fetchone() is not None
        return self.connection.execute("SELECT 1 FROM parents WHERE id=?", (parent_id,)).fetchone() is not None

    def search_bm25(self, query: str, top_k: int) -> list[tuple[str, float]]:
        terms = []
        for token in _tokenize(query):
            parts = [token] + [part.casefold() for part in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", token)]
            terms.extend(parts)
        terms = list(dict.fromkeys(term.replace('"', '""') for term in terms if term))
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        if self._is_unified_fts:
            rows = self.connection.execute(
                """SELECT chunk_id AS id, -bm25(chunks) AS score
                   FROM chunks
                   WHERE chunks MATCH ? ORDER BY bm25(chunks) ASC LIMIT ?""",
                (expression, top_k),
            )
        else:
            fts_table = self._fts_table_name
            rows = self.connection.execute(
                f"""SELECT chunks.id AS id, -bm25({fts_table}) AS score
                    FROM {fts_table} JOIN chunks ON chunks.rowid={fts_table}.rowid
                    WHERE {fts_table} MATCH ? ORDER BY bm25({fts_table}) LIMIT ?""",
                (expression, top_k),
            )
        return [(str(row["id"]), float(row["score"])) for row in rows]
