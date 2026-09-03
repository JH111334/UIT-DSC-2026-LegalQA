from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TypedDict

from ..core.config import ChunkingConfig
from ..core.metadata import extract_amendments, extract_reference_keys
from ..core.normalize import canonical_text, normalize_text, text_fingerprint
from ..core.schema import LegalChunk, LegalDocument, ParentSection

_ARTICLE_RE = re.compile(
    r"(?im)^[ \t]*(?P<header>điều\s+(?P<number>\d+[a-zđ]?)(?:\s*[.:/-])?[^\n]*)$"
)
_CHAPTER_RE = re.compile(r"(?im)^[ \t]*(chương\s+[^\n]+)$")
_SECTION_RE = re.compile(r"(?im)^[ \t]*(mục\s+[^\n]+)$")
_CLAUSE_RE = re.compile(r"(?im)^[ \t]*(?P<number>\d+)[.)][ \t]+")
_POINT_RE = re.compile(r"(?im)^[ \t]*(?P<number>[a-zđ])[.)][ \t]+")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


class _CommonFields(TypedDict):
    document_type: str | None
    document_number: str | None
    source_name: str


@dataclass(slots=True)
class ChunkingResult:
    chunks: list[LegalChunk]
    parents: list[ParentSection]
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0


def _stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


def _last_heading(pattern: re.Pattern[str], text: str, position: int) -> str | None:
    result = None
    for match in pattern.finditer(text, 0, position):
        result = normalize_text(match.group(1), preserve_newlines=False)
    return result


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    value = normalize_text(text)
    if not value:
        return []
    if len(value) <= max_chars:
        return [value]
    sentences = [part.strip() for part in _SENTENCE_BREAK_RE.split(value) if part.strip()]
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                windows.append(current)
                current = ""
            start = 0
            while start < len(sentence):
                end = min(len(sentence), start + max_chars)
                windows.append(sentence[start:end].strip())
                if end >= len(sentence):
                    break
                start = max(start + 1, end - overlap_chars)
            continue
        proposed = sentence if not current else f"{current} {sentence}"
        if len(proposed) <= max_chars:
            current = proposed
        else:
            windows.append(current)
            tail = current[-overlap_chars:].lstrip() if overlap_chars else ""
            current = f"{tail} {sentence}".strip()
    if current:
        windows.append(current)
    return list(dict.fromkeys(window for window in windows if window))


def _subsegments(text: str) -> list[tuple[str, str | None, str | None, str]]:
    clauses = list(_CLAUSE_RE.finditer(text))
    if not clauses:
        return [("article", None, None, text)]
    units: list[tuple[str, str | None, str | None, str]] = []
    intro = text[: clauses[0].start()].strip()
    if intro:
        units.append(("article_intro", None, None, intro))
    for index, clause_match in enumerate(clauses):
        end = clauses[index + 1].start() if index + 1 < len(clauses) else len(text)
        clause_text = text[clause_match.start() : end].strip()
        clause = clause_match.group("number")
        points = list(_POINT_RE.finditer(clause_text))
        if not points:
            units.append(("clause", clause, None, clause_text))
            continue
        intro = clause_text[: points[0].start()].strip()
        if intro:
            units.append(("clause", clause, None, intro))
        for point_index, point_match in enumerate(points):
            point_end = (
                points[point_index + 1].start()
                if point_index + 1 < len(points)
                else len(clause_text)
            )
            units.append(
                (
                    "point",
                    clause,
                    point_match.group("number").casefold(),
                    clause_text[point_match.start() : point_end].strip(),
                )
            )
    return units


def _pack_article_units(
    prefix: str,
    units: list[tuple[str, str | None, str | None, str]],
    config: ChunkingConfig,
) -> list[tuple[str, str | None, str | None, str]]:
    """Pack adjacent legal units into article windows without cutting boundaries."""

    packed: list[tuple[str, str | None, str | None, str]] = []
    current: list[tuple[str, str | None, str | None, str]] = []

    def emit() -> None:
        if not current:
            return
        levels = {value[0] for value in current}
        clauses = {value[1] for value in current}
        points = {value[2] for value in current}
        level = next(iter(levels)) if len(current) == 1 else "article_group"
        clause = next(iter(clauses)) if len(clauses) == 1 else None
        point = next(iter(points)) if len(points) == 1 else None
        payload = "\n".join(value[3] for value in current)
        packed.append((level, clause, point, f"{prefix}\n{payload}".strip()))
        current.clear()

    for unit in units:
        unit_text = normalize_text(unit[3])
        if not unit_text:
            continue
        available = max(200, config.max_chars - len(prefix) - 1)
        if len(unit_text) > available:
            emit()
            for window in split_long_text(unit_text, available, config.overlap_chars):
                packed.append((unit[0], unit[1], unit[2], f"{prefix}\n{window}".strip()))
            continue
        proposed = "\n".join([value[3] for value in current] + [unit_text])
        if current and len(prefix) + 1 + len(proposed) > config.max_chars:
            emit()
        current.append((unit[0], unit[1], unit[2], unit_text))
    emit()
    return packed


class LegalAwareChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk_document(self, document: LegalDocument) -> ChunkingResult:
        # Child chunks use the aggressively cleaned retrieval view, while
        # parent sections retain the conservative generation view.  Matching
        # repeated article numbers by occurrence keeps the two representations
        # aligned without assuming article numbers are globally unique.
        text = normalize_text(document.retrieval_text or document.passage)
        generation_text = normalize_text(document.generation_text or document.passage)
        article_matches = list(_ARTICLE_RE.finditer(text))
        if not article_matches:
            return self._fixed(document)
        generation_articles: dict[tuple[str, int], str] = {}
        generation_counts: dict[str, int] = {}
        generation_matches = list(_ARTICLE_RE.finditer(generation_text))
        for index, generation_match in enumerate(generation_matches):
            end = (
                generation_matches[index + 1].start()
                if index + 1 < len(generation_matches)
                else len(generation_text)
            )
            article = generation_match.group("number").casefold()
            occurrence = generation_counts.get(article, 0)
            generation_counts[article] = occurrence + 1
            generation_articles[(article, occurrence)] = generation_text[
                generation_match.start() : end
            ].strip()
        chunks: list[LegalChunk] = []
        parents: list[ParentSection] = []
        preamble = text[: article_matches[0].start()].strip()
        generation_preamble = (
            generation_text[: generation_matches[0].start()].strip()
            if generation_matches
            else preamble
        )
        if preamble:
            self._append_preamble(document, preamble, generation_preamble, chunks, parents)

        retrieval_counts: dict[str, int] = {}
        for index, match in enumerate(article_matches):
            end = (
                article_matches[index + 1].start()
                if index + 1 < len(article_matches)
                else len(text)
            )
            article_text = text[match.start() : end].strip()
            article = match.group("number").casefold()
            occurrence = retrieval_counts.get(article, 0)
            retrieval_counts[article] = occurrence + 1
            parent_text = generation_articles.get((article, occurrence), article_text)
            heading = normalize_text(match.group("header"), preserve_newlines=False)
            chapter = _last_heading(_CHAPTER_RE, text, match.start())
            section = _last_heading(_SECTION_RE, text, match.start())
            path = tuple(value for value in (document.name, chapter, section, heading) if value)
            # Content-based IDs intentionally collapse exact repeated copies of
            # the same article while preserving materially different versions.
            parent_id = _stable_id(document.id, "article", article, parent_text)
            references = extract_reference_keys(parent_text)
            amendments = extract_amendments(parent_text)
            common = self._common(document)
            parents.append(
                ParentSection(
                    id=parent_id,
                    document_id=document.id,
                    text=parent_text,
                    level="article",
                    heading=heading,
                    chapter=chapter,
                    section=section,
                    article=article,
                    path=path,
                    references=references,
                    amendments=amendments,
                    **common,
                )
            )
            body = article_text[match.end() - match.start() :].strip()
            prefix = " > ".join(path)
            packed_units = _pack_article_units(prefix, _subsegments(body), self.config)
            for window_index, (level, clause, point, window) in enumerate(packed_units):
                if len(canonical_text(window)) < self.config.min_chars:
                    continue
                chunks.append(
                    LegalChunk(
                        id=_stable_id(parent_id, level, str(window_index), window),
                        document_id=document.id,
                        parent_id=parent_id,
                        retrieval_text=window,
                        level=level,
                        heading=heading,
                        chapter=chapter,
                        section=section,
                        article=article,
                        clause=clause,
                        point=point,
                        path=path,
                        references=extract_reference_keys(window),
                        amendments=extract_amendments(window),
                        ordinal=len(chunks),
                        **common,
                    )
                )
        cleaned, exact_removed, near_removed = _deduplicate_chunks(
            chunks, near_duplicate="paywall_artifact" in document.flags
        )
        return ChunkingResult(cleaned, parents, exact_removed, near_removed)

    @staticmethod
    def _common(document: LegalDocument) -> _CommonFields:
        return {
            "document_type": document.metadata.get("document_type"),
            "document_number": document.metadata.get("document_number"),
            "source_name": document.name,
        }

    def _append_preamble(
        self,
        document: LegalDocument,
        retrieval_preamble: str,
        generation_preamble: str,
        chunks: list[LegalChunk],
        parents: list[ParentSection],
    ) -> None:
        parent_id = _stable_id(document.id, "preamble")
        parent_text = f"{document.name}\n{generation_preamble}".strip()
        retrieval_parent_text = f"{document.name}\n{retrieval_preamble}".strip()
        common = self._common(document)
        parents.append(
            ParentSection(
                id=parent_id,
                document_id=document.id,
                text=parent_text,
                level="document",
                heading=document.name,
                path=(document.name,),
                references=extract_reference_keys(parent_text),
                amendments=extract_amendments(parent_text),
                **common,
            )
        )
        for index, window in enumerate(
            split_long_text(
                retrieval_parent_text[: self.config.preamble_chars],
                self.config.max_chars,
                self.config.overlap_chars,
            )
        ):
            if len(canonical_text(window)) < self.config.min_chars:
                continue
            chunks.append(
                LegalChunk(
                    id=_stable_id(parent_id, str(index), window),
                    document_id=document.id,
                    parent_id=parent_id,
                    retrieval_text=window,
                    level="preamble",
                    heading=document.name,
                    path=(document.name,),
                    references=extract_reference_keys(window),
                    amendments=extract_amendments(window),
                    ordinal=len(chunks),
                    **common,
                )
            )

    def _fixed(self, document: LegalDocument) -> ChunkingResult:
        parent_id = _stable_id(document.id, "document")
        common = self._common(document)
        generation_text = normalize_text(document.generation_text or document.passage)
        retrieval_text = normalize_text(document.retrieval_text or document.passage)
        parent = ParentSection(
            id=parent_id,
            document_id=document.id,
            text=generation_text,
            level="document",
            heading=document.name,
            path=(document.name,),
            references=extract_reference_keys(generation_text),
            amendments=extract_amendments(generation_text),
            **common,
        )
        chunks: list[LegalChunk] = []
        for index, window in enumerate(
            split_long_text(
                f"{document.name}\n{retrieval_text}",
                self.config.max_chars,
                self.config.overlap_chars,
            )
        ):
            if len(canonical_text(window)) < self.config.min_chars:
                continue
            chunks.append(
                LegalChunk(
                    id=_stable_id(parent_id, str(index), window),
                    document_id=document.id,
                    parent_id=parent_id,
                    retrieval_text=window,
                    level="fixed",
                    heading=document.name,
                    path=(document.name,),
                    references=extract_reference_keys(window),
                    amendments=extract_amendments(window),
                    ordinal=index,
                    **common,
                )
            )
        cleaned, exact_removed, near_removed = _deduplicate_chunks(
            chunks, near_duplicate="paywall_artifact" in document.flags
        )
        return ChunkingResult(cleaned, [parent], exact_removed, near_removed)


class FixedChunker(LegalAwareChunker):
    def chunk_document(self, document: LegalDocument) -> ChunkingResult:
        return self._fixed(document)


def _deduplicate_chunks(
    chunks: list[LegalChunk], *, near_duplicate: bool = False
) -> tuple[list[LegalChunk], int, int]:
    seen: set[str] = set()
    result: list[LegalChunk] = []
    exact_removed = 0
    near_removed = 0
    near_candidates: dict[tuple[str, int, str, str], list[str]] = {}
    for chunk in chunks:
        fingerprint = text_fingerprint(chunk.retrieval_text)
        if fingerprint in seen:
            exact_removed += 1
            continue
        canonical = canonical_text(chunk.retrieval_text)
        if near_duplicate and len(canonical) >= 100:
            key = (
                chunk.article or "",
                len(canonical) // 50,
                canonical[:64],
                canonical[-64:],
            )
            candidates = near_candidates.setdefault(key, [])
            duplicate = any(
                abs(len(canonical) - len(other)) / max(len(canonical), len(other)) <= 0.02
                and SequenceMatcher(None, canonical, other, autojunk=False).ratio() >= 0.98
                for other in candidates
            )
            if duplicate:
                near_removed += 1
                continue
            candidates.append(canonical)
        seen.add(fingerprint)
        chunk.ordinal = len(result)
        result.append(chunk)
    return result, exact_removed, near_removed
