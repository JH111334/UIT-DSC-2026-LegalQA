"""Vietnamese legal-document metadata extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import normalize_text

DOCUMENT_TYPES = (
    "Bộ luật",
    "Luật",
    "Nghị định",
    "Thông tư liên tịch",
    "Thông tư",
    "Nghị quyết",
    "Quyết định",
    "Chỉ thị",
    "Công văn",
    "Pháp lệnh",
)

_TYPE_RE = re.compile(r"\b(" + "|".join(re.escape(x) for x in DOCUMENT_TYPES) + r")\b", re.I)
_DOCUMENT_NUMBER_RE = re.compile(
    r"\b(?:số\s*)?(?P<number>\d{1,5}/(?:\d{4}|[A-ZĐ]{1,8})(?:/[A-ZĐ0-9-]{2,30})?)\b",
    re.I,
)
_ARTICLE_CITATION_RE = re.compile(
    r"điều\s+(?P<article>\d+[a-zđ]?)"
    r"(?:\s+(?P<doc_type>bộ\s+luật|luật|nghị\s+định|thông\s+tư(?:\s+liên\s+tịch)?|"
    r"nghị\s+quyết|quyết\s+định|chỉ\s+thị|công\s+văn|pháp\s+lệnh))?"
    r"(?:\s+(?:số\s*)?(?P<doc_number>\d{1,5}/(?:\d{4}|[A-ZĐ]{1,8})(?:/[A-ZĐ0-9-]{2,30})?))?",
    re.I,
)
_CLAUSE_IN_PREFIX_RE = re.compile(r"khoản\s+(\d+[a-zđ]?)", re.I)
_POINT_IN_PREFIX_RE = re.compile(r"điểm\s+([a-zđ])", re.I)
_AMENDMENT_RE = re.compile(
    r"\b(sửa đổi|bổ sung|sửa đổi,\s*bổ sung|bãi bỏ|thay thế|hết hiệu lực)\b",
    re.I,
)


def canonical_document_number(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", "", value).upper()


@dataclass(frozen=True, slots=True)
class LegalCitation:
    article: str
    document_number: str | None = None
    document_type: str | None = None
    clause: str | None = None
    point: str | None = None
    raw: str = ""

    @property
    def key(self) -> str:
        return "|".join(
            (
                self.document_number or "",
                self.article.casefold(),
                (self.clause or "").casefold(),
                (self.point or "").casefold(),
            )
        )


def extract_document_type(text: str) -> str | None:
    match = _TYPE_RE.search(text)
    if not match:
        return None
    found = match.group(1).casefold()
    return next((value for value in DOCUMENT_TYPES if value.casefold() == found), match.group(1))


def extract_document_number(text: str) -> str | None:
    match = _DOCUMENT_NUMBER_RE.search(text)
    return canonical_document_number(match.group("number")) if match else None


def extract_citations(text: str) -> list[LegalCitation]:
    citations: list[LegalCitation] = []
    seen: set[str] = set()
    normalized = normalize_text(text, preserve_newlines=False)
    for match in _ARTICLE_CITATION_RE.finditer(normalized):
        prefix = normalized[max(0, match.start() - 70) : match.start()]
        boundary = max(prefix.rfind(";"), prefix.rfind("."), prefix.rfind(":"))
        prefix = prefix[boundary + 1 :]
        clauses = [value.casefold() for value in _CLAUSE_IN_PREFIX_RE.findall(prefix)] or [None]
        points = [value.casefold() for value in _POINT_IN_PREFIX_RE.findall(prefix)] or [None]
        # Multiple clauses before one article are independent labels. Multiple points
        # are paired only when unambiguous; parent/article matching remains valid.
        pairs = [(clause, points[-1] if len(points) == 1 else None) for clause in clauses]
        for clause, point in pairs:
            citation = LegalCitation(
                article=match.group("article").casefold(),
                document_number=canonical_document_number(match.group("doc_number")),
                document_type=(match.group("doc_type") or "").strip() or None,
                clause=clause,
                point=point,
                raw=f"{prefix} {match.group(0)}".strip(),
            )
            if citation.key not in seen:
                seen.add(citation.key)
                citations.append(citation)
    return citations


def extract_reference_keys(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(citation.key for citation in extract_citations(text)))


def extract_amendments(text: str) -> tuple[str, ...]:
    if not _AMENDMENT_RE.search(text):
        return ()
    return extract_reference_keys(text)


def extract_document_metadata(name: str, passage: str, link: str = "") -> dict[str, str | None]:
    head = f"{name}\n{passage[:5000]}\n{link}"
    return {
        "document_type": extract_document_type(head),
        "document_number": extract_document_number(head),
    }
