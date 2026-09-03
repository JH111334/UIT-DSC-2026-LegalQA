"""Text normalization primitives shared by all pipeline stages."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from html.parser import HTMLParser

_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\r\n]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n\s*\n+")
_HTML_HINT_RE = re.compile(r"</?(?:table|tr|td|th|p|div|br|li|ul|ol)\b", re.I)
_CITATION_SPACING_RE = re.compile(
    r"\b(\d{1,5})\s*/\s*(\d{4})\s*/\s*([A-ZĐ]+)\s*-\s*([A-ZĐ]+)\b",
    re.I,
)


class _LegalHTMLParser(HTMLParser):
    """Conservative HTML-to-text conversion that retains table cell boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"br", "p", "div", "tr", "ul", "ol"}:
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "div", "tr", "li", "td", "th"}:
            self.parts.append("\n" if tag.lower() not in {"td", "th"} else " | ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(text: str) -> str:
    if not _HTML_HINT_RE.search(text):
        return html.unescape(text)
    parser = _LegalHTMLParser()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts)


def normalize_legal_citations(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "/".join(
            (match.group(1), match.group(2), f"{match.group(3).upper()}-{match.group(4).upper()}")
        )

    return _CITATION_SPACING_RE.sub(replace, text)


def normalize_text(text: str | None, *, preserve_newlines: bool = True) -> str:
    """Normalize encoding/layout without deleting legally meaningful language."""

    if not text:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    value = html_to_text(value)
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u00a0", " ").replace("\u202f", " ")
    value = _ZERO_WIDTH_RE.sub("", value)
    value = value.replace("•", "- ").replace("●", "- ").replace("▪", "- ")
    value = _HORIZONTAL_SPACE_RE.sub(" ", value)
    value = "\n".join(line.strip() for line in value.splitlines())
    value = _BLANK_LINES_RE.sub("\n\n", value).strip()
    value = normalize_legal_citations(value)
    if not preserve_newlines:
        value = re.sub(r"\s+", " ", value).strip()
    return value


def canonical_text(text: str | None) -> str:
    return normalize_text(text, preserve_newlines=False).casefold()


def text_fingerprint(text: str | None) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()
