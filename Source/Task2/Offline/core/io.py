"""Input and output helpers shared by the pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from .metadata import extract_document_metadata
from .normalize import normalize_text, text_fingerprint
from .schema import LegalDocument, QAExample

_CONTEXT_ID_RE = re.compile(r"context_([^/\\]+)\.json$", re.I)
_TRAILING_SOURCE_ID_RE = re.compile(r"-\d+(?:\.aspx)?$", re.I)


def load_qa(path: str | Path) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    result: list[QAExample] = []
    for example_id, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Example {example_id!r} must be an object")
        question = normalize_text(value.get("question"), preserve_newlines=False)
        raw_answer = value.get("answer")
        answer = None if raw_answer is None else normalize_text(raw_answer)
        result.append(QAExample(str(example_id), question, answer))
    return result


def recover_name_from_link(link: str, document_id: str) -> str:
    path = urllib.parse.urlparse(link).path.rstrip("/")
    slug = urllib.parse.unquote(path.rsplit("/", 1)[-1])
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = _TRAILING_SOURCE_ID_RE.sub("", slug)
    slug = re.sub(r"[-_]+", " ", slug).strip()
    return normalize_text(slug, preserve_newlines=False) or f"Văn bản {document_id}"


def _document_id(member_name: str, value: dict[str, Any]) -> str:
    if value.get("id") is not None:
        return str(value["id"])
    match = _CONTEXT_ID_RE.search(member_name)
    return match.group(1) if match else text_fingerprint(member_name)[:16]


def iter_context_zip(path: str | Path) -> Iterator[LegalDocument]:
    """Compatibility alias for callers that previously required a ZIP."""

    yield from iter_contexts(path)


def iter_context_payloads(path: str | Path) -> Iterator[tuple[str, bytes]]:
    """Yield relative names and bytes from a context directory or ZIP."""

    source = Path(path)
    if source.is_dir():
        for item in sorted(source.rglob("*.json")):
            yield item.relative_to(source).as_posix(), item.read_bytes()
        return
    if not source.is_file():
        raise FileNotFoundError(f"Context source does not exist: {source}")

    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or member.suffix.lower() != ".json":
                continue
            yield info.filename, archive.read(info)


def context_source_sha256(path: str | Path) -> str:
    """Hash a ZIP directly or a directory by sorted path and file content."""

    source = Path(path)
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not source.is_dir():
        raise FileNotFoundError(f"Context source does not exist: {source}")
    for name, payload in iter_context_payloads(source):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def context_source_count(path: str | Path) -> int:
    """Count JSON members without materializing their content."""

    source = Path(path)
    if source.is_dir():
        return sum(1 for _ in source.rglob("*.json"))
    if not source.is_file():
        raise FileNotFoundError(f"Context source does not exist: {source}")
    with zipfile.ZipFile(source) as archive:
        return sum(
            not info.is_dir() and PurePosixPath(info.filename).suffix.lower() == ".json"
            for info in archive.infolist()
        )


def iter_contexts(path: str | Path) -> Iterator[LegalDocument]:
    """Read organizer context JSON without rewriting the raw passage."""

    for member_name, payload in iter_context_payloads(path):
        value = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(value, dict):
            continue
        document_id = _document_id(member_name, value)
        original_name = normalize_text(value.get("name"), preserve_newlines=False)
        link = str(value.get("link") or "").strip()
        name = original_name or recover_name_from_link(link, document_id)
        raw_text = str(value.get("passage") or "")
        passage = normalize_text(raw_text)
        flags = ("recovered_name",) if not original_name else ()
        yield LegalDocument(
            id=document_id,
            name=name,
            original_name=original_name,
            passage=passage,
            link=link,
            source_ids=(document_id,),
            flags=flags,
            metadata=extract_document_metadata(name, passage, link),
            raw_text=raw_text,
            retrieval_text=passage,
            generation_text=passage,
        )


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            value = record.to_dict() if hasattr(record, "to_dict") else record
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Replace a JSON file atomically so interrupted checkpoints stay readable."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
