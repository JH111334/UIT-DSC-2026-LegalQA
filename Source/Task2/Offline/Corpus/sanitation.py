from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..core.config import SanitationConfig
from ..core.normalize import canonical_text, normalize_text
from ..core.schema import LegalDocument

PAYWALL_RE = re.compile(
    r"bạn\s+phải\s+đăng\s+nhập\s+hoặc\s+đăng\s+ký\s+thành\s+viên\s+tvpl\s+pro\s+"
    r"để\s+sử\s+dụng\s+được\s+đầy\s+đủ\s+các\s+tiện\s+ích\s+gia\s+tăng\s+liên\s+quan\s+"
    r"đến\s+nội\s+dung\s+tcvn\s*\.?\s*mọi\s+chi\s+tiết\s+xin\s+liên\s+hệ\s*:\s*"
    r"đt\s*:\s*\(?028\)?\s*3930\s*3279\s*dđ\s*:\s*0906\s*22\s*99\s*66\s*\.?",
    re.IGNORECASE,
)
_MEDIA_NOTE_RE = re.compile(r"(?i)\((?:Hình từ Internet|Ảnh minh họa)\)")
_TRAILING_MEDIA_BLOCK_RE = re.compile(
    r"(?is)\n\s*\n[^\n]{1,300}?\s*\((?:Hình từ Internet|Ảnh minh họa)\)\s*$"
)
_TRAILING_RELATED_QUESTION_RE = re.compile(r"(?:…\s*){2,}\.?\s*.*$", re.DOTALL)
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_NOI_NHAN_RE = re.compile(r"(?im)^[ \t]*nơi\s+nhận\s*:")
_ROUTING_LINE_RE = re.compile(r"(?m)^[ \t]*[-\u2013\u2014+]\s*\S+")
_SIGNATORY_RE = re.compile(
    r"(?im)^[ \t]*(?:kt\.|tl\.|tuq\.)?\s*(?:bộ\s+trưởng|thứ\s+trưởng|chủ\s+tịch|phó\s+chủ\s+tịch)\b"
)
_TOKEN_RE = re.compile(r"\S+")
_WEB_CODE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\$\s*\(",
        r"\$\.(?:ajax|get|post)\s*\(",
        r"\bfunction\s+[A-Za-z_$][\w$]*\s*\(",
        r"\b(?:var|let|const)\s+[A-Za-z_$][\w$]*\s*=",
        r"\b(?:window|document)\.",
        r"\.(?:html|css|attr|append|show|hide|dialog|cookies)\s*\(",
        r"https?://(?:www\.)?thuvienphapluat\.vn/(?:page|banan|phap-luat|hoi-dap)",
        r"^\s*(?:if|else\s+if|for|while)\s*\(",
        r"^\s*(?:\}|\{|\}\s*else\s*\{|\);|\}\);)\s*$",
        r"^\s*[.#][A-Za-z_][\w .#>:+-]*\{\s*$",
        r"^\s*(?:position|margin(?:-[a-z]+)?|padding(?:-[a-z]+)?|z-index|color|font-size|font-weight|"
        r"background|border(?:-[a-z]+)?|display|overflow|text-align|line-height|opacity|animation)\s*:",
    )
)
_WEB_UI_LINES = {
    canonical_text(value)
    for value in (
        "MỤC LỤC VĂN BẢN",
        "In mục lục",
        "Bạn Chưa Đăng Nhập Thành Viên!",
        "Bạn Đang Đăng Nhập Thành Viên Free!",
        "Bạn Đang Đăng Nhập Thành Viên Basic!",
        "NỘI DUNG GỐC",
        "NỘI DUNG SỬA ĐỔI, HƯỚNG DẪN",
        "Văn bản bị thay thế",
        "Văn bản thay thế",
        "Tải Văn bản tiếng Việt",
        "Tải biểu mẫu",
        "Lưu trữ",
        "Ghi chú",
        "Ý kiến Facebook",
        "Bản án liên quan",
        "PHÁP LUẬT DOANH NGHIỆP",
        "Hỏi đáp pháp luật",
        "FILE ĐƯỢC ĐÍNH KÈM THEO VĂN BẢN",
        "FILE ATTACHED TO DOCUMENT",
    )
}
_WEB_UI_PREFIXES = tuple(
    canonical_text(value)
    for value in (
        "Vì chưa Đăng Nhập nên Bạn chỉ xem được",
        "Bạn chưa xem được Hiệu lực của Văn bản",
        "Nếu chưa là Thành Viên, mời Bạn Đăng ký Thành viên",
        "Vì Đăng Nhập Thành Viên Free nên Bạn chỉ xem được",
        "Vì Đăng Nhập Thành Viên Basic nên Bạn chỉ xem được",
        "Nếu muốn làm Thành Viên",
        "Rà chuột vào nội dụng văn bản để sử dụng",
        "Click trái để xem cụ thể",
        "Click phải để xem những nội dung",
        "Double click để xem tất cả",
        "Tắt so sánh [X]",
    )
)


@dataclass(slots=True)
class SanitationResult:
    retrieval_text: str
    generation_text: str
    metrics: dict[str, Any]
    flags: tuple[str, ...]


def sanitize_answer_raw(text: str) -> str:
    """Create the model answer while preserving the untouched answer_raw field.

    A media marker may occur before genuine legal content, so only a final,
    blank-line-delimited caption is removed as a block. Other occurrences lose
    the marker itself but retain surrounding legal text. Literal ASCII form
    placeholders such as ``.........`` are deliberately outside the ellipsis rule.
    """

    value = normalize_text(str(text))
    value = _TRAILING_RELATED_QUESTION_RE.sub("", value)
    value = _TRAILING_MEDIA_BLOCK_RE.sub("", value)
    value = _MEDIA_NOTE_RE.sub("", value)
    lines = [_HORIZONTAL_WHITESPACE_RE.sub(" ", line).rstrip() for line in value.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def answer_sanitation_residue(text: str) -> dict[str, bool]:
    """Return machine-readable A1 residue checks for a sanitized answer."""

    value = str(text)
    return {
        "media_note": bool(_MEDIA_NOTE_RE.search(value)),
        "related_question_ellipsis": bool(_TRAILING_RELATED_QUESTION_RE.search(value)),
    }


def _normalized_block(value: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(value, preserve_newlines=False)).casefold().strip()


def _informative_block(value: str, min_chars: int) -> bool:
    """Exclude punctuation-only form placeholders from corruption metrics."""

    if len(value) < min_chars:
        return False
    alphanumeric = sum(character.isalnum() for character in value)
    word_count = len(re.findall(r"\w+", value, flags=re.UNICODE))
    return alphanumeric / max(1, len(value)) >= 0.25 and word_count >= 8


def local_repetition_metrics(text: str, min_chars: int = 100) -> dict[str, Any]:
    blocks = []
    for block in re.split(r"\n\s*\n+", normalize_text(text)):
        normalized = _normalized_block(block)
        if _informative_block(normalized, min_chars):
            blocks.append(normalized)
    counts = Counter(blocks)
    total_chars = sum(len(block) for block in blocks)
    duplicate_chars = sum((count - 1) * len(block) for block, count in counts.items() if count > 1)
    return {
        "block_count": len(blocks),
        "unique_block_count": len(counts),
        "repetition_ratio": round(1.0 - len(counts) / len(blocks), 6) if blocks else 0.0,
        "max_block_frequency": max(counts.values(), default=0),
        "duplicate_char_ratio": round(duplicate_chars / max(1, total_chars), 6),
    }


def _token_overlap(
    left_tokens: list[str], right_tokens: list[str], config: SanitationConfig
) -> int:
    """Return longest exact suffix-prefix overlap using incremental tokens."""

    if not left_tokens or not right_tokens:
        return 0
    limit = min(config.overlap_max_tokens, len(left_tokens), len(right_tokens))

    def compute(current_limit: int) -> int:
        pattern = right_tokens[:current_limit]
        suffix = left_tokens[-current_limit:]
        sentinel = object()
        sequence: list[object] = [*pattern, sentinel, *suffix]
        prefix = [0] * len(sequence)
        for index in range(1, len(sequence)):
            candidate = prefix[index - 1]
            while candidate and sequence[index] != sequence[candidate]:
                candidate = prefix[candidate - 1]
            if sequence[index] == sequence[candidate]:
                candidate += 1
            prefix[index] = candidate
        return min(prefix[-1], current_limit)

    overlap = compute(limit)
    full_limit = min(len(left_tokens), len(right_tokens))
    if overlap == limit and limit < full_limit:
        overlap = compute(full_limit)
    return overlap if overlap >= config.overlap_min_tokens else 0


def _merge_paywall_fragments(
    parts: list[str], config: SanitationConfig
) -> tuple[str, dict[str, int]]:
    merged = ""
    merged_tokens: list[str] = []
    seen_fragments: set[str] = set()
    exact_fragments_removed = 0
    overlap_tokens_removed = 0
    overlap_chars_removed = 0
    for raw_part in parts:
        part = normalize_text(raw_part)
        if not part:
            continue
        fingerprint = hashlib.sha1(_normalized_block(part).encode("utf-8")).hexdigest()
        if len(part) >= config.block_min_chars and fingerprint in seen_fragments:
            exact_fragments_removed += 1
            continue
        seen_fragments.add(fingerprint)
        if not merged:
            merged = part
            merged_tokens = [match.group(0).casefold() for match in _TOKEN_RE.finditer(part)]
            continue
        right_matches = list(_TOKEN_RE.finditer(part))
        right_tokens = [match.group(0).casefold() for match in right_matches]
        overlap = _token_overlap(merged_tokens, right_tokens, config)
        if overlap:
            offset = right_matches[overlap - 1].end()
            overlap_tokens_removed += overlap
            overlap_chars_removed += offset
            novel = part[offset:].lstrip()
            if novel:
                merged = f"{merged}\n{novel}"
                merged_tokens.extend(right_tokens[overlap:])
        else:
            merged = f"{merged}\n{part}"
            merged_tokens.extend(right_tokens)
    return normalize_text(merged), {
        "exact_fragments_removed": exact_fragments_removed,
        "overlap_tokens_removed": overlap_tokens_removed,
        "overlap_chars_removed": overlap_chars_removed,
    }


def _deduplicate_exact_blocks(text: str, min_chars: int) -> tuple[str, int, int]:
    blocks = re.split(r"(\n\s*\n+)", normalize_text(text))
    seen: set[str] = set()
    output: list[str] = []
    removed_count = 0
    removed_chars = 0
    for block in blocks:
        if not block or re.fullmatch(r"\n\s*\n+", block):
            if output and block:
                output.append(block)
            continue
        normalized = _normalized_block(block)
        if len(normalized) >= min_chars:
            fingerprint = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            if fingerprint in seen:
                removed_count += 1
                removed_chars += len(block)
                continue
            seen.add(fingerprint)
        output.append(block)
    return normalize_text("".join(output)), removed_count, removed_chars


def _remove_web_artifact_lines(text: str) -> tuple[str, int, int]:
    """Remove reviewed TVPL page-code/navigation lines from corrupted pages."""

    output: list[str] = []
    removed_lines = 0
    removed_chars = 0
    for line in normalize_text(text).splitlines():
        canonical = canonical_text(line)
        web_ui = canonical in _WEB_UI_LINES or any(
            canonical.startswith(prefix) for prefix in _WEB_UI_PREFIXES
        )
        web_code = any(pattern.search(line) for pattern in _WEB_CODE_PATTERNS)
        if web_ui or web_code:
            removed_lines += 1
            removed_chars += len(line)
            continue
        output.append(line)
    return normalize_text("\n".join(output)), removed_lines, removed_chars


class CorpusSanitizer:
    def __init__(self, config: SanitationConfig) -> None:
        self.config = config
        blacklist = yaml.safe_load(Path(config.blacklist_path).read_text(encoding="utf-8")) or {}
        protected = (
            yaml.safe_load(Path(config.protected_patterns_path).read_text(encoding="utf-8")) or {}
        )
        self.blacklist_lines = {canonical_text(value) for value in blacklist.get("exact_lines", [])}
        self.blacklist_regex = [
            re.compile(value, re.IGNORECASE) for value in blacklist.get("regex_lines", [])
        ]
        self.review_candidates = tuple(
            str(value).casefold() for value in blacklist.get("review_candidates", [])
        )
        self.protected_regex = [re.compile(value, re.IGNORECASE) for value in protected.get("patterns", [])]

    def _protected(self, line: str) -> bool:
        normalized = canonical_text(line)
        return any(pattern.search(normalized) for pattern in self.protected_regex)

    def classify_line(self, line: str) -> str:
        """Classify a DF candidate; classification alone never deletes text."""

        normalized = canonical_text(line)
        if self._protected(line):
            return "protected"
        if normalized in self.blacklist_lines or any(
            pattern.fullmatch(normalized) for pattern in self.blacklist_regex
        ):
            return "blacklist"
        return "manual_review"

    def _clean_retrieval_lines(self, text: str) -> tuple[str, int]:
        removed = 0
        output = []
        for line in normalize_text(text).splitlines():
            normalized = canonical_text(line)
            blacklisted = normalized in self.blacklist_lines or any(
                pattern.fullmatch(normalized) for pattern in self.blacklist_regex
            )
            if blacklisted and not self._protected(line):
                removed += 1
                continue
            output.append(line)
        return normalize_text("\n".join(output)), removed

    def _remove_terminal_noi_nhan(self, text: str) -> tuple[str, bool, float | None]:
        matches = list(_NOI_NHAN_RE.finditer(text))
        if not matches:
            return text, False, None
        position = matches[-1].start()
        ratio = position / max(1, len(text))
        if ratio < self.config.terminal_noi_nhan_min_ratio:
            return text, False, round(ratio, 6)
        tail = text[position:]
        # Position alone is not sufficient: require a routing-list bullet or
        # a conventional signatory marker before deleting a terminal block.
        if not _ROUTING_LINE_RE.search(tail) and not _SIGNATORY_RE.search(tail):
            return text, False, round(ratio, 6)
        return text[:position].rstrip(), True, round(ratio, 6)

    def sanitize(self, document: LegalDocument) -> LegalDocument:
        base = normalize_text(document.raw_text or document.passage)
        paywall_matches = list(PAYWALL_RE.finditer(base))
        generation = base
        recovery = {
            "exact_fragments_removed": 0,
            "overlap_tokens_removed": 0,
            "overlap_chars_removed": 0,
        }
        exact_blocks_removed = 0
        exact_block_chars_removed = 0
        web_artifact_lines_removed = 0
        web_artifact_chars_removed = 0
        flags = list(document.flags)
        if paywall_matches:
            flags.extend(("paywall_artifact", "aggressive_corruption_recovery"))
            if self.config.recover_paywall_documents:
                recovery_base, web_artifact_lines_removed, web_artifact_chars_removed = (
                    _remove_web_artifact_lines(base)
                )
                if web_artifact_lines_removed:
                    flags.append("web_artifact_recovered")
                generation, recovery = _merge_paywall_fragments(
                    PAYWALL_RE.split(recovery_base), self.config
                )
                generation, exact_blocks_removed, exact_block_chars_removed = (
                    _deduplicate_exact_blocks(generation, self.config.block_min_chars)
                )
            else:
                generation = PAYWALL_RE.sub(" ", base)

        retrieval = generation
        removed_lines = 0
        noi_nhan_removed = False
        noi_nhan_ratio = None
        if self.config.remove_retrieval_boilerplate:
            retrieval, removed_lines = self._clean_retrieval_lines(retrieval)
            retrieval, noi_nhan_removed, noi_nhan_ratio = self._remove_terminal_noi_nhan(retrieval)

        review_matches = {
            pattern: base.casefold().count(pattern)
            for pattern in self.review_candidates
            if pattern in base.casefold()
        }
        repetition = local_repetition_metrics(base, self.config.block_min_chars)
        post_repetition = local_repetition_metrics(generation, self.config.block_min_chars)
        if (
            repetition["repetition_ratio"] >= self.config.local_repetition_flag_ratio
            and repetition["max_block_frequency"] >= self.config.local_max_tf_flag
        ):
            flags.append("local_repetition_review")
            if repetition["max_block_frequency"] >= 20:
                flags.append("local_repetition_critical")
        metrics: dict[str, Any] = {
            "sanitation_version": self.config.version,
            "raw_chars": len(document.raw_text or ""),
            "normalized_chars": len(base),
            "generation_chars": len(generation),
            "retrieval_chars": len(retrieval),
            "paywall_count": len(paywall_matches),
            "paywall_chars_removed": sum(match.end() - match.start() for match in paywall_matches),
            "retrieval_boilerplate_lines_removed": removed_lines,
            "noi_nhan_removed": noi_nhan_removed,
            "noi_nhan_position_ratio": noi_nhan_ratio,
            "exact_blocks_removed": exact_blocks_removed,
            "exact_block_chars_removed": exact_block_chars_removed,
            "web_artifact_lines_removed": web_artifact_lines_removed,
            "web_artifact_chars_removed": web_artifact_chars_removed,
            "review_candidate_matches": review_matches,
            **recovery,
            **repetition,
            **{f"post_{key}": value for key, value in post_repetition.items()},
        }
        document.passage = generation
        document.retrieval_text = retrieval
        document.generation_text = generation
        document.flags = tuple(dict.fromkeys(flags))
        document.metadata["sanitation"] = metrics
        return document
