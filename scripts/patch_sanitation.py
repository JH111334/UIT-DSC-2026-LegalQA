from pathlib import Path

san_path = Path("src/legalqa/preprocessing/sanitation.py")
text = san_path.read_text(encoding="utf-8")
if "AD_TEXT" not in text:
    extra = """

AD_TEXT = (
    "\\n\\nBạn phải đăng nhập hoặc đăng ký thành viên TVPL Pro để sử dụng được đầy đủ "
    "các tiện ích gia tăng liên quan đến nội dung TCVN. Mọi chi tiết xin liên hệ: "
    "ĐT: (028) 3930 3279 DĐ: 0906 22 99 66."
)

_MEDIA_NOTE_RE = re.compile(r"(?i)\\((?:Hình từ Internet|Ảnh minh họa)\\)")
_TRAILING_MEDIA_BLOCK_RE = re.compile(
    r"(?is)\\n\\s*\\n[^\\n]{1,300}?\\s*\\((?:Hình từ Internet|Ảnh minh họa)\\)\\s*$"
)
_TRAILING_RELATED_QUESTION_RE = re.compile(r"(?:…\\s*){2,}\\.?\\s*.*$", re.DOTALL)
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \\t\\u00a0]+")


def sanitize_answer_raw(text: str) -> str:
    value = normalize_text(str(text))
    value = _TRAILING_RELATED_QUESTION_RE.sub("", value)
    value = _TRAILING_MEDIA_BLOCK_RE.sub("", value)
    value = _MEDIA_NOTE_RE.sub("", value)
    lines = [_HORIZONTAL_WHITESPACE_RE.sub(" ", line).rstrip() for line in value.splitlines()]
    return re.sub(r"\\n{3,}", "\\n\\n", "\\n".join(lines)).strip()


def sanitize_corpus_document(text: str) -> str:
    sanitizer = CorpusSanitizer(SanitationConfig())
    doc = LegalDocument(
        document_id="doc_temp",
        original_name="temp.txt",
        text=text,
        normalized_text=normalize_text(text),
    )
    result = sanitizer.sanitize_document(doc)
    return result.generation_text
"""
    san_path.write_text(text + extra, encoding="utf-8")
    print("Added AD_TEXT, sanitize_answer_raw, sanitize_corpus_document to legalqa sanitation.py")
