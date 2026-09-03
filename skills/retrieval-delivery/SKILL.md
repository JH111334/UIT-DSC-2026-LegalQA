---
name: retrieval-delivery
description: Triển khai thay đổi retrieval có phạm vi, evidence, replay và failure gate trong DSC-TinDipLaPo.
---

# Giao nhận retrieval

## Gate đầu vào

1. Đọc AGENTS.md, Workflows/flowinfo.md và Documents/General/RULE.md.

2. Kiểm tra Git; bảo toàn thay đổi ngoài phạm vi.

3. Khóa input/output, corpus/query, access scope, evidence, fallback và stop gate.

4. Chạy deterministic retrieval/fixture trước LLM, external index hoặc hạ tầng.

5. Chỉ dùng model thuộc allowlist và revision đã duyệt.

## Học từ run

Trước comparison, khóa anchor/parent, hypothesis, evaluation role, gold status,
evaluation contract, slice registry và case budget. Báo paired delta,
wins|ties|losses, counterexample; tách observation khỏi cause và kết luận
supported|refuted|inconclusive. Run hợp lệ nhưng kém hơn là Checkpoint
REJECT|REPEAT; Error chỉ dành cho invalid run hoặc quality gate FAIL.

## Gate hoàn thành

    uv lock --check
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy
    uv run pytest --cov=text_retrieval_agent --cov-report=term-missing
    uv run python skills/retrieval-delivery/scripts/validate_project.py

Citation requirement áp dụng Task 1/shared retrieval fixture. Task 2 tuân theo
`dsc-legal-qa-delivery`: giữ E0 direct-generation làm control; chỉ mở E1 trên
`selected-contexts` có provenance Task 2 sau ADR/data gate. Không dùng lại corpus,
index, runtime, label hoặc checkpoint Task 1. Citation-derived qrels là proxy chẩn
đoán, không phải gold. Tài liệu viết tiếng Việt có dấu, viết hoa đầu câu, gọn và
không chép debug narrative.
