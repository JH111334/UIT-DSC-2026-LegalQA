---
name: dsc-legal-ir-delivery
description: Triển khai Task 1 LegalIR với candidate bounded và bằng chứng retrieval tái lập.
---

# Giao nhận Task 1

1. Đọc AGENTS.md, Workflows/flowinfo.md và ADR Task 1.
2. Khóa corpus manifest, split, qrels, access scope và output schema BTC.
3. Chạy BM25 + TF-IDF/RRF trước neural adapter; từ chối model ngoài allowlist.
4. Giữ document/article ID, component score, rank và corpus version.
5. Retrieve rộng trước rerank bounded; so cùng query set và evaluation contract.
6. Báo metric, latency, memory, failure, paired case/slice delta và counterexample.
7. Macro Recall là metric chính thức; Macro Precision là tie-break; tối đa 5 ID.
8. Giữ model, index, organizer data, run và submission ngoài Git.

Train phải audit passage rỗng/trùng và chia split theo duplicate group.

Chạy gate retrieval-delivery gốc và warm-up validator khi file tồn tại. Log theo
Documents/General/RULE.md; tài liệu viết tiếng Việt có dấu, gọn.
