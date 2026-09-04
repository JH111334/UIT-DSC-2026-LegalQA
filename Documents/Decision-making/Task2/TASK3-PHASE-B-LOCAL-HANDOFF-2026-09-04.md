# TASK3 Phase B — Local technical handoff 2026-09-04

## Trạng thái

- Phase A release/index technical acceptance: `PASS`.
- Runtime đọc trực tiếp bundle `bm25.sqlite3/index_manifest.json`: `PASS`.
- C1 BM25 diagnostic evaluation trên release v3: `PASS`, `USER_REVIEW_REQUIRED`.
- Local schema-only prediction 5/5: `PASS`.
- E1 SFT preparation smoke 5/5: `PASS`.
- Python compile và test suite: `87 passed`.
- Full SFT cache 6.300 mẫu: `NOT_RUN_EXPENSIVE`.
- Step 1 E0 adapter probe: `BLOCKED_MISSING_E0_ADAPTER`.
- P4 Kaggle QLoRA: `BLOCKED_USER_GO_REQUIRED`.
- Public inference/submission/promotion: `NOT_AUTHORIZED`.

## Bằng chứng local

### Acceptance release/index

- Report: `Data/Task2/preflight/task2-data-v3/phase_b_index_acceptance.json`
- Release: `task2-data-v3`
- Index: `task2-data-v3-bm25-v1`
- Indexed chunks: `316100`
- SQLite integrity: `ok`
- Index artifact SHA-256: `292bfc540208c42165320994e52098d8b58c2692cb5668c64891aa07d12bf75d`
- Index manifest SHA-256: `4982f8549c09fa0affca082f13ec133f98867264b616792664d3889f6fca7227`

Acceptance này là kỹ thuật; không thay thế human freeze, review chất lượng hoặc quyết định promote.

### C1 BM25 diagnostic

- Report: `Data/Task2/runs/c1_recall_v3.json`
- HIGH-confidence judged queries: `173`
- HIGH-confidence Recall@5: `0.4101776335`
- HIGH-confidence MRR: `0.3846336538`
- All-confidence judged queries: `343`
- All-confidence Recall@5: `0.4174980405`
- All-confidence MRR: `0.4217427785`

Qrels có trạng thái `diagnostic_proxy`; manual citation audit còn pending nên metric này không có
quyền promote. Kết quả cũng cho thấy BM25 đơn thuần còn là bottleneck chất lượng.

### Smoke runtime

- Prediction: `Data/Task2/runs/task3_local_smoke_v4/smoke.json`
- Trace: `Data/Task2/runs/task3_local_smoke_v4/smoke.trace.json`
- Run manifest: `Data/Task2/runs/task3_local_smoke_v4/smoke.json.manifest.json`
- Records: `5/5`, complete.
- Execution mode: `schema_only`.
- Quality status: `NOT_APPLICABLE_SCHEMA_ONLY`.

### Smoke SFT preparation

- Cache: `Data/Task2/runs/task3_sft_smoke_v1/train_sft.jsonl`
- Report: `Data/Task2/runs/task3_sft_smoke_v1/train_sft.jsonl.report.json`
- Records: `5/5`.
- Đây chỉ là smoke wiring, không được dùng làm input Step 2 hoặc tạo expected hash production.

## Thay đổi kỹ thuật chính

- `legalqa` hỗ trợ trực tiếp schema SQLite FTS5 candidate do Phase A phát hành.
- Candidate search cache row BM25 và dùng chung store với evidence assembler, tránh full scan khi
  lấy chunk/parent trên virtual table 2 GB.
- Thêm `accept-e1-index` để xác minh release ID, manifest binding, SHA-256, row count và integrity.
- `expected-hashes` fail-closed trên release/index binding và ghi cả release/index artifact hash.
- Hai notebook Kaggle mount đúng đường dẫn `Data/Task2/indexes/...`, kiểm cả manifest lẫn artifact
  trước khi tải model/chạy GPU, và dừng sau validation.
- Bổ sung fixture Shared bị thiếu; toàn bộ test suite hiện chạy được từ clean checkout có fixture.

## Điều kiện để tiếp tục P4

1. Cung cấp E0 adapter thật tại `models/e0-full-v1` để tạo `expected_hashes_step1.json` và chạy Step 1.
2. Nếu vẫn quyết định thử E1 sau khi review Recall@5 thấp, tạo full cache 6.300 mẫu bằng
   `prepare-e1-sft`; không dùng cache smoke.
3. User ký GO rõ ràng trước khi chạy notebook Step 2 trên Kaggle GPU.
4. Review METEOR/ROUGE-L và run manifest trên validation trước mọi Public inference/submission.
5. Hoàn tất manual citation audit nếu muốn dùng qrels cho bất kỳ quyết định promote nào.

