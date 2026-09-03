# Checkpoint V2 — E0 training có thật, Public chỉ format-valid

## Đã chứng minh

- `task2-data-v1` và tokenizer audit sẵn sàng.
- QLoRA E0 hoàn thành 1 epoch, train loss `1.113865`; adapter tồn tại.
- Public ZIP SHA-256 `bf9ac32c...` pass exact schema/ID/UTF-8/ZIP replay.
- Output audit unlabeled: 1.000 records, 0 empty, 0 duplicate.
- Code evaluation đã tạo contract cho metric, slice, case, paired và error artifacts.
- Ghi nhận lịch sử 01-09: SourceAPI từng fail-closed. Từ 03-09, ADR-T2-0006 cho phép
  private Kaggle batch có hash/trace; HF inference API vẫn bị khóa.

## Chưa chứng minh

- Public ZIP được sinh bởi adapter E0.
- METEOR/ROUGE-L hoặc legal correctness.
- Validation score/slices/case regressions.
- E1 BM25 hoặc E2b hybrid tốt hơn E0.

## Quyết định

Giữ E0 checkpoint làm anchor huấn luyện, nhưng chưa có anchor quality. Bước kế tiếp là
validation canonical trên 700 câu; không mở retrieval hoặc model mới trước diagnosis.
