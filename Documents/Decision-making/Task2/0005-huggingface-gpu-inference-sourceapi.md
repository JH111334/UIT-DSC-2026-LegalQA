# ADR-T2-0005: Khóa remote API inference cho Task 2

- **Trạng thái**: `SUPERSEDED_BY_ADR_T2_0006`
- **Ngày rà soát**: 01-09-2026
- **Thay thế quyết định cũ**: chấp nhận Hugging Face/Kaggle inference trong `SourceAPI`

## Quyết định

Task 2 không dùng Hugging Face Inference API hoặc endpoint tương tự để sinh answer.
Quyết định cấm Kaggle tại thời điểm này đã được ADR-T2-0006 thay thế sau xác nhận mới
của operator với BTC. Hugging Face vẫn chỉ được dùng để tải weights đã nằm trong allowlist.

Tại thời điểm ADR này, `SourceAPI/pipeline.py` được giữ làm ranh giới fail-closed.
ADR-T2-0006 hiện đã thay implementation đó bằng private Kaggle batch có hash gate;
runtime model canonical vẫn được đóng gói từ `Source/Task2`.

## Bằng chứng dẫn tới thay đổi

Artifact `submissions/Task2/public_submission.zip` có format hợp lệ và đủ 1.000 ID,
nhưng runner Kaggle lịch sử:

- chỉ gọi `AutoModelForCausalLM.from_pretrained` với base Qwen;
- không nạp adapter `e0-full-v1`;
- không pin revision;
- không lưu trace nối output hash với input/model/adapter/config/code;
- nhúng Public organizer data trực tiếp vào source kernel.

Vì vậy artifact chỉ được gọi là **candidate format-valid**, không phải output đã chứng
minh của SFT E0. Audit canonical nằm tại
`Data/Task2/runs/public-external-artifact-audit-v1/provenance_audit.json`.

## Điều kiện để mở lại

Chỉ ADR mới bằng văn bản của nhóm, phù hợp quy định BTC, mới có thể mở remote runtime.
ADR đó phải khóa access right, môi trường đội kiểm soát, input hash, base revision,
adapter hash, code/config hash, per-sample trace, output hash và quy trình xóa dữ liệu.
