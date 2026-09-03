# ADR-T2-0006: Cho phép private Kaggle batch do đội kiểm soát

- **Trạng thái**: `ACCEPTED`.
- **Ngày hiệu lực**: 03-09-2026.
- **Authority**: Operator xác nhận đã đối chiếu lại với BTC và cho phép riêng cách chạy này.
- **Thay thế**: Phần cấm Kaggle trong ADR-T2-0005; lệnh cấm inference API vẫn giữ nguyên.

## Quyết định

Task 2 được train và batch inference trong private Kaggle kernel do tài khoản đội kiểm soát.
Kaggle API chỉ làm control plane để tạo/version private dataset, push kernel, đọc trạng thái và
tải output; nó không phải model inference API.

Phạm vi upload tối thiểu cho E0:

- `qa/train.jsonl`, `qa/validation.jsonl`, `qa/public.jsonl` từ release Task 2 đã khóa;
- release manifest, training/decoding/model/tokenizer controls và runtime source có SHA-256;
- không upload corpus 3,2 GB vì E0 không sử dụng retrieval;
- không upload Task 1, external data, secret hoặc Private labels.

Hugging Face chỉ được dùng trong kernel để tải đúng public model weights theo model ID và
revision allowlist. Hugging Face Inference API và mọi endpoint sinh answer vẫn bị cấm.

## Gate bắt buộc

1. Dataset và kernel đều private.
2. Local preflight xác minh release/task/split/hash trước upload.
3. Remote worker kiểm lại payload, QA và model-file hashes trước training.
4. QLoRA training phải lưu adapter hash; inference phải nạp adapter đó.
5. Validation chạy trước Public trong cùng immutable batch.
6. Output lưu run/model/config/code/decode/input hashes và per-sample trace.
7. Submission chỉ được import sau local replay validator PASS.
8. Không dùng Public score để tune dày; mọi improvement vẫn cần validation/paired evidence.

## Rollback

Nếu authority BTC thay đổi, dataset/kernel mất private status, hash mismatch hoặc trace thiếu,
`SourceAPI` quay lại fail-closed và artifact bị gắn `METHOD_PROVENANCE_FAIL`. ADR này không
cho phép API inference hoặc mở E1/E2 retrieval.
