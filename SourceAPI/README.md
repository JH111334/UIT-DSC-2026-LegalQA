# SourceAPI — private Kaggle batch cho Task 2

`SourceAPI` điều phối GPU từ PowerShell nhưng model vẫn chạy trong private Kaggle kernel do
đội kiểm soát. Không cần mở giao diện web. Kaggle API chỉ upload/version payload, push kernel,
monitor và tải output; không phải model inference API.

Authority: ADR-T2-0006, sau xác nhận của operator với BTC ngày 03-09-2026.

## Contract

```powershell
uv run python SourceAPI\pipeline.py contract
```

Chỉ hỗ trợ một operation compact:

```text
train-evaluate-predict
```

Một job thực hiện theo thứ tự:

```text
hash preflight local
  -> private Kaggle dataset tối thiểu
  -> pinned Qwen snapshot
  -> QLoRA answer-only SFT
  -> load LoRA adapter
  -> 700 validation + metric/slice/case/error reports
  -> 1.000 Public predictions
  -> submission ZIP
  -> download + local replay validation
```

## Credential

Đặt Kaggle credential tại `%USERPROFILE%\.kaggle\kaggle.json` hoặc dùng biến môi trường
Kaggle CLI. Không ghi token vào request, source, log hoặc Git.

Kiểm tra đăng nhập mà không lộ secret:

```powershell
kaggle datasets list -m
```

## Chạy E0 v2

```powershell
uv run python SourceAPI\pipeline.py run --request Data\Task2\control\requests\kaggle_e0_v2_r1.json
```

Request chỉ giữ path, slug, revision, package pin và timeout. Các knob training/decoding lấy
từ control artifacts canonical; không nhân bản trên CLI.

Output khi PASS:

```text
Data/Task2/runs/e0-kaggle-v2-r1/
├── checkpoint-or-adapter/
├── run_manifest.json
├── checkpoint_manifest.json
├── train_metrics.json
├── resource_log.json
├── validation_predictions.json
├── validation_trace.json
├── metrics.json
├── slice_metrics.json
├── case_metrics.jsonl
├── error_table.jsonl
├── public_predictions.json
├── public_trace.json
├── public_output_audit.json
├── public_submission_report.json
├── public_submission.zip
└── remote_execution_manifest.json

submissions/Task2/public_submission_e0-kaggle-v2-r1.zip
```

Submission chỉ được copy ra `submissions/` sau khi hash payload khớp và validator local xác
nhận đúng 1.000 ID, answer không rỗng, UTF-8 không BOM, ZIP chỉ chứa `submission.json`.

## Ranh giới vẫn giữ

- Không dùng Hugging Face Inference API.
- Không upload Task 1, corpus 3,2 GB, Private labels, external data hoặc augmentation.
- Dataset và kernel phải private.
- Không dùng điểm Public làm tuning loop.
- Một kernel hoàn tất không tự chứng minh chất lượng; quyết định vẫn dựa trên validation,
  slice/case report và paired evidence.
