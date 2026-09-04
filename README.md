# UIT DSC 2026 LegalQA

## Cấu trúc mã nguồn

Mã nguồn được chia theo từng giai đoạn của pipeline:

```text
src/legalqa/
|-- core/           # Cấu hình, schema, I/O, metadata và chuẩn hóa dùng chung
|-- preprocessing/  # Audit, làm sạch, chunking và đóng gói data release
|-- retrieval/      # Index, ranking, evidence, citation graph và QA-neighbor
|-- generation/     # Generator và pipeline sinh câu trả lời
|-- evaluation/     # Diagnostic, metric, evaluation và kiểm tra submission
`-- cli.py          # Entrypoint điều phối các task

configs/preprocessing/  # Policy và pattern phục vụ làm sạch corpus
```

Các lệnh CLI hiện có được giữ nguyên; chỉ đường dẫn import Python nội bộ thay đổi theo các package trên.

Pipeline retrieval-first cho Task 2 LegalQA tiếng Việt. Workspace hiện có corpus đã vệ sinh và đóng băng ở phiên bản `corpus-v2.1`, BM25 + multilingual E5 + mMARCO reranker, các ablation retrieval/evidence, generator extractive/Qwen, và phương án QA-neighbor dùng dữ liệu train.

## Trạng thái hiện tại

- Corpus v2.1: tất cả kiểm tra trong `freeze_report.json` đạt.
- Retrieval diagnostic (100 mẫu): Recall@1 `0.59`, Recall@5 `0.81`, MRR `0.6826`.
- QA-neighbor holdout (1.000/7.000 mẫu train): METEOR exact-token `0.3041` với RRF và dense weight `4.0`.
- Public submission: 1.000/1.000 dự đoán, validator đạt toàn bộ kiểm tra.

Các metric local chỉ dùng để so sánh phương án. Điểm chính thức cần được xác nhận bằng scorer của ban tổ chức.

## Thiết lập và kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m legalqa --config configs\hybrid_e5_mmarco_rerank.yaml validate-config
```

Các model trong cấu hình production dùng `local_files_only: true`. Runtime tự bật chế độ Hugging Face offline để không phát sinh network retry.

## Bàn giao preprocessing Task 2

Release canonical nằm tại `artifacts/data_release_v1`. Nó chứa hai nhánh cùng `release_id`: QA train/validation/public và corpus từ `selected-contexts`, cùng diagnostic qrels trích từ citation trong answer của validation. Qrels chỉ là `diagnostic_proxy`, không phải nhãn gold hay dữ liệu train.

```powershell
.\.venv\Scripts\python.exe -m legalqa `
  --config configs\corpus_v2.yaml `
  build-release `
  --release-root artifacts\data_release_v1 `
  --validation-size 700 `
  --tokenizer-model Qwen/Qwen2.5-VL-3B-Instruct

.\.venv\Scripts\python.exe -m legalqa `
  --config configs\corpus_v2.yaml `
  preflight-release `
  --stage training `
  --profile e0-direct `
  --release-root artifacts\data_release_v1 `
  --report artifacts\data_release_v1\preflight_release_e0.json

.\.venv\Scripts\python.exe -m legalqa `
  --config configs\corpus_v2.yaml `
  preflight-release `
  --stage public `
  --profile e1-bm25 `
  --release-root artifacts\data_release_v1 `
  --report artifacts\data_release_v1\preflight_release_e1_public.json
```

Release hiện có 6.300 QA train, 700 validation, 1.000 public, 8.532 document và 316.100 chunk. Cả E0 training lẫn E1 Public đều qua preflight. Có 150 record manual citation audit đang `PENDING`; việc này không chặn smoke kỹ thuật nhưng phải hoàn tất trước khi dùng proxy qrels để quyết định promote.

## Tạo submission khuyến nghị

```powershell
.\.venv\Scripts\python.exe -m legalqa `
  --config configs\default.yaml `
  predict `
  --output artifacts\corpus_v2\submission_qwen.json `
  --trace artifacts\corpus_v2\submission_qwen.trace.json `
  --resume

.\.venv\Scripts\python.exe -m legalqa `
  --config configs\default.yaml `
  validate-predictions `
  --predictions artifacts\corpus_v2\submission_qwen.json
```

## Chạy pipeline corpus-RAG

Lệnh `predict` hỗ trợ checkpoint nguyên tử và tiếp tục sau gián đoạn. Mặc định kết quả được checkpoint sau mỗi câu.

```powershell
.\.venv\Scripts\python.exe -m legalqa `
  --config configs\qa_qwen25_vl_3b.yaml `
  predict `
  --output artifacts\corpus_v2\submission_qwen.json `
  --resume
```

Máy hiện tại chỉ có CPU; Qwen2.5-VL-3B cho chất lượng smoke tốt hơn extractive nhưng không phải lựa chọn khả thi để sinh nhanh toàn bộ 1.000 câu.

## Artifact quan trọng

- `artifacts/data_release_v1/manifest.json`: manifest canonical và SHA-256 của toàn bộ release preprocessing.
- `artifacts/data_release_v1/preflight_release_e0.json`: kết quả gate E0 training.
- `artifacts/data_release_v1/preflight_release_e1_public.json`: kết quả gate E1 Public.
- `artifacts/corpus_v2/freeze_report.json`: kiểm tra và fingerprint corpus.
- `artifacts/corpus_v2/retrieval_diagnostic_metrics_hybrid_mmarco_rerank_100.json`: kết quả retrieval tốt nhất.
- `artifacts/corpus_v2/qa_neighbor_rrf_grid_holdout_1000.json`: chọn trọng số RRF.
- `artifacts/corpus_v2/submission_public_neighbor_rrf.json`: submission public hoàn chỉnh.
- `artifacts/corpus_v2/submission_public_neighbor_rrf.trace.json`: trace láng giềng cho từng câu.

## TASK3 — E1 dùng chung Local và Kaggle

Entrypoint sinh câu trả lời duy nhất là `python -m legalqa predict`. Mọi lần
`predict_file` hoàn tất đều tạo `<output>.manifest.json` với Git commit, code hash,
config hash, input/output hash, decode config và adapter hash. Dry-run local phải ghi rõ
`--schema-only`; artifact này chỉ chứng minh schema/wiring và có
`quality_status=NOT_APPLICABLE_SCHEMA_ONLY`.

```powershell
.\.venv\Scripts\python.exe -m legalqa validate-config --config configs\e1_rag.yaml
.\.venv\Scripts\python.exe -m legalqa accept-e1-index `
  --release-root Data\Task2\releases\task2-data-v3 `
  --report Data\Task2\preflight\task2-data-v3\phase_b_index_acceptance.json `
  --config configs\e1_rag.yaml
.\.venv\Scripts\python.exe -m legalqa retrieve "Mức phạt đối với hành vi không đội mũ bảo hiểm là bao nhiêu?" --config configs\e1_rag.yaml
.\.venv\Scripts\python.exe -m legalqa evaluate-retrieval `
  --release-root Data\Task2\releases\task2-data-v3 `
  --max-examples 700 --output Data\Task2\runs\c1_recall_v3.json `
  --config configs\e1_rag.yaml
.\.venv\Scripts\python.exe -m legalqa predict --schema-only `
  --input Data\Task2\releases\task2-data-v3\qa\public.jsonl `
  --limit 5 --output Data\Task2\runs\task3_local_smoke_v4\smoke.json `
  --config configs\e1_rag.yaml
```

Không chạy `build-index` với config E1: runtime đọc trực tiếp bundle candidate Phase A
`bm25.sqlite3/index_manifest.json`. `accept-e1-index` kiểm release/index binding, SHA-256,
record count và SQLite integrity. Kết quả qrels vẫn là `diagnostic_proxy`, quyết định luôn ở
`USER_REVIEW_REQUIRED` và không có quyền promote.

Chuẩn bị cache SFT chạy local trước P4:

```powershell
.\.venv\Scripts\python.exe -m legalqa prepare-e1-sft --release-root Data\Task2\releases\task2-data-v3 --config configs\e1_rag_step2.yaml
```

Trước khi đóng gói code thành Kaggle Dataset, tạo hai hợp đồng hash ngay trên local. Step 1
phải trỏ `--adapter` đến đúng E0 adapter; Step 2 dùng chính cache SFT đã được chuẩn bị:

```powershell
.\.venv\Scripts\python.exe -m legalqa expected-hashes `
  --release-root Data\Task2\releases\task2-data-v3 `
  --input Data\Task2\releases\task2-data-v3\qa\validation.jsonl `
  --adapter models\e0-full-v1 `
  --index-manifest Data\Task2\indexes\task2-data-v3-bm25-candidate\index\index_manifest.json `
  --output expected_hashes_step1.json --config configs\e1_rag_step1.yaml
.\.venv\Scripts\python.exe -m legalqa expected-hashes `
  --release-root Data\Task2\releases\task2-data-v3 `
  --input Data\Task2\runs\e1-full-v1\train_sft.jsonl `
  --index-manifest Data\Task2\indexes\task2-data-v3-bm25-candidate\index\index_manifest.json `
  --output expected_hashes_step2.json --config configs\e1_rag_step2.yaml
```

Không tạo hai file hash này từ file placeholder. Nếu input, index manifest hoặc adapter chưa
tồn tại, CLI dừng lỗi; đó là NO-GO cho việc đóng gói Kaggle Dataset.

Hai notebook `notebooks/task3_step1_probe.ipynb` và
`notebooks/task3_step2_train.ipynb` chỉ import package `legalqa`, kiểm hash fail-closed,
và dừng sau validation. Step 2 không build submission ZIP; User phải ký GO trước P4 và
review metric/manifest trước mọi bước Public/submission.
