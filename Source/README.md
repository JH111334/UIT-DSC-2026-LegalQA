# Hướng dẫn thực thi cục bộ

`Source` là code canonical. Task 1 và Task 2 tách hoàn toàn về data, index,
checkpoint, config và submission. Task 2 chạy local/offline; Hugging Face chỉ dùng
tải snapshot đã pin. Không dùng model inference API. Private Kaggle batch chỉ đi qua
`SourceAPI` theo ADR-T2-0006 và payload Task 2 tối thiểu có hash.

## 1. Entry point và CLI

Task 2 chỉ có hai lệnh để argparse không phình:

```powershell
uv run python Source\Task2\pipeline.py contract
uv run python Source\Task2\pipeline.py run --request <request.json>
```

Mọi path, profile, stage và output nằm trong JSON request có version. Cấu trúc code:

```text
Source/Task2/
├── Offline/
│   ├── QA/                  # QA release, split, leakage
│   └── Corpus/              # audit, parser, chunks, metadata
├── Online/
│   ├── Training/            # tokenizer audit, answer-only QLoRA SFT
│   └── RetrievingAnswer/    # BM25, evidence, generation, submission
└── pipeline.py              # contract | run --request
```

## 2. Môi trường kiểm tra và Offline preprocessing

Môi trường mặc định chỉ cài bộ nhẹ phục vụ hai người kiểm release, chạy lại
preprocessing/index BM25 và validate code. `PyYAML` là dependency ngoài standard library
duy nhất của nhánh Offline hiện tại; `pytest`, Ruff và mypy phục vụ quality gate:

```powershell
$env:UV_CACHE_DIR = "$PWD\.tmp\uv-cache"
uv sync --locked
uv run python Source\Task2\pipeline.py contract
uv run pytest -q --no-cov -p no:cacheprovider
uv run ruff check .
uv run mypy
```

Lệnh trên cài package nội bộ cùng dev group nhưng không cài `torch`, Transformers, PEFT,
bitsandbytes, model hoặc weights vì project dependencies không chứa ML stack. Không
chạy `pip install -r requirements.txt` trong giai đoạn kiểm data. `requirements.txt` chỉ
lưu candidate ML stack để tái lập môi trường training riêng khi team chủ động mở lại.
Cache workspace nằm trong `.tmp/` và bị Git ignore; có thể xóa sau khi kết thúc đợt làm
việc. Cách này tránh lỗi ACL của cache `uv` toàn cục trên máy hiện tại.

Nếu môi trường ML đã tồn tại từ run trước, chỉ kiểm kê; không cài lại trong lượt này:

```powershell
.venv-task2-ml\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Snapshot anchor được pin tại commit
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Chỉ tải nếu manifest báo thiếu:

```powershell
hf download Qwen/Qwen2.5-1.5B-Instruct `
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 `
  --local-dir Data\Task2\cache\models\qwen2.5-1.5b-instruct-989aa798
```

Không tải lại snapshot nếu cache/manifest đã đủ. Không lưu token vào repo, request, log
hoặc notebook.

## 3. Phase A: data build

Canonical release hiện tại là
`Data/Task2/releases/task2-data-v2`. Không rebuild hoặc ghi đè nếu manifest và preflight
đã pass. Muốn tái tạo, dùng một request `task2-preprocess-request-v1`, output sang
release ID mới rồi đối chiếu hash trước khi promote.

Luồng canonical:

```text
train/public -> QA audit -> group-aware train/validation
selected-contexts -> integrity -> legal parser -> hierarchical chunks -> metadata
gold validation answers -> citation-derived diagnostic qrels
```

Kiểm tra release mà không train:

```powershell
uv run python Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\preflight_e0_training_v2.json
uv run python Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\preflight_e1_release_v2.json
```

Qrels là proxy chẩn đoán. Manual audit 150 mẫu còn pending nên không được dùng để
train retriever hoặc làm bằng chứng promotion.

## 4. Phase B: E0 local

Các lệnh Phase B active dùng request v2. Chạy bằng Python environment đã có đủ dependency ML
trên GPU do đội kiểm soát; `uv` local chỉ dùng cho contract, preflight, test và metric nhẹ.

```powershell
$T2PY = "python"
```

Chạy tay theo thứ tự:

```powershell
& $T2PY Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\audit_qwen_tokenizer_v2.json
& $T2PY Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\train_e0_full_v2.json
& $T2PY Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\evaluate_e0_full_v2.json
& $T2PY Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\predict_public_e0_full_v2.json
```

Checkpoint `e0-full-v1` là bằng chứng lịch sử của release v1. Request active dùng run ID
`e0-full-v2`; không overwrite checkpoint cũ. Vì QA v2 giữ nguyên hash, retrain chỉ hợp lý khi
có hypothesis/training change rõ; nếu reuse checkpoint thì phải tạo lineage reuse riêng.
Validation phải hoàn tất và có `metrics.json status=PASS` trước Public.

Decode local hiện là `e0-greedy-768-b8-v2`: greedy, `do_sample=false`,
`max_new_tokens=768`, repetition penalty 1.0 và micro-batch 8. Batch 8 là cấu hình
lớn nhất đã benchmark hoàn tất trên RTX 2050: 8 câu/311,93 giây, 4.900 output token,
không OOM. Đây chỉ là bằng chứng throughput; validation 700 câu đã được dừng theo
quyết định operator nên chưa có quality metric đầy đủ hoặc evidence leaderboard.

Hugging Face trong workflow này là nguồn snapshot, không phải inference endpoint:

```powershell
hf download Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 `
  --local-dir Data\Task2\cache\models\qwen2.5-1.5b-instruct-989aa798
& $T2PY Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\evaluate_e0_full.json
```

Lệnh thứ hai dùng thư viện Hugging Face Transformers nhưng inference chạy trực tiếp
trên GPU local. Full E0 có thể chạy bằng private Kaggle batch qua `SourceAPI`; runtime local
`Source/Task2` vẫn là implementation canonical được đóng gói sang kernel.

## 5. Inventory và advance

Inventory chỉ đọc artifact, nhận diện bước đã hoàn tất và nêu bước kế tiếp:

```powershell
& $T2PY Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\inventory_e0_full.json
```

Advance chạy đúng một bước kế tiếp sau preflight:

```powershell
& $T2PY Source\Task2\pipeline.py run --request `
  Data\Task2\control\requests\advance_e0_full.json
```

Chạy một bước mỗi lần là chủ ý an toàn: release/tokenizer/checkpoint hợp lệ được skip;
evaluation không bị skip trước Public; lỗi không kéo theo một chuỗi GPU job khác.

## 6. E1 và retrieval ladder

Chỉ mở sau E0 validation:

```text
E0 direct SFT
-> E1 Task2-only BM25
-> E2a dense-only diagnostic
-> E2b BM25 + dense -> RRF challenger
```

E1 build index từ đúng `corpus/chunks.jsonl`, kế thừa checkpoint E0, rồi paired compare
trên cùng validation và decoding. Dense model/FAISS/reranker chưa được chọn hoặc cài.
Citation graph là `OPTIONAL_DISABLED`.

## 7. Submission

Artifact hợp lệ nằm tại:

```text
Data/Task2/runs/<run_id>/public_submission.zip
```

ZIP chỉ chứa `submission.json` UTF-8 không BOM. Schema chính xác:

```json
{
  "<question_id>": {
    "answer": "<non-empty Vietnamese answer>"
  }
}
```

Không có extra field, thiếu ID, trùng key hoặc answer rỗng. Private chỉ chạy khi BTC
phát hành payload và manifest thật.

Sau mỗi lệnh `predict`, pipeline tự sinh thêm:

```text
<stage>_submission_report.json
<stage>_output_audit.json
<stage>_case_flags.jsonl
```

Audit khóa đủ ID, empty/duplicate, phân phối độ dài, marker pháp lý/danh sách và cue lặp
4-gram. Đây là observability không nhãn; nó không thay METEOR/ROUGE-L và không được dùng
để tuyên bố chất lượng khi Public/Private không có reference answer.

## 8. Gate cuối

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=text_retrieval_agent --cov-report=term-missing
uv run python skills\retrieval-delivery\scripts\validate_project.py
```

Task 1 tiếp tục dùng entry point và hướng dẫn ở [README gốc](../README.md); không được
dùng runtime/index Task 1 cho các lệnh Task 2 ở trên.
