# TASK1 — Kế hoạch hậu tiền xử lý Task 2 LegalQA

- **Trạng thái**: `PHASE_A_V2_READY; PRIVATE_KAGGLE_E0_REPLAY_APPROVED`
- **Ngày chốt bằng chứng**: 03-09-2026
- **Phạm vi**: training, retrieval, generation, evaluation và submission của Task 2.

## 1. Kiến trúc đã khóa

```text
OFFLINE
train.json -> QA release -> group-aware train/validation -> answer-only SFT
           -> answer citations -> diagnostic qrels

selected-contexts -> integrity -> legal parser -> hierarchy/metadata -> chunks
                  -> E1 BM25
                  -> E2a dense diagnostic
                  -> E2b BM25 + dense -> RRF

ONLINE
question -> BM25 + dense -> RRF -> metadata/context budget
         -> optional reranker/dynamic-k/parent expansion
         -> generator -> answer -> exact submission contract
```

E0 direct generation luôn là control. E1 chỉ dùng BM25 trên Task2-only corpus. Dense
không được promote một mình; mục tiêu neural đầu tiên là hybrid BM25+dense/RRF. Citation
graph giữ `OPTIONAL_DISABLED` cho tới khi paired error evidence chứng minh quan hệ pháp
lý nhiều bước là bottleneck.

## 2. Bằng chứng hiện có

| Artifact | Bằng chứng | Trạng thái |
|---|---|---|
| Data release | 6.300 train, 700 validation, 1.000 Public; 8.532 docs, 316.100 chunks | `READY` |
| Diagnostic qrels | 917 qrels; proxy coverage 49%; manual audit 0/150 | `DIAGNOSTIC_PENDING_REVIEW` |
| Model | Qwen2.5-1.5B-Instruct revision `989aa798...` | `READY` |
| E0 training | QLoRA NF4, 1 epoch, train loss `1.113865` | `TRAINED_NOT_EVALUATED` |
| Adapter | `e0-full-v1/checkpoint-or-adapter`, safetensors SHA-256 `97a34fdf...` | `READY` |
| Public ZIP | SHA-256 `bf9ac32c...`, 1.000/1.000, không rỗng, ZIP replay pass | `FORMAT_PASS` |
| Public method provenance | Runner lịch sử không nạp adapter/pin revision/trace | `FAIL` |
| Validation metrics | Chưa có predictions cho 700 validation | `NOT_RUN` |
| METEOR/ROUGE-L | Public không có reference; local scorer parity còn provisional | `UNAVAILABLE` |
| Điểm Public do operator báo | METEOR `0,18`, ROUGE-L `0,30`, gắn với ZIP SHA-256 `bf9ac32c...` | `OPERATOR_REPORTED; NOT_LOCALLY_VERIFIED` |
| Đồng bộ code Phase A | 25 file so với ZIP: 22 file tương đương sau chuẩn hóa newline, 3 delta an toàn về release identity | `PASS_WITH_INTENTIONAL_LOCAL_DELTAS` |

Không đổi `train_loss` thành quality claim. Không gọi ZIP Public là output SFT E0 cho
tới khi method trace chứng minh base revision, adapter, code, config và input hash. Điểm
`0,18/0,30` hiện chỉ mô tả hành vi của artifact Public lịch sử: runner tạo artifact đó nạp
base Qwen nhưng không nạp LoRA adapter. Vì vậy điểm này không phải bằng chứng rằng SFT,
release v2 hoặc retrieval kém.

## 3. Cấu hình anchor cần replay

```text
base_model_id         = Qwen/Qwen2.5-1.5B-Instruct
base_model_revision   = 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
adapter_type          = QLoRA NF4
seed                  = 2026
epochs                = 1
learning_rate         = 2e-4
max_sequence_length   = 1024
decode                = greedy, do_sample=false
candidate_output_caps = measured grid; 768 is historical, not promoted truth
parameter_budget      < 4B for every running neural component
```

## 4. Observability đã implement

`Online/evaluation.py` có thể tạo `metrics.json`, `slice_metrics.json`,
`case_metrics.jsonl`, `error_table.jsonl` và `paired_comparison.json`. Slices hiện gồm
answer length, legal structure, list, multiline và question type. Paired report có
wins/ties/losses, slice deltas, 20 gains, 20 regressions và counterexamples.

`Online/output_audit.py` kiểm unlabeled Public mà không bịa quality: schema/ID, empty,
duplicate, độ dài, newline/list/legal-marker, repeated 4-gram và ID-only review flags.
Từ phiên bản pipeline hiện tại, mọi lệnh `predict` hợp lệ tự sinh
`<stage>_output_audit.json` và `<stage>_case_flags.jsonl` ngay sau khi ZIP được đóng gói;
không cần một CLI kiểm tra riêng.
Audit hiện tại ghi:

```text
records=1000; empty=0; duplicate=0
characters p50/p95/max=1327/2942/3475
word tokens p50/p95/max=296/643/723
multiline=876; legal-marker=760; list-marker=772
```

Đây chỉ là `measured_local_behavior_only`; không phải METEOR, ROUGE-L hoặc correctness.

## 5. Artifact còn thiếu trong vòng lý tưởng

1. `validation_predictions.json` và `validation_trace.json` của E0 canonical.
2. `metrics.json`, `slice_metrics.json`, `case_metrics.jsonl`, `error_table.jsonl`.
3. Scorer parity với implementation BTC hoặc trạng thái provisional được ghi rõ.
4. `paired_comparison.json` trước mọi PROMOTE.
5. Public trace/run manifest nối input, model revision, adapter, code/config và ZIP hash.
6. Reviewer hoàn tất 150 citation-qrels samples trước khi dùng qrels để promote retrieval.

## 6. Chuỗi tối ưu sau first valid result

```text
RESULT -> OBSERVE -> DIAGNOSE -> HYPOTHESIS -> ONE MAIN CHANGE
       -> PAIRED VALIDATION -> CASE/SLICE REVIEW -> PROMOTE | REJECT | REPEAT
```

Knob không dùng default như chân lý. Chỉ sweep nhóm gắn với bottleneck đã đo:

- truncation/verbosity: max input/output, context budget, length bucket;
- retrieval miss: BM25 top-k/k1/b rồi dense top-k và RRF;
- context noise: dedup, order, parent expansion, final-k;
- generator ignores evidence: prompt/SFT recipe sau khi evidence-survival đã pass;
- resource: batch size, gradient accumulation, quantization, latency/VRAM frontier.

Mỗi challenger ghi parent, hypothesis, một biến chính, stop condition, hashes, paired
delta và counterexample. Average tăng nhưng regression chưa giải thích giữ
`REVIEW_REQUIRED`.

## 7. Bước kế tiếp tối thiểu

Không sửa Phase A, mở E1 hoặc chạy Public lại trước. Trước tiên chạy E0 validation canonical
trên 700 câu bằng engine có trace chứng minh đã nạp đúng adapter, tạo đủ artifact mục 5 và
review 20 case xấu nhất. Đây là sửa gate thực thi bị hỏng sớm nhất, không phải thay đổi
performance knob.
Chỉ nếu evidence cho thấy thiếu căn cứ pháp lý do retrieval miss mới build E1 BM25.
Nếu BM25 miss semantic có paired evidence, mở E2a diagnostic rồi E2b hybrid/RRF.

Public ZIP hiện có thể lưu như candidate format-valid; không dùng nó làm anchor quality
hoặc bằng chứng SFT.

## 8. Chốt vận hành trước lượt chạy kế tiếp

Inventory nhẹ cập nhật ngày 03-09-2026 xác nhận release, tokenizer report và checkpoint E0 đều
`PASS`; không có operation lock. `validation/metrics.json` và canonical
`public_submission_report.json` trong run `e0-full-v1` còn thiếu, nên trạng thái là
`READY_TO_ADVANCE` với đúng bước kế tiếp `evaluate`. Quyết định operator hiện tại là ghi
nhận trạng thái này. Báo cáo đồng bộ code Phase A nằm tại
`Data/Task2/control/imports/dsc2026-zip-20260902/phase_a_code_sync_report.json`.

Môi trường cộng tác chuẩn được đồng bộ bằng `uv sync --locked`. Nó chứa package nội bộ,
bộ checking và dependency nhẹ cho Offline preprocessing/BM25 (`PyYAML`, pytest, Ruff,
mypy và type stub); code lõi dùng standard library/SQLite. Không cài lại Torch,
Transformers, PEFT, bitsandbytes, model hoặc weights. Candidate ML pins vẫn được lưu ở
`requirements.txt` để tái lập một môi trường tách biệt khi có quyết định chạy model.

Khi teammate tạo release hoặc index mới: không overwrite artifact cũ; dùng ID mới, pin
parent checksum/config/code, chạy preflight + canonical validator, rồi paired compare
trước PROMOTE. Nhiều lượt chạy data/index là dự kiến bình thường, nhưng chỉ artifact có
lineage đầy đủ mới được dùng làm anchor.

Authority mới ngày 03-09-2026: operator xác nhận đã hỏi lại BTC và cho phép private Kaggle
batch do đội kiểm soát. ADR-T2-0006 chỉ mở Kaggle orchestration cho E0; model inference API,
Task 1 data, external data/augmentation và secret upload vẫn cấm. Lượt `e0-kaggle-v2-r1`
phải train, chạy validation rồi Public trong cùng payload có hash và nạp đúng adapter.

## 9. Quyết định sau điểm Public `0,18/0,30`

Gate hỏng sớm nhất là `METHOD_EXECUTION_FIDELITY`: artifact format-valid không chạy đúng
phương pháp dự kiến. Thứ tự xử lý bị khóa như sau:

1. Không dùng điểm này để yêu cầu Phase A rebuild hoặc để promote E1.
2. Không đổi đồng thời training length, prompt, decoding và retrieval.
3. Xác thực/replay E0 bằng base revision đã pin + LoRA adapter + trace bất biến.
4. Chạy validation trước Public; local scorer vẫn ghi `PROVISIONAL` tới khi parity BTC đạt.
5. Chỉ sau first valid E0 result mới chọn một bottleneck từ slice/case evidence.

Nếu quyết định retrain exact config trên lineage v2, run đó là `ANCHOR_REPLAY`, không phải
cải tiến. Nếu reuse checkpoint v1 do QA/model/prompt hashes giống nhau, phải tạo explicit
reuse lineage; không được suy từ tên thư mục.
