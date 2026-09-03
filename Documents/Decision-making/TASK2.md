# TASK2 — Runbook Phase A/B và tối ưu tiết kiệm compute

- **Trạng thái**: `PHASE_A_V2_ACCEPTED; PRIVATE_KAGGLE_E0_BATCH_APPROVED`.
- **Next gate**: `KAGGLE_E0_V2_TRAIN_VALIDATE_PREDICT`.
- **Ngày chốt**: 03-09-2026.
- **Nguồn sự thật hiện trạng**: [TASK1.md](TASK1.md).
- **Hợp đồng bàn giao**: [BanGiao.md](BanGiao.md).
- **Chiến lược compute**:
  [COMPUTE-AWARE-EXPERIMENT-STRATEGY.md](Task2/COMPUTE-AWARE-EXPERIMENT-STRATEGY.md).

## 1. Nguyên lý duy nhất

```text
Khóa contract và data lineage
  -> quan sát system/slice/case/component từ ngoài vào trong
  -> tìm gate upstream hỏng sớm nhất
  -> screening nhiều candidate rẻ nhưng độc lập
  -> confirmation một can thiệp trên held-out cố định
  -> paired evidence
  -> PROMOTE | REJECT | REPEAT | REVIEW_REQUIRED
```

Không tối ưu bằng cảm giác. Không bundle nhiều performance knobs để tiết kiệm một run.
Tiết kiệm đúng cách là loại candidate ở local/component gate, dùng cache và chỉ đưa ứng
viên mạnh nhất lên full generation.

## 2. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| `task2-data-v2` | `READY`; E0/E1 release preflight `PASS` |
| `task2-data-v1` | Đã loại khỏi vùng active; run lịch sử vẫn được giữ |
| E0 QLoRA checkpoint | `TRAINED_NOT_EVALUATED` |
| E0 validation 700 | `NOT_RUN_CANONICAL` |
| Local scorer | `PROVISIONAL` |
| Public ZIP lịch sử | `FORMAT_PASS; METHOD_PROVENANCE_FAIL` |
| E1 canonical BM25 result | `NOT_PROMOTED` |
| Dense/RRF | `ROADMAP`, dense-only diagnostic |
| Citation graph | `OPTIONAL_DISABLED` |
| Điểm Public do operator báo | METEOR `0,18`, ROUGE-L `0,30`; artifact lịch sử không nạp adapter |
| Đồng bộ code Phase A | `PASS_WITH_INTENTIONAL_LOCAL_DELTAS` trên 25 file |

Bước tiếp theo không phải rerun Phase A. QA train/validation/Public của v2 có cùng hash với v1,
nhưng corpus và raw-source provenance đã đổi. Điểm Public lịch sử không chẩn đoán được Phase A
hoặc SFT vì runner không nạp adapter. Vì vậy cần sửa execution fidelity và chạy E0 validation
canonical trước; E1 phải build lại index từ corpus v2 khi và chỉ khi gate E1 được mở.

## 3. Phân công

### Phase A — Đồng đội, local

Sở hữu:

- raw integrity, minimal normalization;
- group-aware QA split;
- legal parser, hierarchy, metadata và chunks;
- diagnostic qrels và manual-audit status;
- immutable release;
- optional BM25 candidate index ngoài release.

Không sở hữu model run, prediction, score hoặc submission.

### Phase B — Tác giả và agent

Sở hữu:

- acceptance preflight và tokenizer/model/scorer controls;
- canonical E0/E1/E2 run;
- retrieval routing, context packing, generation và decoding;
- metrics, error taxonomy, paired comparison và submission replay.

Không normalize/resplit/rechunk release đã nhận.

## 4. Quyết định reuse hay chạy lại Phase A

### Reuse v1 khi

- không có một data/chunking failure được trace hỗ trợ;
- config/code preprocessing không đổi;
- hashes/preflight còn hợp lệ;
- mục tiêu vòng tới chỉ là E0 validation hoặc Phase B knob.

### Tạo v2 khi

- sửa normalization/split/parser/chunk/metadata/qrels;
- artifact/report v1 fail hard gate;
- một hypothesis Phase A có expected delta và stop condition rõ.

Không rebuild chỉ để “cho chắc”. Rebuild 2,7 GB không tạo thêm bằng chứng nếu input/config
hash không đổi.

### Ảnh hưởng tới checkpoint/index

| Thay đổi | E0 checkpoint | BM25 index |
|---|---|---|
| chỉ corpus/chunk/index | có thể reuse nếu QA/model hashes giữ nguyên | rebuild |
| QA normalization/split/train | không reuse làm canonical v2; cần retrain | chỉ rebuild nếu corpus đổi |
| tokenizer/prompt serializer | cần audit compatibility; thường retrain | không nhất thiết rebuild |
| scorer/slice registry | không retrain | không rebuild; chỉ recompute metrics |

Mọi reuse phải được chứng minh bằng hash, không dựa vào tên folder.

## 5. Quy trình Phase A

1. Ghi change request với parent/new release ID và một thay đổi chính.
2. Nếu không có thay đổi, chạy preflight read-only và bàn giao v1.
3. Nếu có thay đổi, build release mới bằng JSON request qua entrypoint duy nhất:

```powershell
uv run python "Source\Task2\pipeline.py" contract
uv run python "Source\Task2\pipeline.py" run --request <phase-a-request.json>
```

4. Chạy E0-training và E1-evaluation release preflight phù hợp.
5. Review delta counts/hashes, leakage, Unicode/newline, parser/chunk và qrels.
6. Freeze release; optional index nằm ngoài release.
7. Bàn giao theo [BanGiao.md](BanGiao.md).

Không dùng các lệnh `build_release`, `freeze_manifest` hoặc `paired_eval` trong bản strategy
cũ vì các CLI đó không tồn tại.

## 6. Phase B sau khi nhận release

1. Verify handoff và xác định artifact nào reuse/rebuild bằng hash.
2. Tạo/fix `evaluation_panel_manifest.json` với probe và confirmation IDs không giao nhau.
3. Chạy E0 canonical validation trước retrieval challenger.
4. Sinh bộ metric tối thiểu và phân loại failure.
5. Chọn một bottleneck upstream có evidence.
6. Lập experiment batch tối đa ba candidate độc lập cho cùng một knob.
7. Local/component screening, rồi probe; chỉ top candidate vào confirmation/full 700.
8. Ghi paired decision.
9. Chỉ mở E1 BM25 nếu experiment goal hợp lệ; E2a dense diagnostic và E2b RRF theo sau.

## 7. Bộ metric tối thiểu

Không tạo thêm nhiều report nếu field có thể nằm trong artifact đã có.

| Artifact | Nội dung sống còn |
|---|---|
| `run_manifest.json` | profile, parent, data/model/config/code/scorer/slice hashes, seed, resource budget |
| `metrics.json` | evaluated/expected IDs, METEOR, ROUGE-L, empty/invalid/cap-hit, latency/throughput/peak VRAM khi measured |
| `slice_metrics.json` | `n` và metric cho length, legal structure và question type; list/multiline giữ nếu đã có |
| `case_metrics.jsonl` | ID, per-case metric, length/cap/repetition flags và trace refs |
| `component_metrics.json` | chỉ E1/E2: qrels coverage, Recall@20 candidate, Recall@final-k, evidence survival |
| `error_table.jsonl` | deterministic/heuristic/human-required label, confidence, evidence refs, review status |
| `paired_comparison.json` | delta, paired interval, wins/ties/losses, slice guardrail, gains/regressions và decision |

Rút bỏ khỏi canonical output:

- `resource_metrics.json`: gộp resource fields vào `metrics.json`;
- `evaluation_manifest.json`: gộp contract hashes vào `run_manifest.json`;
- `error_summary.json`: aggregate trực tiếp vào `metrics.json`;
- `priority_queue.jsonl`: tạo on-demand từ `error_table.jsonl`, không lưu canonical.

E0 ghi retrieval/context metric là `not_applicable`, không ghi zero. Qrels metric luôn
`diagnostic_proxy`. Scorer local ghi `PROVISIONAL` cho tới parity.

Không dùng universal thresholds như METEOR > 0,35, Recall@5 > 0,75, survival > 95% hoặc
length ratio 0,85–1,15. Chúng chưa có evidence DSC. Dùng parent-relative delta, coverage,
paired uncertainty và slice guardrails.

## 8. Error và causal priority

Hard gate trước mọi xếp hạng:

- scorer/serialization drift;
- leakage/cross-task contamination;
- missing/extra ID, invalid ZIP hoặc method provenance;
- data/model/config hash không truy được.

Sau hard gate, xếp hạng failure instance theo frequency, impact, causal reach,
actionability và confidence. Nhãn heuristic không tự là root cause.

```text
contract -> data/split -> tokenizer/budget -> retrieval -> packing
         -> generator -> decoding -> submission
```

Sửa gate upstream hỏng sớm nhất. Khi Phase A là root cause, Phase B gửi change request và
không vá dữ liệu trong runtime.

## 9. Compute-aware experiment round

```text
G0 local contract
 -> G1 local component screening
 -> G2 frozen probe generation
 -> G3 disjoint confirmation
 -> full 700 report
 -> sequential integration
```

Một round được khảo sát nhiều candidate, nhưng mỗi candidate chỉ đổi một scientific knob
so với parent. Bundle chỉ sau independent evidence hoặc ablation interaction. Không đặt số
run/ngày; ledger quyết định bằng expected information per GPU minute.

Wins > Losses chỉ là diagnostic. Promotion cần METEOR/ROUGE-L paired delta, paired
uncertainty, slice guardrail và regression review. Full 700 không dùng cho mọi thay đổi nhỏ
để giảm adaptive overfit validation.

## 10. Hardware và service gate

Local RTX 2050 4 GB dùng cho preprocessing/index/BM25, static checks, metric, trace và
micro-probe. Full SFT/generation chỉ chạy trên GPU do đội kiểm soát sau approval.

Budget Kaggle 30 giờ/tuần là `operator-stated`. Operator xác nhận lại với BTC ngày
03-09-2026 và ADR-T2-0006 cho phép private Kaggle batch do đội kiểm soát. Chỉ upload ba
QA split E0 cùng control/runtime đã hash; không upload corpus vì E0 không dùng retrieval.
Kaggle API chỉ orchestration, Hugging Face chỉ tải weight revision đã pin; inference API
vẫn cấm. Không ghi thời gian/batch thành measured truth nếu thiếu run trace.

## 11. Next actions

1. Phase B dùng `task2-data-v2`; không dùng lại request active của v1.
2. Chạy request `kaggle_e0_v2_r1.json`: retrain exact E0 anchor, validation rồi Public trong
   cùng batch. Đây là replay để sửa method provenance, không phải performance challenger.
3. Chỉ import ZIP khi remote manifest chứng minh adapter load/hash và local ZIP replay PASS.
4. Nếu mở E1, build lại BM25 bằng `build_index_e1_v2.json`; không reuse index corpus v1.
5. Chỉ chạy Public sau validation, metric/error trace và submission preflight đạt gate.

`e0-full-v2` nếu chạy với config hiện tại là canonical replay/anchor trên lineage v2, không phải
candidate cải tiến. Muốn gọi là experiment tối ưu phải khai báo một scientific knob khác parent.

Điểm METEOR `0,18` và ROUGE-L `0,30` chỉ được ghi `operator_reported_public_score` gắn với
SHA-256 artifact. Không dùng nó làm parent metric cho paired comparison vì method trace fail.

## 12. Stop rules

- Không rerun Phase A khi không có input/config change hoặc failed gate.
- Không bundle performance knobs chưa ablate.
- Không full-run candidate chưa pass local/probe gate.
- Không dùng Public làm tuning loop.
- Không mở dense/reranker/graph trước E1 và evidence tương ứng.
- Không promote từ train loss, format pass, proxy qrels hoặc aggregate metric đơn lẻ.
