# Bàn giao Phase A sang Phase B — Task 2 LegalQA

- **Tên cũ**: `checkoutpreprocess.md`.
- **Trạng thái contract**: `ACCEPTED_V2`.
- **Mục tiêu**: Phase B nhận đúng release/index, kiểm được lineage và không phải preprocess
  lại.
- **Nguồn machine-readable**: `Source/Task2/Offline/required_artifacts.json`.

## 1. Ranh giới sở hữu

```text
PHASE A — đồng đội
raw organizer data
  -> integrity + normalization
  -> group-aware QA split
  -> legal parser + hierarchical chunks + metadata
  -> citation-derived diagnostic qrels
  -> immutable data release
  -> optional BM25 candidate index bundle ngoài release
                    |
                    | manifests + hashes + reports
                    v
PHASE B — tác giả/agent
acceptance preflight
  -> tokenizer/model controls
  -> canonical E0/E1 run
  -> retrieval/context/generation
  -> metrics/error/paired comparison
  -> submission replay
```

Phase B không normalize, resplit hoặc rechunk âm thầm. Nếu Phase B phát hiện root cause ở
data/chunking, tạo change request để Phase A phát hành release ID mới.

Index được Phase A build vẫn là artifact dẫn xuất, không nằm trong immutable data release.
Phase B phải verify index manifest hoặc rebuild. Dense index không bắt buộc cho E0/E1.

## 2. Bốn bundle bắt buộc

Không rút xuống sáu file vì sẽ mất evidence kiểm soát leakage/provenance. Handoff tối giản
được tổ chức thành bốn bundle thay vì hàng chục file rời.

### Bundle A — Release control

```text
<release>/
├── manifest.json
├── data_report.json
├── split_manifest.json
└── leakage_report.json
```

Yêu cầu:

- `task_id=Task2`, release ID/version rõ và `status=READY|PASS` đúng schema;
- mọi artifact canonical có SHA-256 và record count trong manifest;
- split `group-aware`, train/validation group overlap bằng 0;
- cross-task hit, Public label usage và Private label usage bằng 0;
- raw source, transform, split và parser version truy được.

### Bundle B — QA payload

```text
<release>/qa/
├── train.jsonl
├── validation.jsonl
├── public.jsonl
└── public_manifest.json
```

Private chỉ thêm `private.jsonl` và `private_manifest.json` khi BTC phát hành. Public/Private
không chứa `answer`, `answer_raw` hoặc `answer_model`.

Train/validation record tối thiểu giữ:

```text
question_id, source_task, question_raw, answer_raw,
question_model, answer_model, question_group, split,
source_sha256, transform_version
```

Public/Private giữ cùng lineage nhưng không có answer fields.

### Bundle C — Corpus payload và provenance

```text
<release>/corpus/
├── corpus_manifest.json
├── raw_inventory.jsonl
├── documents.jsonl
├── chunks.jsonl
├── quarantine.jsonl
└── corpus_report.json
```

`chunks.jsonl` phải đủ để search và parent expansion:

```text
chunk_id, doc_id, source_task, parent_chunk_id,
doc_type, doc_number_surface, doc_number_canonical,
chapter, article, clause, point,
raw_text, canonical_text, retrieval_text, parent_text,
source_sha256, source_start, source_end,
parser_version, transform_version, indexable, quality_flags
```

Quarantine giữ provenance; không xóa record raw. Corpus report phải reconcile document,
indexable/non-indexable, quarantine, chunk, orphan/span failure và token statistics.

### Bundle D — Diagnostic qrels

```text
<release>/diagnostics/
├── citation_qrels.jsonl
├── citation_match_report.json
└── citation_manual_audit.jsonl    # khi review hoàn tất
```

Qrels luôn ghi `qrels_status=diagnostic_proxy`, confidence, parser/matcher version và match
reason. Manual audit pending không chặn E0, nhưng chặn mọi promotion dựa trên qrels.

## 3. Attachment có điều kiện

### Tokenizer report

Phase A có thể bàn giao attachment:

```text
<control>/tokenizer/tokenizer_report_qwen.json
```

Nếu không có, data release vẫn có thể `READY`; Phase B bắt buộc tạo/verify report bằng đúng
model/tokenizer snapshot trước training. Report phải đo question, target, retrieval chunk,
packed context, total sequence và truncation theo revision/prompt version.

### BM25 index bundle

Nếu Phase A build index local:

```text
<index-run>/index/
├── bm25.sqlite3
└── index_manifest.json
```

Manifest tối thiểu chứa corpus manifest hash, chunks hash, SQLite/FTS tokenizer/config,
artifact hash và indexed chunk count. Phase A không đặt index vào `<release>/` và không gọi
nó canonical nếu chưa qua Phase B acceptance.

Current CLI `build-index` cần `parent_run_root` E0 để tạo lineage E1. Nếu Phase A không có
parent/control bundle do Phase B cung cấp, hãy bàn giao release trước và để Phase B build
canonical index; không dùng lệnh ngoài contract để né gate.

Dense/FAISS, reranker và citation graph không thuộc handoff bắt buộc. Citation graph giữ
`OPTIONAL_DISABLED`.

## 4. Status đã kiểm định của `task2-data-v2`

Theo import/preflight ngày 02-09-2026:

- 6.300 train, 700 validation và 1.000 Public;
- 8.532 documents, 8.502 indexable và 316.100 chunks;
- 917 qrel pairs từ 343/700 validation questions, coverage khoảng 49%;
- manual citation audit 0/150, `PENDING`;
- 30 documents non-indexable gồm empty, paywall và exact-duplicate alias theo report;
- manifest canonical SHA-256 là
  `0ea1b4c86b12dcb11d0ee76ff99252ee6cd1af3ef6fa5fb28b6d4320c54a0162`;
- QA payload giữ nguyên hash so với v1; corpus/source hash đã đổi;
- tokenizer attachment Qwen được reuse cho E0 vì QA/model/prompt không đổi và có `reuse_basis`;
- E1 canonical index chưa được coi là hoàn tất chỉ từ data handoff.

Source ZIP chỉ được lấy release/preflight v2; `.git`, `.venv`, code, artifact v1, model, cache,
run và submission trong ZIP đều bị loại. Import report nằm tại
`Data/Task2/control/imports/dsc2026-zip-20260902/import_report.json`.

Code Phase A trong ZIP cũng đã được đối chiếu riêng: 25 file thuộc
`Source/Task2/Offline` và `configs/Task2/preprocessing`, 22 file tương đương sau khi bỏ khác
biệt CRLF/LF; ba file local còn lại chỉ bổ sung release identity v2 và gate chống mismatch.
Báo cáo machine-readable nằm tại
`Data/Task2/control/imports/dsc2026-zip-20260902/phase_a_code_sync_report.json`. Kết luận:
không thiếu implementation Phase A và không cần rebuild release chỉ vì điểm Public lịch sử.

## 5. Quy trình chạy lại Phase A

### A0 — Change request

Trước khi chạy, ghi:

```text
parent_release_id
new_release_id
one_main_data_change
hypothesis/failure targeted
config/code revision
expected artifact delta
stop condition
```

Không overwrite `task2-data-v1`. Dùng `task2-data-v2` hoặc ID có version tương ứng.

### A1 — Khóa request JSON

Ví dụ schema đúng với code hiện tại:

```json
{
  "schema_version": "task2-preprocess-request-v1",
  "operation": "build",
  "config": "../../../configs/Task2/preprocessing/release_v1.yaml",
  "release_root": "../../../Data/Task2/releases/task2-data-v2",
  "validation_size": 700,
  "report": "../../../Data/Task2/preflight/task2-data-v2/build_report.json"
}
```

Mọi path tương đối được resolve cạnh file request. Tên config/version phải được đổi nếu
logic transform thay đổi; ví dụ trên chỉ minh họa schema, không cho phép giả danh v1 logic
thành v2.

### A2 — Build qua entrypoint duy nhất

```powershell
uv sync --locked
uv run python "Source\Task2\pipeline.py" contract
uv run python "Source\Task2\pipeline.py" run --request <build-request.json>
```

Không dùng các module `build_release`, `audit_leakage`, `freeze_manifest` hoặc
`paired_eval` được nêu trong bản nháp cũ: các CLI đó không tồn tại trong repo hiện tại.

### A3 — Deep release preflight

Tạo request cùng schema với `operation=preflight`, `profile=e0-direct`,
`stage=training`; chạy lại entrypoint trên. Nếu chuẩn bị corpus cho E1, chạy thêm
`profile=e1-bm25`, `stage=evaluation`.

Public/Private preflight thuộc phase tương ứng và chỉ chạy khi payload/control tương ứng
tồn tại.

### A4 — Delta review

So parent với candidate:

- record counts, hashes và excluded reasons;
- train/validation group assignment;
- Unicode/newline/legal structure preservation;
- parser coverage, orphan/span failures và chunk-size distribution;
- qrels match/confidence/coverage;
- bounded sample gồm changed, quarantined, longest và random IDs có seed.

Không dùng một aggregate đẹp để bỏ qua regression record-level.

### A5 — Freeze và bàn giao

Chỉ bàn giao khi build report và preflight PASS, artifact manifest reconcile, release path
read-only theo convention và change summary nêu rõ khác biệt so với parent. Teammate gửi
đường dẫn + hash, không gửi raw payload qua chat/service.

## 6. Phase B acceptance

Phase B kiểm theo thứ tự:

- [ ] manifest/schema/task/source đúng Task 2;
- [ ] hashes và record counts khớp;
- [ ] group overlap và cross-task hits bằng 0;
- [ ] QA/corpus/diagnostic bundles đủ theo profile;
- [ ] Public/Private không có label;
- [ ] tokenizer report đúng snapshot hoặc được Phase B tạo trước training;
- [ ] optional index nằm ngoài release và manifest gắn đúng corpus/chunks hash;
- [ ] manual qrels status được giữ nguyên;
- [ ] preflight report PASS và được tham chiếu trong approval.

Fail bất kỳ hard gate nào thì không train/evaluate; trả lại Phase A bằng release ID và
evidence path cụ thể.

## 7. Output Phase B không thuộc bàn giao Phase A

Không yêu cầu Phase A tạo:

```text
model/checkpoint manifest
training/decoding/scorer config
validation/public predictions
metrics/slices/case/error/paired reports
submission ZIP
```

Các artifact này thuộc run/control root Phase B. Việc tách này tránh việc một data release
bị hiểu nhầm thành model/index/submission result.

## 8. Definition of ready

### `DATA_READY_E0`

Bundle A+B, QA release preflight PASS và tokenizer attachment có status rõ
`PROVIDED|NOT_PROVIDED`.

### `DATA_READY_E1`

`DATA_READY_E0` + Bundle C+D + E1 release preflight PASS. Index có thể
`CANDIDATE_PROVIDED|NOT_PROVIDED`; canonical index vẫn cần Phase B acceptance.

### `QRELS_READY_FOR_PROMOTION`

Manual audit hoàn tất theo sample contract, match report cập nhật và reviewer decision rõ.

Không trạng thái nào ở trên tự chứng minh model quality hoặc cho phép API/remote upload.
