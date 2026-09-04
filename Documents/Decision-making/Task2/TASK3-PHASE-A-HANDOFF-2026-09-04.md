# TASK3 Phase A — Session handoff 2026-09-04

## Trạng thái tại điểm dừng

- Phase A implementation: `DONE`.
- `task2-data-v3` build: `READY`.
- E0 training preflight: `PASS`.
- E1 BM25 evaluation preflight: `PASS`.
- External SQLite FTS5/BM25 bundle: `CANDIDATE`, build report `PASS`.
- Human freeze/approval, Git commit và GitHub push: `PENDING_NEXT_SESSION`.
- Không chạy GPU, training, inference chất lượng, submission hoặc checkpoint promotion.

Không rebuild hoặc ghi đè release/index hiện có. Mọi lần chạy lại phải dùng release/index ID mới
hoặc chỉ chạy kiểm tra read-only.

## Artifact canonical local

### Release

- Root: `Data/Task2/releases/task2-data-v3`
- Manifest: `Data/Task2/releases/task2-data-v3/manifest.json`
- Release ID: `task2-data-v3`
- Status: `READY`
- Release manifest SHA-256: `d6ca3086b0fe4e3883769347fcd1b84da1e8d2d83a02d57957e5de7c6b70a09d`
- QA transform: `task2-answer-sanitation-v2`
- Corpus transform: `task2-corpus-v2`

### Index bundle

- Root: `Data/Task2/indexes/task2-data-v3-bm25-candidate/index`
- Manifest: `Data/Task2/indexes/task2-data-v3-bm25-candidate/index/index_manifest.json`
- Status: `CANDIDATE` — Phase B acceptance vẫn bắt buộc.
- SQLite artifact SHA-256: `292bfc540208c42165320994e52098d8b58c2692cb5668c64891aa07d12bf75d`
- Chunks SHA-256: `8bfc957c20f0277d85629f88fbc9906eea752fc2de3bd4d1e3212e54b6c6b82f`
- Corpus manifest SHA-256: `f1a5da22ccf62f321515d219a323cef10eaf60d041b092303c87da68c1f82fe4`
- Indexed chunks: `316100`
- SQLite integrity: `ok`

## Gate evidence

### A1 — QA sanitation

- Report: `Data/Task2/releases/task2-data-v3/qa/sanitation_report.json`
- QA checked: `7000`
- `answer_model` changed by bounded sanitation: `2041`
- `answer_raw` changes versus parent: `0`
- Media-note residue after: `0`
- Related-question ellipsis residue after: `0`
- Deterministic random QA sample: `20`
- Literal ASCII form placeholders such as `.........` are outside the ellipsis deletion rule.

### A2 — Legal chunking and metadata

- Corpus report: `Data/Task2/releases/task2-data-v3/corpus/corpus_report.json`
- Documents: `8532`
- Chunks: `316100`
- Quarantine: `30`
- Every chunk contains the fields `doc_id`, `article_id`, and `clause_id`.
- Article IDs populated: `290183`; clause IDs populated: `71256`; null is permitted for
  preamble/fallback or levels without that hierarchy.
- Form audit: `10/10` preserved from `1393` eligible documents.
- Orphan/source-span/empty retrieval hard gates: `PASS` through deep E1 preflight.

### A3 — Four release bundles and index attachment

- Release control, QA payload, corpus payload and diagnostic qrels are present and manifested.
- External candidate index is bound to release/corpus/chunks hashes and is not placed inside the
  immutable release.
- E0 report: `Data/Task2/preflight/task2-data-v3/preflight_e0_training.json`
- E1 report: `Data/Task2/preflight/task2-data-v3/preflight_e1_evaluation.json`
- Index build report: `Data/Task2/preflight/task2-data-v3/build_index_report.json`
- E1 warning: `150` citation manual-audit records remain pending. This does not block E1
  evaluation handoff, but it blocks qrels-based promotion.

## Paired v2 → v3 audit

- Shared QA IDs: `7000`; ID delta: `0`.
- Split changes: `0`; question-group changes: `0`; question-model changes: `0`.
- `answer_raw` changes: `0`; `answer_model` changes: `2041`.
- Paired chunks: `316100`.
- Chunk payload mismatches after excluding the expected `article_id`, `clause_id`, and
  `transform_version` fields: `0`.
- Corpus/qrels scale remains fixed: `316100` chunks, `917` qrels, judged coverage `0.49`.
- Parent caveat: directory `task2-data-v2` carries legacy manifest ID `task2-data-v1`; do not use
  that parent manifest as a valid acceptance identity. The paired audit used it only as a
  record-level comparison source.

## Code/config/request đã thay đổi cho Phase A

- `Source/Task2/Offline/Corpus/sanitation.py`
- `Source/Task2/Offline/release_builder.py`
- `Source/Task2/Offline/release_audit.py`
- `Source/Task2/Offline/pipeline.py`
- `Source/Task2/Online/RetrievingAnswer/bm25.py`
- `configs/Task2/preprocessing/release_v3.yaml`
- `requests/Task2/phase_a_v3/*.json`
- `tests/Task2/test_preprocessing_handoff.py`
- `tests/Task2/test_training_test_runtime.py`
- `Documents/Decision-making/BanGiao.md`

Test gần nhất: `30 passed` cho toàn bộ `tests/Task2`; test handoff/index mục tiêu cũng PASS.

## Việc phiên sau phải làm

1. Đọc handoff này và ba report preflight/build-index; không chạy lại build.
2. Chạy `git diff --check` và `python -m pytest tests/Task2 -q`.
3. User ký human freeze/approval cho release manifest SHA-256 ở trên. Không tự tạo chữ ký giả.
4. Stage **chỉ** các file Phase A liệt kê ở mục trước. Không stage `Data/`, `artifacts/`,
   `Documents.zip`, các request cũ ở root, `submissions/public_submission.zip`, hoặc các thay đổi
   Phase B trong `src/legalqa`, `README.md`, `pyproject.toml`.
5. Commit gợi ý: `feat(task2): publish phase A data-v3 pipeline contract`.
6. Push nhánh `master` lên `origin` chỉ sau khi review staged diff. Payload nhiều GB và SQLite
   không được đẩy lên GitHub; Git commit chỉ chứa code/config/request/handoff nhỏ. Teammate nhận
   payload bằng đường dẫn artifact + hashes ở trên.

## Lệnh replay đã dùng

```powershell
.\.venv\Scripts\python.exe Source\Task2\pipeline.py run --request requests\Task2\phase_a_v3\build_release.json
.\.venv\Scripts\python.exe Source\Task2\pipeline.py run --request requests\Task2\phase_a_v3\preflight_e0.json
.\.venv\Scripts\python.exe Source\Task2\pipeline.py run --request requests\Task2\phase_a_v3\preflight_e1.json
.\.venv\Scripts\python.exe Source\Task2\pipeline.py run --request requests\Task2\phase_a_v3\build_index.json
```

Các lệnh build không idempotent theo kiểu overwrite: target tồn tại phải được coi là bằng chứng
đã chạy, không xóa để chạy lại nếu chưa có change request/release ID mới.
