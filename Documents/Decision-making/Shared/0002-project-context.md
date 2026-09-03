# Ngữ cảnh dự án — DSC-TinDipLaPo

## Vai trò đã khóa

Đây là workspace competition LegalIR/LegalQA, không phải sản phẩm khách hàng. Mục tiêu
là timeboxed iteration trên official specification: baseline chạy được, local validator,
held-out metric, submission replay và postmortem. Không kéo product hosting, generalized
agent platform hoặc hạ tầng capstone vào critical path thi.

## MUST DO

1. Ngày đầu có end-to-end fixture/submission-shaped baseline và format validator.
2. Khóa organizer revision, corpus/split/qrels/evaluator và local submission replay.
3. Kiểm useful content, coverage, mapping và task semantics trước scale/model.
4. Một hypothesis/một biến chính; báo paired case/slice delta và resource budget.
5. Freeze last-known-good sớm; rehearsal clean checkout trước deadline.

## MUST PREVENT

- Architecture/distributed abstraction trước first scoreable result.
- Gọi đủ số row/file là hoàn thành khi payload rỗng, collapse hoặc sai semantics.
- OCR/ASR/generator confidence thay ground truth và task metric.
- Cue nhiều model nhưng không có selection rule, ablation hay rollback.
- Chờ coordinator/máy mạnh rồi mới kiểm parser, query, packaging hoặc evaluator.

Phần cứng 4 GB hạn chế batch/model nhưng không giải thích parser/schema, missing evaluator,
hard-coded output hoặc thiếu rehearsal. Vì vậy thiếu chuẩn bị và feedback loop là nguyên
nhân ưu tiên điều tra trước hardware.

## Ranh giới chuyển giao sang capstone

`Evidence-Grounded-Text-Agent` nhận các pattern bền: manifest, access scope, retrieval,
fusion, evidence, abstention, evaluation và ServiceAdapters fail-closed. Organizer data,
submission schema, leaderboard objective và task-specific shortcut không được chuyển.
Capstone đổi metric từ score thi sang task success, time-to-correct, latency, failure,
privacy và replay trên customer-like corpus.

## Ngữ cảnh refine 26/08/2026

- Trước comparison đọc `Documents/General/RULE.md`; khóa anchor/parent, hypothesis,
  evaluation role, gold status, evaluation contract, slice registry và case budget.
- Học từ paired delta, regression, rescued/persistent failure, counterexample và random
  sample có seed; tách observation khỏi cause.
- Trạng thái hiện hành vẫn là smoke/historical evidence nếu chưa có official benchmark.

Căn cứ kỹ thuật: [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml/)
và [ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/).
