# ADR-T2-0001: Evidence first, generator optional

Status: superseded by ADR-T2-0002.

Decision: answer from retrieved excerpts first. Compare
`Qwen/Qwen2.5-1.5B-Instruct` with `AITeamVN/Vi-Qwen2-1.5B-RAG` only after the Task 1
release is frozen. Generation remains disabled by default.

Reasons:

- Warm-up has 500 long-form answers but no source corpus or official scoring code.
- A 4 GB GPU favors sequential 1.5B experiments over concurrent 3B/4B models.
- Citation support and abstention remain testable when a model is unavailable.
- `ViLegalQwen*-Base` needs task-specific post-training; models with incomplete
  provenance enter only isolated experiments.

Lý do supersede: thông báo BTC mới xác nhận hai task độc lập, không được dùng data
Task 1 cho Task 2. Contract Codabench Task 2 là direct answer với METEOR/ROUGE-L.
File này chỉ còn mô tả reusable fixture, không phải competition method.
