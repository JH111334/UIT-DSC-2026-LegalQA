# Architecture

## Competition boundary

Hai task là hai hệ thống độc lập về dữ liệu, code, config, checkpoint, experiment
và submission. Không có cạnh Task 1 retrieval → Task 2 answer.

    Task 1 data -> preprocess -> retrieval/rerank -> tối đa 5 document ID

    Task 2 data -> QA release + official Task 2 corpus release
      -> E0 direct generator | approved E1 Task2-only retrieval + generator
      -> deterministic decoding -> METEOR/ROUGE-L -> answer submission

## Implemented smoke profile

- JSONL document contract with source URI, language, access scope, and provenance.
- Deterministic sentence-group chunk IDs.
- Vietnamese-English normalization.
- Access-scoped BM25 and sparse TF-IDF retrieval.
- RRF with component score and rank preservation.
- Lookup, comparison, abstention và extractive answer chỉ là reusable fixture.
- Evaluation fixture cho Recall@k, MRR, abstention và citation integrity.
- Warm-up schema/model policy và project validator.

Profile này chưa triển khai Task 2 SFT hoặc metric chính thức. Xem
Documents/References/MAP.md cho critical path.

## Trust boundaries

Task 1 vẫn giữ provenance và access scope trước candidate generation. Task 2 chỉ
nhận artifact mang provenance Task 2; mọi cross-task path phải fail. Việc cả hai
lane cùng có tên BM25/RRF không cho phép dùng chung corpus, index, code runtime hoặc
trace. Raw organizer data, model, run và submission ở ngoài Git.

## Development adapters

1. Task 2 immutable raw manifest và integrity audit.
2. Group-aware split, corpus legal chunks/provenance và phase-specific handoff gate.
3. Tokenizer/length audit và answer-only SFT cho E0 direct control.
4. Exact METEOR/ROUGE-L scorer parity và case-level error analysis.
5. Task2-only BM25 E1 sau ADR; Task 1 dense/rerank adapters ở lane riêng.
6. Submission validator và clean-environment replay.

<!-- BEGIN competition-product-transfer:v1 -->

## Competition và product transfer boundary

Root architecture tối ưu scoreable/replayable competition slice; không mở multi-tenant,
managed database, object store, vector service hay queue chỉ để giống production.
Capstone `Multimodal-Asset-Retrieval` tái sử dụng contract qua adapter, không import
organizer data/submission package. Mọi transfer phải có domain-neutral test và license/
privacy review.
<!-- END competition-product-transfer:v1 -->
