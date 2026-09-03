# Roadmap — competition execution và chuyển giao

## P0 — Deterministic smoke baseline đã triển khai

- Stable chunks, access-scoped BM25/TF-IDF, RRF, extractive evidence và abstention.
- Fixture/unit tests chỉ chứng minh contract, chưa phải competition-quality score.

## P0.5 — Task 2 preprocessing và first trained run

- Freeze Task 2-only raw checksum, schema, split, tokenizer report và submission contract.
- Xây validate → preprocess → train smoke → predict → score → package.
- Chạy Qwen2.5-1.5B anchor và giữ run manifest/error table.

## P1 — Task challengers

- Task 1: dense encoder rồi bounded reranker, từng stage paired với BM25/RRF.
- Task 2: direct generator; Qwen2.5-1.5B anchor rồi ViLegalQwen3-1.7B-Base SFT.
- Một thay đổi chính mỗi run; giữ last-known-good.

## P2 — Integration và rehearsal

- Tách tuyệt đối artifact hai task, resource budget, packaging và clean-environment replay.
- Freeze submission candidate trước deadline; chỉ nhận fix có regression evidence.

## P3 — Postmortem và transfer

- Lưu failure taxonomy, rejected hypotheses và reproducible artifacts.
- Chỉ chuyển pattern bền sang Text Asset Retrieval capstone; không chuyển organizer data,
  score objective hoặc submission-specific code vào product core.
