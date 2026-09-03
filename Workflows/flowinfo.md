# Delivery workflow

## Invariants

```text
Task 1 manifest -> legal units -> retrieval/fusion/rerank -> document IDs

Task 2 manifest -> QA release + official Task 2 corpus release
  -> E0 supervised generator | approved E1 Task2-only retrieval + generator
  -> answer -> scorer
```

- Preserve query, document, article/chunk, source, access, corpus version, rank, and score.
- Filter access before retrieval and before answer assembly.
- Keep BM25/RRF operational for Task 1 and the generic fixture.
- Không tái sử dụng BM25/RRF/index Task 1 cho Task 2. E1 phải build index từ
  `selected-contexts` có provenance Task 2 và chỉ mở sau ADR/provenance gate.
- Sau E1, dense-only là diagnostic; BM25+dense/RRF là hybrid challenger chính.
  Citation graph optional disabled và chỉ mở theo ADR-T2-0004 sau hybrid evidence.
- Accept only model IDs in `configs/Shared/model_allowlist.json`.
- Treat organizer data and generated artifacts as local-only by default. Task 2 may use the
  private, hash-locked Kaggle batch explicitly governed by ADR-T2-0006; no inference API.
- Reject cross-task data, context and trained checkpoints.
- Treat citation-derived qrels as a diagnostic proxy with confidence and coverage;
  never as organizer gold or Public/Private inference input.

## Work loop

1. Select Task 1 or Task 2; read its local skill and decision record.
2. Freeze manifest, split, config, seed, code revision, and hardware profile.
3. Validate a deterministic fixture before any model adapter.
4. Change one component; retain the previous baseline for ablation.
5. Chỉ tune knob gắn với bottleneck đã đo; dùng grid nhỏ coarse-to-fine và ghi
   nguồn cho từng giá trị: official, measured, paired-result hoặc hypothesis.
6. Measure metric đúng task, output integrity, latency, memory, and failures.
7. Promote only a reproducible held-out improvement; otherwise record rejection.
8. Validate submission schema against the current Codabench/BTC contract.

## Evidence labels

- `official`: organizer page, file, rule, or metric.
- `measured`: reproducible local run with frozen inputs.
- `prior-art`: external paper or implementation; not DSC evidence.
- `decision`: internal choice supported by constraints.
- `roadmap`: not implemented.
