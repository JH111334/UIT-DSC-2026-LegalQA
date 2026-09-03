# Evaluation

## Current fixture gate

The committed corpus is project-generated and validates behavior, not domain quality.

```powershell
uv run dsc evaluate --top-k 3
```

Acceptance:

- three answerable bilingual queries achieve Recall@3 and MRR of `1.0`;
- the unanswerable query is rejected;
- every citation resolves to an authorized retrieved chunk;
- internal content is absent from public candidates and answers;
- agent tool calls and evidence items remain bounded.

Các acceptance trên chỉ là synthetic fixture, không phải metric DSC.

## Competition evaluation

Freeze the corpus, chunker version, query set, qrels, access policy, configuration,
and Git revision. Report:

- Task 1 Macro Recall chính, Macro Precision tie-break, cùng Recall@k/MRR/nDCG
  diagnostic và contract tối đa 5 ID;
- hybrid delta against BM25 and dense retrieval separately;
- Task 2 METEOR chính, ROUGE-L phụ, exact scorer parity và case/slice delta;
- empty output, truncation, repetition và cấu trúc Điều/Khoản;
- scope-leak test count and prompt-injection resistance;
- p50/p95 retrieval, rerank, and answer latency;
- cost and failure rate for every optional model adapter.

Không dùng metric Task 1 cho Task 2 hoặc ngược lại. Local scorer không được gọi là
official nếu chưa khớp implementation của BTC.
