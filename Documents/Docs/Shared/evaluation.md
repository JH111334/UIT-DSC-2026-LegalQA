# Evaluation

## Current gate

The committed corpus is project-generated and validates behavior, not domain quality.

```powershell
uv run egta evaluate --top-k 3
```

Acceptance:

- three answerable bilingual queries achieve Recall@3 and MRR of `1.0`;
- the unanswerable query is rejected;
- every citation resolves to an authorized retrieved chunk;
- internal content is absent from public candidates and answers;
- agent tool calls and evidence items remain bounded.

## Production evaluation

Freeze the corpus, chunker version, query set, qrels, access policy, configuration,
and Git revision. Report:

- Recall@k, MRR, and nDCG@k by language, domain, and query type;
- hybrid delta against BM25 and dense retrieval separately;
- citation precision, citation coverage, and unsupported-claim rate;
- answerable/unanswerable precision and abstention accuracy;
- scope-leak test count and prompt-injection resistance;
- p50/p95 retrieval, rerank, and answer latency;
- cost and failure rate for every optional model adapter.

LLM output quality must never hide a retrieval miss or unauthorized source.
