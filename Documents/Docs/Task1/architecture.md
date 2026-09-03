# Task 1 — LegalIR

## Contract

Input: organizer query ID and Vietnamese legal question. Output: ordered, unique
organizer document IDs. Corpus IDs, article boundaries, source URI, version, raw
scores, and ranks must survive retrieval and submission serialization.

## Pipeline

```text
legal corpus manifest
  -> article/clause-preserving units
  -> BM25 + TF-IDF smoke baseline
  -> optional dense candidate set
  -> RRF with component ranks
  -> optional bounded cross-encoder rerank
  -> fixed top-5 anchor; optional calibrated adaptive-k
  -> tối đa 5 ordered unique document IDs
```

Dense and reranker stages are disabled in committed configs. The warm-up file has
query labels but no corpus in this repository; it cannot establish retrieval quality.

## Gate

- Compare identical queries and corpus versions.
- Report official Macro Recall trước, Macro Precision làm tie-break.
- Also report Recall@k, MRR, nDCG@k, p50/p95 latency, peak memory, and failures.
- Check duplicate IDs, unknown IDs, empty predictions, and split leakage.
- Audit train passage rỗng/trùng và chia split theo duplicate group.
