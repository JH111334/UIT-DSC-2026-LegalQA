# Delivery workflow

## Invariants

```text
manifest -> deterministic legal units -> scoped candidate generation
  -> sparse/dense fusion -> bounded reranking -> Task 1 document IDs
  -> bounded Task 2 answer -> retrieved citations or abstention
```

- Preserve query, document, article/chunk, source, access, corpus version, rank, and score.
- Filter access before retrieval and before answer assembly.
- Keep BM25/RRF and extractive fallback operational without neural models.
- Accept only model IDs in `configs/Shared/model_allowlist.json`.
- Treat organizer data and generated artifacts as local-only.

## Work loop

1. Select Task 1 or Task 2; read its local skill and decision record.
2. Freeze manifest, split, config, seed, code revision, and hardware profile.
3. Validate a deterministic fixture before any model adapter.
4. Change one component; retain the previous baseline for ablation.
5. Measure retrieval, evidence integrity, abstention, latency, memory, and failures.
6. Promote only a reproducible held-out improvement; otherwise record rejection.
7. Validate submission schema only after BTC publishes the authoritative format.

## Evidence labels

- `official`: organizer page, file, rule, or metric.
- `measured`: reproducible local run with frozen inputs.
- `prior-art`: external paper or implementation; not DSC evidence.
- `decision`: internal choice supported by constraints.
- `roadmap`: not implemented.
