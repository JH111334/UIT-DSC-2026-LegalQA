# Roadmap

## P0 — Completed smoke baseline

- Versioned corpus and deterministic chunks.
- Access-scoped BM25 and TF-IDF.
- RRF, bounded routes, citations, abstention, evaluation, and CI.

## P1 — Multilingual semantic retrieval

- Sentence Transformer adapter with model and preprocessing version.
- Exact dense-search conformance and BEIR-style evaluation slice.
- Cross-encoder rerank over bounded top-k with latency and ablation.

## P2 — Agent reliability

- Structured query analyzer and tested route registry.
- LLM answer adapter restricted to allowed citation IDs.
- Prompt-injection, contradiction, timeout, and malformed-output tests.

## P3 — Serving and product

- FastAPI service, streaming-independent response contract, and audit events.
- Immutable index releases, health/readiness, observability, and Docker.
- Evidence-first UI with source preview, filters, feedback, and trace inspection.
