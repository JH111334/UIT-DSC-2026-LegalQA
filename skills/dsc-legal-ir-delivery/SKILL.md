---
name: dsc-legal-ir-delivery
description: Deliver UIT DSC Task 1 LegalIR changes with bounded candidates and reproducible retrieval evidence.
---

# Task 1 delivery

## Entry gate

1. Read `AGENTS.md`, `Workflows/flowinfo.md`, and the Task 1 ADR.
2. Confirm corpus manifest, split, qrels, access scope, and organizer output schema.
3. Run BM25 + TF-IDF/RRF before enabling a neural adapter.
4. Reject model IDs absent from `configs/Shared/model_allowlist.json`.

## Experiment gate

- Preserve document/article IDs, component scores, ranks, and corpus version.
- Retrieve broad candidates before bounded reranking.
- Compare identical query sets; report retrieval metrics, latency, memory, and failure rate.
- Mark Macro-F2 as local until the organizer confirms the official metric.
- Store models, indexes, organizer data, runs, and submissions outside Git.

## Completion gate

Run the root retrieval-delivery gate plus local warm-up validation when the file is present.
