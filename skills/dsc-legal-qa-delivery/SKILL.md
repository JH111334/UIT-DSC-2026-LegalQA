---
name: dsc-legal-qa-delivery
description: Deliver UIT DSC Task 2 LegalQA changes with retrieved evidence, citations, fallback, and abstention.
---

# Task 2 delivery

## Entry gate

1. Read `AGENTS.md`, `Workflows/flowinfo.md`, and the Task 2 ADR.
2. Freeze the Task 1 retrieval release and active corpus version.
3. Validate extractive fallback before enabling a generator.
4. Reject model IDs absent from `configs/Shared/model_allowlist.json`.

## Answer gate

- Pass only bounded, authorized chunks to the answer stage.
- Require every citation and quoted span to resolve to retrieved evidence.
- Never invent a law, article, source, or document ID.
- On no evidence, timeout, model failure, or malformed output: extract or abstain.
- Evaluate answer quality separately from retrieval quality and record failure behavior.

## Completion gate

Run the root retrieval-delivery gate plus local warm-up validation when the file is present.
