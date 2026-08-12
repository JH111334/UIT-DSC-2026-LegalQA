# AGENTS

## Project identity

`DSC-TinDipLaPo` is the private team workspace for UIT DSC 2026 Task 1 LegalIR and
Task 2 LegalQA. It may retrieve bounded legal evidence, answer with cited chunks,
or abstain. It may not invent document IDs, citations, permissions, laws, or scores.

The runtime is a deterministic smoke baseline: BM25, TF-IDF, RRF, scope filtering,
citation validation, and extractive fallback. Dense retrieval, reranking, generation,
and official DSC quality remain unimplemented until adapters and evaluation exist.

## Mandatory workflow

Read `Workflows/flowinfo.md`, `skills/retrieval-delivery/SKILL.md`, and the relevant
task skill before non-trivial retrieval, QA, evaluation, API, or submission work.

## Technical rules

- Preserve query, document, chunk, source, access, corpus version, score, and rank.
- Enforce access scope before candidates and before answer assembly.
- Keep lexical retrieval and extractive fallback operational without models.
- Cite only retrieved chunks from the active corpus version; otherwise abstain.
- Use only IDs in `configs/Shared/model_allowlist.json`; BTC rules override it.
- Keep organizer data, indexes, weights, runs, submissions, traces, and secrets out of Git.
- Keep Task 1 and Task 2 data, source, configs, tests, experiments, submissions, and docs split.
- Keep documents concise; omit greetings, filler, repeated background, and debug logs.

## Definition of done

- Clean-environment fixture ingestion, retrieval, answer, and evaluation run.
- Unauthorized documents cannot enter candidates or citations.
- Tests, Ruff, mypy, validator, and lockfile check pass.
- Implemented, measured, prior-art, decision, and roadmap claims remain distinct.
