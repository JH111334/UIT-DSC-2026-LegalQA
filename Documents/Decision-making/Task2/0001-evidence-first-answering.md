# ADR-T2-0001: Evidence first, generator optional

Status: accepted.

Decision: answer from retrieved excerpts first. Compare
`Qwen/Qwen2.5-1.5B-Instruct` with `AITeamVN/Vi-Qwen2-1.5B-RAG` only after the Task 1
release is frozen. Generation remains disabled by default.

Reasons:

- Warm-up has 500 long-form answers but no source corpus or official scoring code.
- A 4 GB GPU favors sequential 1.5B experiments over concurrent 3B/4B models.
- Citation support and abstention remain testable when a model is unavailable.
- `ViLegalQwen*-Base` needs task-specific post-training; models with incomplete
  provenance enter only isolated experiments.

Promotion gate: no unsupported citation, no invented source ID, measurable held-out
gain under the organizer metric, bounded latency/memory, and extractive fallback on
timeout or invalid output.
