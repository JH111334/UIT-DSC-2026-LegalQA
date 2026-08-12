# Task 2 — LegalQA

## Contract

Input: organizer query ID and legal question. Internal output: answer or abstention,
plus ordered evidence chunk IDs from the active corpus. Submission serialization
remains pending until BTC publishes an authoritative schema.

## Pipeline

```text
question -> Task 1 retrieval -> bounded evidence set
  -> extractive answer baseline
  -> optional allowlisted generator
  -> citation/support validator
  -> answer or abstention
```

The generator cannot add document IDs, sources, or legal claims outside retrieved
evidence. Retrieval failure, adapter failure, timeout, or invalid output falls back
to extractive evidence or abstention.

## Gate

- Evaluate with the same retrieval release used for the answer.
- Record citation support, unsupported claims, abstention, latency, memory, and failures.
- Treat warm-up answers as historical organizer references, not current legal advice.
- Keep human or organizer scoring separate from local diagnostics.
