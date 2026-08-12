---
name: retrieval-delivery
description: Deliver scoped text retrieval and bounded-agent changes with citations and replay tests.
---

# Retrieval delivery

## Entry gate

1. Read `AGENTS.md` and `Workflows/flowinfo.md`.
2. Inspect Git status and preserve unrelated work.
3. State corpus/query contracts, access scope, evidence, validation, and abstention.
4. Test deterministic retrieval before adding an LLM or external index.
5. Read the relevant Task 1 or Task 2 skill and validate every model ID against
   `configs/Shared/model_allowlist.json`.

## Completion gate

Run:

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=text_retrieval_agent --cov-report=term-missing
uv run python skills/retrieval-delivery/scripts/validate_project.py
```

Generated prose without retrieved, authorized citations is not a valid answer.
