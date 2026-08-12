# DSC-TinDipLaPo

Private team repository for UIT Data Science Challenge 2026:
Task 1 Legal Information Retrieval and Task 2 Legal Question Answering.

## Status

Implemented: deterministic chunks, access-scoped BM25 and TF-IDF, RRF, extractive
answers, citation validation, abstention, warm-up schema validation, and smoke tests.

Not implemented: dense retrieval, neural reranking, generative answering, official
submission adapters, and competition-quality evaluation. No DSC score is claimed.

| Branch | Scope |
|---|---|
| `Task1-LegalIR` | ingestion, retrieval, fusion, reranking |
| `Task2-LegalQA` | evidence-bound answers and abstention |
| `Data-Evaluation` | manifests, splits, qrels, evaluation |
| `Integration-Submission` | reproducibility and Codabench packaging |

Each task has separate `Data`, `Source`, `configs`, `tests`, `experiments`,
`submissions`, and `Documents` areas. Organizer data, model weights, indexes, runs,
and submission packages remain local-only.

## Baselines

- Task 1: BM25 + TF-IDF/RRF; first neural pair is
  `AITeamVN/Vietnamese_Embedding_v2` + `AITeamVN/Vietnamese_Reranker`.
- Task 2: extractive evidence; first generator comparison is
  `Qwen/Qwen2.5-1.5B-Instruct` versus `AITeamVN/Vi-Qwen2-1.5B-RAG`.
- Neural stages are disabled until held-out evaluation passes on the RTX 2050 4 GB profile.

## Run

```powershell
uv sync --locked
uv run dsc search --text "Cách quay lại phiên bản ổn định khi triển khai lỗi?"
uv run dsc ask --text "Why must the agent cite retrieved chunks and abstain?"
uv run dsc evaluate --top-k 3
uv run dsc-warmup Task1 Data/Task1/warmup_data/warmup_Task1.json
```

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=text_retrieval_agent --cov-report=term-missing
uv run python skills/retrieval-delivery/scripts/validate_project.py
```

MIT License.
