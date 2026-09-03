# DSC-TinDipLaPo

Private team repository for UIT Data Science Challenge 2026:
Task 1 Legal Information Retrieval and Task 2 Legal Question Answering.

## Status

Implemented: deterministic Task 1/shared fixture; Task 2 canonical QA/corpus release;
E0 answer-only QLoRA checkpoint; request-driven local pipeline; submission validator;
và một Public candidate 1.000/1.000 có format PASS. Public candidate hiện thiếu lineage
đủ để gắn chắc với checkpoint E0, nên chỉ được gọi là format-valid external artifact.

Not implemented/measured: official scorer parity, full held-out METEOR/ROUGE-L, E1
Task2-only BM25 paired run, dense+BM25 RRF, neural reranking và điểm DSC chính thức.

| Branch | Scope |
|---|---|
| `Task1-LegalIR` | ingestion, retrieval, fusion, reranking |
| `Task2-LegalQA` | Task 2-only QA/corpus preprocessing, generation, retrieval experiment và scoring |
| `Data-Evaluation` | manifests, splits, qrels, evaluation |
| `Integration-Submission` | reproducibility and Codabench packaging |

Each task has separate `Data`, `Source`, `configs`, `tests`, `experiments`,
`submissions`, and `Documents` areas. Organizer data, model weights, indexes, runs,
and submission packages remain local-only.

## Baselines

- Task 1: BM25 + TF-IDF/RRF; first neural pair is
  `AITeamVN/Vietnamese_Embedding_v2` + `AITeamVN/Vietnamese_Reranker`.
- Task 2: E0 direct question-to-answer control; anchor là
  `Qwen/Qwen2.5-1.5B-Instruct`. E1 đề xuất BM25 trên `selected-contexts` Task 2,
  chỉ mở sau ADR/provenance gate; challenger chuyên ngành là
  `ntphuc149/ViLegalQwen3-1.7B-Base` sau supervised fine-tuning.
- Task 1 và Task 2 không dùng chéo data/context/checkpoint. Neural stages vẫn tắt
  cho tới khi có held-out evaluation trên profile RTX 2050 4 GB.

## Run

```powershell
$env:UV_CACHE_DIR = "$PWD\.tmp\uv-cache"
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

<!-- BEGIN competition-product-transfer:v1 -->

## Vai trò và chuyển giao

Repository này là competition project có deadline. Definition of done là scoreable
vertical slice, held-out evaluation, submission replay và postmortem; không phải product
hosting. Pattern bền chỉ được chuyển có chọn lọc sang `Multimodal-Asset-Retrieval`;
organizer data và submission logic ở lại đây.
<!-- END competition-product-transfer:v1 -->
