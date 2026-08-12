# Task 1 evidence map

UIT DSC identity: Legal Information Retrieval on the dedicated
[Codabench competition](https://www.codabench.org/competitions/17715/). The local
warm-up file and Drive URL came from the organizer message supplied by the user;
the public page does not expose the full scoring/submission specification.

| Evidence | Finding | Use |
|---|---|---|
| [ALQAC 2024 task page](https://sites.google.com/view/alqac-2024/home) | Analogous Vietnamese legal retrieval uses article-level relevance and F2 | Local diagnostic only; not an UIT rule |
| [NOWJ1 at ALQAC 2023](https://arxiv.org/abs/2309.09070) | Lexical, neural features, article/clause preprocessing, learning-to-rank | Preserve structured units; defer learned fusion |
| [NOWJ at ALQAC 2024](https://doi.org/10.1109/KSE63888.2024.11063594) | Winning system combines document-ranking signals and learned ranking | Benchmark staged fusion/reranking |
| [ViDRILL](https://aclanthology.org/2025.vlsp-1.17/) | BM25 + dense retrieval + cross-encoder; chunking and hard negatives matter | Adopt bounded multi-stage benchmark |
| [Vietnamese IR study](https://aclanthology.org/2026.findings-eacl.110/) | Legal-domain results favor Vietnamese embedding plus lexical retrieval | Prioritize Vietnamese_Embedding_v2 hybrid |
| [se7enese implementation](https://github.com/baohl00/alqac24) | Reproducible ALQAC retrieval/QA code | Inspect adapters; do not copy manual post-processing |
| [ViRE implementation](https://github.com/longstnguyen/ViRE) | Multi-domain Vietnamese IR evaluation code | Reference evaluation protocol |

Decision: deterministic lexical baseline first; Vietnamese_Embedding_v2 and
Vietnamese_Reranker first neural pair; BGE-M3, E5, Qwen3, and VietLegal-Harrier are
challengers. No external result is a DSC score.
