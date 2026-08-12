# Four workstreams

| Branch | Owner scope | Merge gate |
|---|---|---|
| `Task1-LegalIR` | ingestion, chunking, retrieval, reranking | retrieval metrics and ID integrity |
| `Task2-LegalQA` | answer contracts, evidence, abstention | citation support and failure tests |
| `Data-Evaluation` | manifests, splits, qrels, evaluation | leakage check and reproducibility |
| `Integration-Submission` | configs, packaging, submission checks | clean-environment end-to-end run |

All four branches start from and are synchronized to the validated `main` baseline.
Experimental divergence requires a focused commit and review before merge.
