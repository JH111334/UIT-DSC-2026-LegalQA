# ADR-T1-0001: Sparse baseline before neural retrieval

Status: accepted.

Decision: retain BM25 + TF-IDF/RRF as the always-available baseline. First neural
experiment adds `AITeamVN/Vietnamese_Embedding_v2`; first reranking experiment uses
`AITeamVN/Vietnamese_Reranker` over bounded candidates. Run models sequentially.

Reasons:

- Warm-up contains 500 labeled queries but no legal corpus in this repository.
- Measured hardware: RTX 2050 4 GB, RAM 16 GB.
- Vietnamese legal prior art favors hybrid retrieval and reranking, but transfer to
  UIT DSC is unproven.

Promotion gate: same split/corpus/config; positive Macro-F2 and recall delta; no ID
or access leak; acceptable p95 latency and peak memory. Official BTC metric overrides
the provisional local diagnostic.
