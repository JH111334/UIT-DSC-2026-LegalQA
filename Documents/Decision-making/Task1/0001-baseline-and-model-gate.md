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

Promotion gate: same split/corpus/config; Macro Recall không giảm và Macro Precision
cải thiện khi Recall bằng nhau; không ID/access leak; p95 latency và peak memory
chấp nhận được. Mỗi submission trả tối đa 5 unique document ID.

Task 1 train phải gắn cờ passage rỗng, nhóm passage trùng và chia split theo nhóm.
Toàn hệ thống nhỏ hơn 4 tỷ tham số và không gọi API.
