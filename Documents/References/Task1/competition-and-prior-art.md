# Bản đồ bằng chứng Task 1

UIT DSC định danh Task 1 là Legal Information Retrieval trên Codabench. Contract
đã xác minh: tối đa 5 document ID, Macro Recall chính và Macro Precision tie-break.
Thông báo BTC mới do người dùng cung cấp được ưu tiên cho data/model/API.

| Nguồn | Phát hiện | Cách dùng |
|---|---|---|
| [ALQAC 2024](https://sites.google.com/view/alqac-2024/home) | Legal retrieval tiếng Việt tương tự dùng relevance cấp điều và F2 | Chỉ diagnostic local, không phải luật UIT |
| [NOWJ1 2023](https://arxiv.org/abs/2309.09070) | Lexical + neural, cấu trúc điều/khoản và learning-to-rank | Giữ cấu trúc; hoãn learned fusion |
| [NOWJ 2024](https://doi.org/10.1109/KSE63888.2024.11063594) | Winning system kết hợp ranking signal | Benchmark fusion/rerank theo stage |
| [ViDRILL](https://aclanthology.org/2025.vlsp-1.17/) | BM25 + dense + cross-encoder; chunk/hard negative quan trọng | Multi-stage bounded sau baseline |
| [ViRE](https://aclanthology.org/2026.findings-eacl.110/) | Embedding tiếng Việt + lexical mạnh trên nhiều domain | Candidate hybrid, cần held-out |
| [ALQAC code](https://github.com/baohl00/alqac24) | Implementation retrieval/QA công khai | Tham khảo adapter, không chép post-process |
| [ViRE code](https://github.com/longstnguyen/ViRE) | Evaluation code cho IR tiếng Việt | Tham khảo protocol |

Quyết định: audit passage rỗng/trùng rồi lexical deterministic trước. Neural pair đầu là
`AITeamVN/Vietnamese_Embedding_v2` + `AITeamVN/Vietnamese_Reranker`; BGE-M3,
E5, Qwen3 và VietLegal-Harrier chỉ là challenger. Tất cả vẫn bị chặn bởi qrels,
hard-negative analysis, resource probe và cùng evaluation contract; chưa model nào
được bật hoặc có điểm DSC. ALQAC F2 chỉ là prior-art, không thay metric UIT.

Xem organizer-clarification-summary.md cho contract và trade-off Recall/Precision.
