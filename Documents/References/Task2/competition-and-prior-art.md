# Bản đồ bằng chứng Task 2

UIT DSC định danh Task 2 là Legal Question Answering trên Codabench. Contract đã
xác minh: answer tiếng Việt, METEOR chính và ROUGE-L phụ. Thông báo BTC mới cấm
dùng dữ liệu Task 1, external data, augmentation và API.

| Nguồn | Phát hiện | Cách dùng |
|---|---|---|
| [ALQAC 2024](https://sites.google.com/view/alqac-2024/home) | QA tương tự có extractive, yes/no và multiple choice | Phân tích type sau official schema |
| [METEOR](https://aclanthology.org/W05-0909/) | Alignment có precision/recall và fragmentation | Phân tích verbosity/coverage theo case |
| [ROUGE](https://aclanthology.org/W04-1013/) | ROUGE-L dùng longest common subsequence | Giữ thứ tự và cấu trúc target để ablation |
| [NeCo 2023](https://arxiv.org/abs/2309.05500) | Extraction theo question type | Giữ extractive path |
| [NOWJ1 2023](https://arxiv.org/abs/2309.09070) | Tách extraction khỏi classification | Không dùng một prompt cho mọi type |
| [ViLQA](https://github.com/ntphuc149/ViLQA) | So span extraction và generation | Benchmark khi corpus hợp lệ |
| [ALQAC code](https://github.com/baohl00/alqac24) | Retrieval + LLM công khai | Chỉ tham khảo implementation |
| [ViLegalLM](https://aclanthology.org/2026.findings-acl.1801/) | Base model tiếng Việt pháp luật 1.5B/1.7B | Challenger SFT, không nhập corpus/synthetic data |
| [ViT5](https://aclanthology.org/2022.naacl-srw.18/) | Vietnamese text-to-text | Challenger encoder-decoder |
| [QLoRA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html) | Fine-tune tiết kiệm bộ nhớ | Không thay luật đếm tham số |
| [VLSP 2025 MLQA-TSR](https://aclanthology.org/2025.vlsp-1.48/) | Retrieval đa phương thức có relevant-article labels, F2 và top-5 baseline | Chỉ học pattern retrieval; không chuyển metric/labels sang DSC |
| [VLSP official baseline](https://github.com/sonlam1102/VLSP2025-MLQA-TSR) | Ví dụ corpus encoding và retrieval baseline | Prior art; không nhập code/data khi chưa audit license/fit |
| [ViDRILL](https://aclanthology.org/2025.vlsp-1.17/) | Legal retrieval tiếng Việt dùng BM25, dense retrieval và reranking nhiều tầng | Cơ sở ưu tiên hybrid BM25+dense; vẫn phải ablate trên DSC |
| [COLIEE 2025 overview](https://link.springer.com/article/10.1007/s12626-026-00199-9) | Tám đội Task 1 dùng biến thể multi-stage legal retrieval | Hỗ trợ system-level tuning; không chuyển top-k/threshold/metric sang DSC |
| [Graph-based MLQA-TSR](https://aclanthology.org/2025.vlsp-1.49/) | Graph dị thể nối text, image và table trong traffic-law multimodal retrieval | Graph chỉ là optional hypothesis vì modality/corpus/metric khác DSC |

Quyết định: `E0` là direct supervised generation với
`Qwen/Qwen2.5-1.5B-Instruct`; `ntphuc149/ViLegalQwen3-1.7B-Base` là legal challenger
sau SFT. `E1` đề xuất BM25 chỉ trên corpus chính thức Task 2 rồi dùng cùng generator
E0. Không nhận Task 1 retrieval. Dense-only là diagnostic; neural challenger chính
là BM25+dense/RRF. Citation graph optional disabled cho tới khi hybrid và relation
failure gate có paired end-to-end evidence. Xem
organizer-contract-and-data-audit.md, Decision-making/Task2/EXTENSION.md và
Decision-making/Task2/0003-task2-official-corpus-retrieval.md cùng ADR-T2-0004.
