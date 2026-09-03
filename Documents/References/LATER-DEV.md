# LATER DEV — phần hoàn thiện sau đường thi chính

Tệp này giữ những việc cần bổ sung để project đạt 100%, nhưng không được chen lên
trước preprocessing, first trained run, scorer parity và submission replay trong MAP.

## Làm ngay sau first run

- Thêm nhiều seed/fold cho 1–2 candidate tốt nhất.
- Hoàn thiện near-duplicate review và leakage dashboard tĩnh.
- So scorer local/Codabench bằng golden cases.
- Thêm resume checkpoint, OOM recovery và deterministic data loader.
- Tách train, evaluate, predict, package thành CLI có config versioned.
- Thêm test cho truncation, loss mask, missing ID, malformed output và ZIP contract.
- Lưu model card, data card, run manifest và bảng rejected experiments.

## Development để hoàn thiện đường thi

- Tối ưu batch/gradient accumulation, mixed precision và activation checkpointing
  bằng peak VRAM/throughput đo được.
- Grid nhỏ cho learning rate, LoRA rank, epoch, max tokens và decoding; không chạy
  grid lớn thiếu hypothesis.
- Checkpoint averaging chỉ khi tạo ra một model cuối duy nhất và BTC xác nhận cách
  tính hệ thống; không ensemble nhiều model làm vượt tổng tham số.
- Docker hoặc ZIP reproducible, offline smoke, download script có revision/hash.
- Script tạo submission và validator độc lập với notebook.
- Runbook cho public/private, rollback về last-known-good và deadline freeze.

## Task 1 để owner hoàn thiện

- Empty/duplicate audit có provenance.
- BM25 anchor, dense candidate, bounded reranker và fusion ablation.
- Fixed top-5 so adaptive-k dưới Macro Recall trước, Precision sau.
- Submission validator, latency, memory và error taxonomy theo loại query.

## Không quan trọng ở giai đoạn hiện tại

- RAG nối Task 1 sang Task 2.
- Hugging Face Inference Endpoint, API thương mại hoặc API phi lợi nhuận.
- Agent, MCP tool-calling, vector database, Redis, queue, microservice và UI.
- Multi-tenant auth, observability platform, Kubernetes hoặc distributed training.
- Dữ liệu synthetic, external corpus, web search enrichment hoặc pseudo-label từ
  nguồn ngoài.
- Architecture tổng quát chỉ để trông giống production.

Các mục trên chỉ mở nếu luật BTC thay đổi bằng văn bản hoặc blocker đo được buộc
phải có. Phần product/system architecture thuộc Multimodal-Asset-Retrieval, không
phải điểm mạnh cần thể hiện ở competition này.

## Mức hoàn thiện 100%

Simple hoàn tất khi một raw fixture đi qua validate → preprocess → train smoke →
predict → score → package mà không thao tác tay.

Development hoàn tất khi organizer train đi qua cùng pipeline, có experiment ledger,
paired comparison, failure recovery và clean-environment replay.

Competition hoàn tất khi candidate cuối có bằng chứng local/public, package đúng
contract, tài liệu tái lập đầy đủ và private submission không vi phạm rule. “100%”
không đồng nghĩa production SaaS; đó là một competition system đầy đủ và kiểm chứng được.

