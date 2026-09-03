# Register dữ liệu và model Task 2

## Data contract

Raw organizer-style JSON là object keyed by question_id; mỗi value có question và
answer là chuỗi không rỗng. Prepared JSONL thêm source_task=Task2, question_group,
source_sha256 và transform_version. Raw luôn bất biến và ở ngoài Git.

Pipeline không nhận evidence/chunk/corpus release từ Task 1. Corpus release hợp lệ
chỉ được sinh từ `selected-contexts` có provenance Task 2 và phải tuân theo root
`Documents/Decision-making/BanGiao.md`. Fixture
examples/fixture_task2.json chỉ kiểm code.

## Model ladder

| Candidate | Vai trò | Gate |
|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | Anchor first-run | pin revision/license; SFT Task 2 |
| ntphuc149/ViLegalQwen3-1.7B-Base | Legal challenger | base model; bắt buộc post-train |
| ntphuc149/ViLegalQwen2.5-1.5B-Base | Legal ablation | base model; paired comparison |
| VietAI/vit5-base | Encoder-decoder challenger | tokenizer/target-length audit |
| VietAI/vit5-large | Scale ablation | chỉ khi VRAM/runtime cho phép |

Sparse retrieval E1 dùng BM25 Task2-only, không có neural parameter. Dense
retriever/reranker chưa được chọn và không được tự lấy model mang role Task 1.

Allowlisted không đồng nghĩa được chọn. Canonical parameter audit chặn model từ 4B
trở lên và hold model chưa xác minh. Chạy một model mỗi lần; không runtime ensemble
làm vượt tổng parameter budget.

## Artifact mỗi run

Raw/split checksum, tokenizer/model revision, plan/config/seed, SFT dataset hash,
checkpoint/adapter, predictions, METEOR/ROUGE-L, case/slice delta, VRAM/latency,
failure log và submission hash.
