# Quy ước dự án DSC-TinDipLaPo

## Định danh

Đây là workspace private của nhóm UIT DSC 2026 cho Task 1 LegalIR và Task 2 LegalQA.
Task 1 trả tối đa 5 document ID có provenance. Task 2 độc lập, sinh answer từ dữ
liệu Task 2 và được chấm bằng METEOR/ROUGE-L. Runtime deterministic BM25/TF-IDF/RRF/
extractive hiện có chỉ là Task 1/shared fixture, không phải phương pháp Task 2.

Không tuyên bố dense retrieval, reranking, chất lượng sinh, production hoặc điểm
DSC chính thức trước khi được triển khai và đánh giá theo quy định BTC.
Không bịa query, document ID, citation, access right, permission, luật, score hoặc
answer không được evidence hỗ trợ.

## Workflow bắt buộc

Đọc Workflows/flowinfo.md, skills/retrieval-delivery/SKILL.md và skill đúng task
trước thay đổi kiến trúc, data, indexing, retrieval, agent, evaluation, API hoặc
UI.

Task 1: manifest → deterministic units → retrievers → fusion/rerank → document IDs.

Task 2: raw Task 2 → QA release + corpus `selected-contexts` Task 2 release → E0
answer-only SFT trực tiếp; sau ADR/provenance gate mới mở E1 BM25 Task2-only →
deterministic answer → METEOR/ROUGE-L → submission.

## Quy tắc kỹ thuật

- Task 1 giữ query/document/article ID, provenance, rank và component score.
- Task 2 giữ question ID, raw/split/model/config hash và answer.
- Không dùng data, context, label hoặc trained checkpoint của task này cho task kia.
- Runtime BM25/TF-IDF/RRF hiện có không được dùng làm Task 2 retrieval. Nhánh E1
  phải có code, corpus, index, config và trace Task2-only; citation-derived qrels
  chỉ là diagnostic proxy, không phải gold hoặc permission để train retriever.
- Chỉ dùng model trong configs/Shared/model_allowlist.json; quy định BTC ưu tiên.
- Tách Task 1/Task 2 về data, code, config, test, experiment, submission và docs.
- Giữ organizer data, run, submission, corpus, index, model, cache, trace và secret ngoài Git.
- Đánh giá đúng metric từng task, output integrity, latency, memory và failure behavior.
- Không dùng emoji, sticker, banner ký hiệu hoặc thông báo ăn mừng trong mã, test, log hay CLI; chỉ giữ ký hiệu có nghĩa kỹ thuật hoặc dữ liệu protocol đã ghi rõ.

## Data và inference gate

- Với organizer data, khóa quyền truy cập, split, schema, checksum và submission contract trước khi đổi data/model.
- Chỉ dùng dữ liệu BTC; không external data hoặc augmentation.
- BTC cấm model inference API trong phương pháp. Hugging Face chỉ dùng tải weights hợp lệ.
  Theo xác nhận lại của operator với BTC ngày 03-09-2026, private Kaggle kernel do đội kiểm
  soát được dùng cho Task 2 train/batch inference; Kaggle API chỉ làm orchestration.
- Tổng tham số mọi thành phần đang chạy trong từng task phải nhỏ hơn 4 tỷ.
- Chỉ upload payload Task 2 tối thiểu, có hash, vào private Kaggle dataset/kernel theo
  ADR-T2-0006. Không gửi Task 1, Private labels, secret hoặc artifact ngoài request đã duyệt;
  không dùng Hugging Face Inference API.

## Học từ kết quả và tài liệu

Đọc Documents/General/RULE.md trước comparison. Khóa anchor/parent, hypothesis,
evaluation role, gold status, evaluation contract, paired case/slice delta và
counterexample. Run hợp lệ nhưng kém hơn là Checkpoint REJECT|REPEAT; Error chỉ
dành cho invalid run hoặc quality gate FAIL.

Tài liệu do project sở hữu viết tiếng Việt có dấu, viết hoa đầu câu, gọn và có
nguồn gần mệnh đề. Không chép debug narrative hoặc dùng scaffold/smoke làm kết quả.

## Hoàn thành

Fixture và Task 2 preprocess/split/training-bundle/submission chạy từ môi trường
sạch; cross-task artifact bị reject; test, Ruff, mypy, workflow validator và lockfile
check đạt. Validator canonical là
`uv run python skills/retrieval-delivery/scripts/validate_project.py`; tài liệu
tách implemented khỏi roadmap.
