---
name: dsc-legal-qa-delivery
description: Triển khai Task 2 LegalQA độc lập với preprocessing, training, scoring và submission.
---

# Giao nhận Task 2

1. Đọc AGENTS.md, Workflows/flowinfo.md và ADR Task 2.
2. Đọc `Documents/Decision-making/BanGiao.md`; đóng băng QA release và
   corpus `selected-contexts` release Task 2 bằng schema, checksum và provenance.
   Trước experiment tốn GPU, đọc
   `Documents/Decision-making/Task2/COMPUTE-AWARE-EXPERIMENT-STRATEGY.md`.
   Một round có thể screening nhiều candidate, nhưng từng candidate chỉ đổi một
   scientific knob; full confirmation không được confound.
3. Reject Task 1 data/context/label/checkpoint, external data và augmentation.
4. E0 là answer-only SFT trực tiếp. Chỉ mở E1 BM25 Task2-only sau ADR/provenance
   gate; không dùng runtime retrieval Task 1. Dense-only chỉ là diagnostic ablation;
   challenger neural đầu tiên là BM25+dense qua RRF. Dense/reranker cần role Task 2
   được duyệt trong allowlist. Citation graph mặc định tắt và chỉ được xem xét sau
   khi hybrid hợp lệ cùng gate ADR-T2-0004 chứng minh blocker quan hệ.
5. Audit tokenizer cho `question + packed context + target`; từ chối model ngoài
   allowlist và tổng runtime từ 4B tham số trở lên.
6. Không model inference API. Pin model/tokenizer revision; chạy local hoặc private Kaggle
   batch do đội kiểm soát theo ADR-T2-0006. Kaggle API chỉ orchestration, payload tối thiểu
   phải có hash và inference phải chứng minh đã nạp adapter.
7. Đánh giá METEOR/ROUGE-L bằng scorer đã pin; nhãn provisional nếu chưa parity.
8. Giữ vòng nhỏ nhất chạy được làm control. Mỗi knob phải có nguồn
   `official|measured|paired-result|hypothesis`; chỉ sweep grid nhỏ gắn với
   bottleneck đã đo và đổi một biến chính mỗi comparison.
9. Entry point Task 2 là `Source/Task2/pipeline.py run --request`; code chia
   `Offline/{QA,Corpus}` và `Online/{Training,RetrievingAnswer}`. CLI chỉ giữ
   `contract` và `run --request`; không mở thêm cờ khi request JSON đã biểu diễn được.
   Tokenizer/index có thể do preprocessing owner hoặc consumer tạo, nhưng phải nằm
   ngoài immutable release nếu tạo sau freeze và phải pass gate trước stage tương ứng.
   Request JSON giữ toàn bộ path/stage/profile để CLI không phình và replay được.
   Retrieval metric chỉ là diagnostic và qrels suy citation phải ghi là proxy.
10. Báo truncation, repetition, failure, paired case/slice delta và counterexample.
11. Validate đủ ID, answer không rỗng và ZIP chỉ có submission.json.

Chạy gate project và warm-up validator khi file tồn tại. Log theo
Documents/General/RULE.md; tài liệu viết tiếng Việt có dấu, gọn.
