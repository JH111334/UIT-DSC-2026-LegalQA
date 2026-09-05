# Checkpoint V3 — Kết quả Thực nghiệm Run 3 (e0-v3-r1) và Triệt tiêu Lặp Từ

## Đã chứng minh

- **Lineage và provenance hoàn chỉnh**: Run e0-v3-r1 thực thi thành công trên private Kaggle GPU (Tesla T4, 16.990 giây ~ 4,72 giờ), sử dụng đúng gói dữ liệu task2-data-v3 (release SHA-256 d6ca3086..., payload SHA-256 d2e4f874...) và mã nguồn runtime src/legalqa pin cố định.
- **Loại bỏ hoàn toàn lỗi lặp suy thoái**: Tham số giải mã repetition_penalty = 1.20 và no_repeat_ngram_size = 4 đã triệt tiêu toàn bộ vòng lặp suy thoái (TOKEN_REPETITION_OR_LOOP giảm từ 96,4% ở Run 2 xuống 0,0% ở Run 3).
- **Kiểm soát độ dài câu trả lời**: Tỷ lệ câu trả lời quá dài (EXCESSIVE_VERBOSITY) giảm từ 46,9% xuống 9,5% (67/700 câu trên tập validation). Tỷ lệ lỗi ngắt cụt câu do chạm trần token (ENDING_REVIEW) trên tập Public giảm từ 76,7% xuống 58,2%.
- **Chỉ số thực nghiệm trên 700 cặp validation**:
  - METEOR: **0.2344**
  - ROUGE-L: **0.1700**
  - Tỷ lệ câu rỗng (empty_rate): **0,0%**
  - Phân tầng câu ngắn (answer_short, 166 câu): METEOR **0.3156**, ROUGE-L **0.2009**
  - Phân tầng câu định nghĩa (question_definition, 33 câu): METEOR **0.2732**, ROUGE-L **0.1924**
- **Độ tin cậy của file nộp bài (Submission)**: Artifact submissions/Task2/e0-v3-r1/submission.zip (SHA-256 b21835579d99b232c797a4aad3b842d2d679f2c8c621add1cc03bd3438a09d39) vượt qua toàn bộ kiểm tra replay nội bộ: đúng 1.000 question ID của tập Public, không câu rỗng, 998/1.000 câu có dấu hiệu cấu trúc pháp lý.

## Chưa chứng minh

- Khả năng sinh chính xác số hiệu văn bản quy phạm pháp luật (Điều, Khoản, Thông tư, Nghị định) khi không có ngữ cảnh văn bản được truy xuất: MISSING_REFERENCE_NUMBER vẫn chiếm 698/700 câu (99,7%) và EXTRA_OR_WRONG_NUMBER chiếm 698/700 câu (99,7%).
- Điểm số METEOR / ROUGE-L trên 0.40 đối với nhánh sinh đơn thuần E0: mô hình 1.5B đã chạm trần dung lượng ghi nhớ trọng số đối với tri thức pháp luật chi tiết.

## Quyết định

1. Khóa checkpoint adapter E0 (e0-v3-r1) làm baseline sạch không còn lỗi lặp từ và định dạng chuẩn.
2. Xác nhận giả thuyết cốt lõi của TASK3: việc nâng điểm từ 0.23 lên 0.50+ bắt buộc phải dựa vào cơ chế RAG (nhánh E1) để nạp văn bản luật thực tế vào ngữ cảnh đầu vào, thay vì tiếp tục tinh chỉnh sinh mù (direct generation).
3. Chuyển trọng tâm kỹ thuật sang tích hợp bộ truy xuất BM25/FTS5 từ task2-data-v3/indexes vào pipeline suy luận và định dạng prompt có ngữ cảnh cho nhánh E1.
