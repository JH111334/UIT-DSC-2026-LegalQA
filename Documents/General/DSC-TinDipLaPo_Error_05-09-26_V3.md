# Error V3 — Phân tích Lỗi Thực nghiệm Run 3 (e0-v3-r1)

- **Phân loại**: Diagnostic error taxonomy trên 700 cặp validation và 1.000 mẫu dự đoán public.
- **Run ID**: \e0-v3-r1- **Mô hình**: \Qwen/Qwen2.5-1.5B-Instruct\ + QLoRA Adapter E0.
- **Artifact đối chiếu**: \Data/Task2/runs/e0-v3-r1/error_table.jsonl\ và \public_output_audit.json\.

## 1. Các lỗi đã được giải quyết triệt để

1. **Vòng lặp suy thoái (\TOKEN_REPETITION_OR_LOOP\)**:
   - Run 2: 675 / 700 câu (**96,4%**) gặp hiện tượng lặp cụm từ liên tiếp hàng chục lần.
   - Run 3: **0 / 700 câu (0,0%)**. Việc thiết lập epetition_penalty = 1.20\ kết hợp o_repeat_ngram_size = 4\ đã xử lý dứt điểm khiếm khuyết giải mã này.
2. **Độ dài câu vượt mức kiểm soát (\EXCESSIVE_VERBOSITY\)**:
   - Run 2: 328 / 700 câu (**46,9%**).
   - Run 3: 67 / 700 câu (**9,5%**). Giới hạn \max_new_tokens = 512\ và cơ chế phạt lặp đã rút gọn câu trả lời về đúng dung lượng diễn đạt tự nhiên (trung vị p50 = 222 từ/câu).
3. **Hiện tượng ngắt câu giữa chừng (\ENDING_REVIEW\)**:
   - Run 2: 767 / 1.000 câu Public (**76,7%**) cạn ngân sách token dẫn đến câu trả lời bị đứt đoạn.
   - Run 3: giảm xuống 582 / 1.000 câu (**58,2%**). Đa số các câu còn lại đã có cấu trúc kết luận pháp lý hoàn chỉnh.

## 2. Các lỗi tồn đọng mang tính bản chất kỹ thuật

1. **Thiếu số hiệu văn bản pháp luật (\MISSING_REFERENCE_NUMBER\)**:
   - Tần suất: 698 / 700 câu (**99,7%**).
   - Bản chất: Mô hình sinh đơn thuần không được cung cấp văn bản trích dẫn. Dù mô hình nắm được khái niệm pháp lý tổng quát, việc yêu cầu tham số 1.5B ghi nhớ chính xác số hiệu từng Thông tư/Nghị định/Luật là điều bất khả thi về mặt lý thuyết thông tin.
2. **Ảo giác số hiệu điều luật (\EXTRA_OR_WRONG_NUMBER\)**:
   - Tần suất: 698 / 700 câu (**99,7%**).
   - Bản chất: Khi bị ép buộc phải trích dẫn căn cứ trong câu mở đầu (\Căn cứ theo Điều... Thông tư...\), mô hình tự suy đoán số hiệu văn bản không có thực hoặc không đúng phạm vi áp dụng.

## 3. Biện pháp kỹ thuật khắc phục

Lỗi thiếu căn cứ và ảo giác số hiệu điều luật không thể giải quyết bằng cách tăng epoch hay điều chỉnh siêu tham số sinh đơn thuần. Lời giải duy nhất là chuyển sang **nhánh E1 (Retrieval-Augmented Generation)**:
- Nạp Top-3/Top-5 đoạn trích văn bản pháp luật chính xác từ \	ask2-data-v3/indexes\ vào prompt đầu vào.
- Khi điều luật thực tế đã nằm trong context, mô hình chỉ thực hiện tác vụ đọc hiểu và tổng hợp (extractive summarization), qua đó loại bỏ 100% việc tự bịa số hiệu văn bản.
