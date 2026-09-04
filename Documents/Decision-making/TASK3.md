# TASK3 — Kế hoạch Cải tiến Toàn diện Lần 3 và Chuyển dịch E0 sang E1 (RAG)

- **Trạng thái**: PLANNING_ITERATION_3; ROADMAP_E1_RETRIEVAL_AUGMENTED.
- **Ngày lập**: 04-09-2026.
- **Mục tiêu**: Nâng điểm từ baseline E0 (METEOR 0.22, ROUGE-L 0.32) lên nhóm dẫn đầu (0.50+ ở cả hai chỉ số).
- **Hợp đồng bàn giao**: [BanGiao.md](BanGiao.md) (Cập nhật cho release 	ask2-data-v3).
- **Chiến lược tính toán**: [COMPUTE-AWARE-EXPERIMENT-STRATEGY.md](Task2/COMPUTE-AWARE-EXPERIMENT-STRATEGY.md).

---

## 1. Phân Tích Thực Nghiệm Lần 2 (E0 Baseline Audit)

Lần chạy thứ hai (E0 Kaggle v2-r1, Qwen2.5-1.5B-Instruct QLoRA, answer-only SFT) đã hoàn thành toàn bộ 6.300 câu huấn luyện, 700 câu kiểm định và 1.000 câu dự đoán public. Kết quả ghi nhận như sau:

### 1.1. Bảng Chỉ Số Thực Nghiệm

| Tập dữ liệu / Metric | METEOR | ROUGE-L | Đánh giá |
|---|---|---|---|
| **Validation 700 câu (Local / Kaggle)** | 0.2659 | 0.2090 | Baseline nội bộ có nhãn tham chiếu |
| **Public Test 1.000 câu (Ban Tổ Chức)** | **~0.22** | **~0.32** | Điểm chính thức ghi nhận từ leaderboard |
| **Nhóm Dẫn Đầu Leaderboard** | **0.50+** | **0.50+** | Khoảng cách cần thu hẹp: +0.28 METEOR, +0.18 ROUGE-L |

### 1.2. Taxonomy Lỗi từ 700 Cặp Kiểm Định và 1.000 Dự Đoán Public

| Mã lỗi | Số lượng | Tỷ lệ | Phân tích bản chất kỹ thuật |
|---|---|---|---|
| TOKEN_REPETITION | 675 / 700 | **96.4%** | Vòng lặp suy thoái (degenerate loop): mô hình lặp lại một mệnh đề pháp lý hàng chục lần do tham số phạt lặp chưa kích hoạt (
epetition_penalty = 1.0). |
| REPEATED_4GRAM_REVIEW | 992 / 1.000 | **99.2%** | Trên tập Public, 99.2% câu trả lời xuất hiện 4-gram lặp lại liên tiếp, phá hủy độ mượt và ngữ pháp câu. |
| MISSING_REFERENCE_NUMBER | 696 / 700 | **99.4%** | Thiếu hoàn toàn số điều luật trích dẫn chính xác (Điều X, Khoản Y, Nghị định Z) do mô hình 1.5B không thể nhớ thuộc lòng toàn bộ hệ thống văn bản quy phạm pháp luật khi không có truy xuất ngữ cảnh. |
| EXTRA_OR_WRONG_NUMBER | 698 / 700 | **99.7%** | Ảo giác (hallucination): mô hình tự bịa số hiệu thông tư, nghị định không có thực trong bối cảnh câu hỏi. |
| EXCESSIVE_VERBOSITY | 328 / 700 | **46.9%** | Câu trả lời dài quá mức cần thiết, chứa nhiều đoạn dẫn nhập sáo rỗng. |
| ENDING_REVIEW | 767 / 1.000 | **76.7%** | Cạn kiệt ngân sách token: do lặp từ liên tục, 76.7% câu trả lời chạm trần max_new_tokens = 768 và bị ngắt cụt giữa câu, không có kết luận. |

### 1.3. Phân Tích Phân Lớp (Slice Breakdown)
- nswer_short (160 câu): METEOR **0.3239** — mô hình đạt điểm tương đối khi câu hỏi có phạm vi hẹp.
- nswer_long (107 câu): METEOR **0.2129** — điểm yếu nhất hệ thống; văn bản càng dài thì nguy cơ rơi vào vòng lặp suy thoái càng cao.
- legal_structure_no (23 câu): METEOR **0.1998**, ROUGE-L **0.1432** — khi câu trả lời mất cấu trúc phân nhánh pháp lý, điểm số giảm sút nghiêm trọng.

---

## 2. Nguyên Nhân Gốc Rễ (Root Cause Analysis)

1. **Sinh Đơn Thuần Không Truy Xuất (Direct Generation vs RAG)**:
   - E0 huấn luyện theo dạng Câu hỏi -> Câu trả lời. Một mô hình 1.5 tỷ tham số không thể lưu trữ hàng triệu điều khoản luật Việt Nam trong trọng số.
   - Nhóm đạt điểm 0.50+ chắc chắn sử dụng mô hình truy xuất (Retrieval-Augmented Generation): trích xuất đúng Điều/Khoản từ selected-contexts, sau đó đưa đoạn trích vào ngữ cảnh (prompt context). Mô hình chỉ cần trích lọc và tổng hợp, loại bỏ hoàn toàn việc bịa số hiệu điều luật.
2. **Thiếu Bộ Lọc Cấu Hình Giải Mã (Decoding Configuration)**:
   - Sử dụng greedy decoding mặc định không có 
epetition_penalty và không có 
o_repeat_ngram_size dẫn đến hiện tượng lặp lại 20-30 lần một cụm từ.
3. **Nhiễu Thu Thập Trong Dữ Liệu Huấn Luyện (Crawled Web Boilerplate)**:
   - Dữ liệu thô chứa nhiều đoạn rác từ trang web pháp luật: (Hình từ Internet), câu hỏi liên quan ở cuối bài (Du học sinh... có cần chứng chỉ ngoại ngữ không?), chuỗi chấm lửng …….. Mô hình học cả các đoạn văn rác này vào câu trả lời.

---

## 3. Kế Hoạch Điều Chỉnh và Cải Tiến Lần 3

Thiết kế tinh gọn, vừa đủ, không làm xáo trộn kiến trúc và đảm bảo từng can thiệp đều có kiểm chứng ablation độc lập:

`	ext
               PHASE A (Đồng đội - Khang)
  Raw QA + Selected-contexts
    │
    ├── [A1] Data Sanitation: lọc sạch boilerplate '(Hình từ Internet)', trailing Q&A, '…….'
    ├── [A2] Legal Chunking: chuẩn hóa 316k chunks có metadata doc_id, article_id, clause_id
    └── [A3] Handoff task2-data-v3 + SQLite FTS5 / BM25 index bundle
                             │
                             ▼
               PHASE B (Tác giả - TinDipLaPo)
  Preflight Acceptance task2-data-v3
    │
    ├── [B1] Retrieval Routing: BM25 truy xuất Top-3/Top-5 chunks từ task2-data-v3
    ├── [B2] Prompt Engineering: Ngữ cảnh văn bản luật + Câu hỏi -> Trả lời căn cứ
    ├── [B3] Decoding Knobs: repetition_penalty = 1.20, no_repeat_ngram = 4, max_tokens = 512
    └── [B4] Training Conditioning: SFT QLoRA trên bộ (Context, Question) -> Answer
`

### 3.1. Nhiệm Vụ Phase A — Tiền Xử Lý và Corpus (Đồng Đội: Khang)

1. **[A1] Nâng cấp Module Làm Sạch Boilerplate (sanitation.py)**:
   - Thêm quy tắc regex loại bỏ triệt để các đuôi câu hỏi liên đới: (?i)(Hình từ Internet|Ảnh minh họa).*$.
   - Cắt bỏ các câu hỏi thừa sau dấu chấm lửng ……. ở cuối câu trả lời tham chiếu.
   - Chuẩn hóa ký tự khoảng trắng thừa, giữ lại toàn vẹn nội dung pháp lý cốt lõi.
2. **[A2] Đóng gói BM25 Index Bundle**:
   - Xây dựng sẵn cơ sở dữ liệu tìm kiếm toàn văn SQLite FTS5 hoặc BM25 chỉ mục trên tập corpus/chunks.jsonl thuộc 	ask2-data-v3.
   - Đảm bảo mỗi chunk có định danh rõ ràng: doc_id, rticle_id, nội dung đoạn trích.
3. **[A3] Phát hành 	ask2-data-v3**:
   - Đóng gói đúng 4 bundle theo hợp đồng BanGiao.md: Release Control, QA Payload, Corpus Payload, Index Bundle.

### 3.2. Nhiệm Vụ Phase B — Truy Xuất, Huấn Luyện và Suy Luận (Tác Giả & AI Agent)

1. **[B1] Khóa Lỗi Lặp Từ (Decoding Gate — Không Cần Huấn Luyện Lại)**:
   - Thêm cấu hình giải mã:
     - 
epetition_penalty = 1.20
     - 
o_repeat_ngram_size = 4
     - max_new_tokens = 512 (giảm từ 768 để tránh lan man và tiết kiệm thời gian)
     - 	emperature = 0.0 (greedy)
2. **[B2] Xây dựng Cơ chế RAG Task 2 (Nhánh E1)**:
   - Với mỗi câu hỏi, truy xuất Top-3 đoạn văn bản pháp luật có điểm BM25 cao nhất từ corpus Task 2.
   - Định dạng prompt mới:
     `	ext
     <|im_start|>system
     Bạn là chuyên gia pháp luật Việt Nam. Hãy căn cứ chính xác vào các văn bản pháp lý được cung cấp dưới đây để trả lời câu hỏi. Trả lời đúng trọng tâm, trích dẫn rõ ràng điều khoản và không lặp lại từ ngữ.<|im_end|>
     <|im_start|>user
     Văn bản pháp lý liên quan:
     {context_chunks}

     Câu hỏi: {question}<|im_end|>
     <|im_start|>assistant
     `
3. **[B3] Huấn Luyện QLoRA E1 trên Dữ Liệu Ngữ Cảnh**:
   - Ghép ngữ cảnh truy xuất vào tập 6.300 câu huấn luyện của 	ask2-data-v3.
   - Huấn luyện 1 epoch SFT trên GPU T4 của Kaggle batch (thời gian dự kiến: ~4-5 giờ, tiết kiệm hơn E0 nhờ độ dài output ngắn gọn hơn).

---

## 4. Kế Hoạch Thực Nghiệm và Ablation Từng Bước (Ablation Protocol)

Để tiết kiệm quota GPU (30 giờ/tuần) và kiểm chứng chính xác tác động của từng can thiệp:

| Bước | Tên thử nghiệm | Mục tiêu kiểm chứng | Môi trường | Chi phí compute |
|---|---|---|---|---|
| **C0** | *Decoding Check* | Đo lường mức giảm TOKEN_REPETITION trên 50 câu validation | Local CPU | 0 GPU |
| **C1** | *BM25 Retrieval Check* | Đo lường Recall@5 của bộ truy xuất BM25 trên 700 câu validation | Local CPU | 0 GPU |
| **Step 1** | *E0 Adapter + RAG + Decoding* | Nạp adapter cũ của E0, kết hợp ngữ cảnh BM25 và 
epetition_penalty=1.20 trên 700 câu | Kaggle Batch | ~1.0 giờ GPU |
| **Step 2** | *E1 Full SFT + Submission* | Huấn luyện adapter mới với dữ liệu có ngữ cảnh, sinh 1.000 câu Public, xuất zip nộp bài | Kaggle Batch | ~4.5 giờ GPU |

Nếu Step 1 đạt METEOR > 0.40 và ROUGE-L > 0.40, tiến hành Step 2 ngay để tối ưu hóa khả năng bám sát ngữ cảnh của mô hình.

---

## 5. Bàn Giao và Trách Nhiệm Tiếp Theo

- **Phase A (Khang)**:
  1. Kéo commit mới nhất từ repository về (git pull).
  2. Thực thi script làm sạch boilerplate trong Source/Task2/Offline/Corpus/sanitation.py.
  3. Xuất bản thư mục dữ liệu Data/Task2/releases/task2-data-v3/ và bàn giao 4 bundle.
- **Phase B (TinDipLaPo)**:
  1. Hoàn thiện mã nguồn nạp ngữ cảnh truy xuất tại SourceAPI/kaggle_worker.py và Source/Task2/Online/RetrievingAnswer/bm25.py.
  2. Thiết lập request batch Kaggle cho E1 sẵn sàng tiếp nhận 	ask2-data-v3.
