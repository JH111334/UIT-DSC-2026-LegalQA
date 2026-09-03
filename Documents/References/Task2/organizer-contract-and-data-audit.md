# Contract BTC và audit dữ liệu Task 2

## Contract đã xác minh

- Task 2 độc lập với Task 1. Chỉ dùng dữ liệu Task 2 do BTC cung cấp; không chuyển
  câu hỏi, context, nhãn, checkpoint huấn luyện bằng dữ liệu Task 1 sang Task 2.
- Không dùng dữ liệu ngoài và không tăng cường dữ liệu. Pretrained model hợp lệ
  không đồng nghĩa được lấy corpus pretrain ra dùng lại.
- Tổng tham số của toàn bộ hệ thống Task 2 phải nhỏ hơn 4 tỷ. LoRA/quantization chỉ
  giảm bộ nhớ, không biến model lớn hơn 4 tỷ thành hợp lệ.
- Không dùng API, kể cả API phi lợi nhuận, trong phương pháp. Hugging Face chỉ dùng
  làm nơi tải trọng số hợp lệ; train và inference phải diễn ra trong môi trường đội
  cầm được và kiểm soát được.
- Submission là ZIP chỉ chứa submission.json. Mỗi question_id ánh xạ tới một object
  có answer là chuỗi tiếng Việt; phải đủ tất cả câu hỏi.
- Codabench công khai METEOR là metric chính và ROUGE-L là metric phụ. Cần lấy đúng
  scorer/version/config của BTC trước khi tối ưu cuối.

Public Test kết thúc lúc 16:59 UTC ngày 18-09-2026; Private Test chạy từ 17:00 UTC
ngày 18-09-2026 đến 16:59 UTC ngày 23-09-2026 theo phase metadata công khai tại
thời điểm kiểm tra 27-08-2026.

## Xử lý xung đột thông tin

Data Overview trong gói Task 2 mô tả `selected-contexts.zip` là kho văn bản làm
ngữ cảnh/căn cứ cho Task 2. Vì vậy, corpus này có thể được xem xét như organizer
data Task 2 khi manifest chứng minh provenance và checksum. Điều này không cho phép
dùng corpus/index/runtime Task 1 hoặc một corpus “shared” không phân định nguồn.

Quyết định fail-closed hiện tại:

- `selected-contexts` có provenance Task 2: được preprocess độc lập; retrieval chỉ
  mở sau [ADR-T2-0003](../../Decision-making/Task2/0003-task2-official-corpus-retrieval.md);
- Task 1/shared/web corpus hoặc artifact mơ hồ: bị reject cho tới khi có trả lời
  bằng văn bản của BTC;
- duyệt corpus không tự động duyệt external upload hoặc train retriever. Private Kaggle E0
  batch được duyệt riêng ngày 03-09-2026 theo ADR-T2-0006 sau xác nhận của operator với BTC.

## Audit dữ liệu chính thức Task 2 (Official Train & Public)

Dữ liệu chính thức gồm `train.json` (7.000 bản ghi), `public-official.json` (1.000 bản ghi), và thư mục `selected-contexts/` (8.532 tệp văn bản ngữ cảnh).
Checksum SHA-256 khớp manifest `Data/Task2/manifest.json`:
- `train.json`: `2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988`
- `public-official.json`: `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5`

| Thuộc tính kiểm toán | Tập Train (7.000) | Tập Public (1.000) |
|---|---:|---:|
| Số bản ghi câu hỏi | 7.000 | 1.000 |
| Question rỗng / Answer rỗng | 0 / 0 | 0 / 1.000 (đáp án ẩn) |
| Cặp trùng lặp hoàn toàn | 2 | 0 |
| Độ dài answer theo ký tự, min / p50 / p95 / max | 129 / 1.410 / 3.143 / 10.755 | Chưa có nhãn |
| Số từ (word tokens), min / p50 / p95 / max | 30 / 314 / 695 / 2.401 | Chưa có nhãn |
| Answer chứa ký tự xuống dòng `\n` | 6.961 / 7.000 (99,4%) | Chưa có nhãn |
| Answer chứa mẫu Điều/Khoản/Nghị định | 6.230 / 7.000 (89,0%) | Chưa có nhãn |

## Audit warm-up Task 2 (Tham chiếu lịch sử)

Audit mô tả 500 bản ghi warm-up ban đầu; không phải tập train/public chính thức.
Checksum SHA-256: `b824e4f18bd9181c021498a28e402b7374d6d559f2ff28caa4120a9d932f82c5`.

## Hệ quả kỹ thuật

- Bảo tồn toàn vẹn định dạng xuống dòng và danh sách trích dẫn Điều/Khoản trong target.
- Chọn độ dài sequence bằng tokenizer report cho `question + packed context +
  target`; không khóa mức tối thiểu từ thống kê ký tự.
- Tách tập train/validation theo group-aware (nhóm câu hỏi tương đồng) với tỉ lệ 90/10 (6.300 train / 700 validation).
- Giữ E0 direct-generation làm control; E1 BM25 Task2-only chỉ mở sau corpus/ADR gate.
- Bộ đánh giá cục bộ METEOR/ROUGE-L hiện là provisional tới khi scorer parity đạt.
- Citation-derived qrels chỉ là diagnostic proxy có confidence/coverage.

## Trạng thái Gate

1. Schema/checksum `train.json` và `public-official.json`: **ĐÃ XÁC MINH**.
2. Corpus release canonical: **ĐÃ TÍCH HỢP VÀ DEEP-AUDIT PASS** với 8.532 document,
   8.502 indexable document và 316.100 chunk; 30 non-indexable có reason/provenance.
3. QA group-aware split và corpus parser/chunker: **ĐÃ TÍCH HỢP**. Trainer,
   Task2-only BM25, deterministic batch, observability và submission builder:
   **ĐÃ TRIỂN KHAI/UNIT TEST**. Full E0 SFT đã `READY`; full validation đang chạy.
4. Local METEOR/ROUGE-L: **PROVISIONAL**; exact scorer parity: **OPEN GATE**.
5. Qwen anchor revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 1.543.714.304
   parameters: **ĐÃ XÁC MINH VÀ TẢI SNAPSHOT LOCAL CÓ HASH**.
6. Giới hạn <4B/no-API/cross-task rejection: **ĐÃ KHÓA TRONG POLICY**; từng runtime
   vẫn phải audit trước run. Hugging Face inference API không được duyệt; private Kaggle E0
   batch chỉ hợp lệ theo payload/hash/trace contract của ADR-T2-0006.

## Nguồn

- [UIT-DSC 2026 LegalQA trên Codabench](https://www.codabench.org/competitions/17716/)
- Thông báo BTC do người dùng cung cấp ngày 27-08-2026.
- [METEOR](https://aclanthology.org/W05-0909/)
- [ROUGE](https://aclanthology.org/W04-1013/)
