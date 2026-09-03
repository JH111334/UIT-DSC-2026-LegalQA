# ADR-T2-0003: Retrieval trên corpus chính thức Task 2

Status: **accepted — 31-08-2026; không cấp quyền external upload hoặc train proxy**.

Ngày đề xuất: 29-08-2026.

## Bối cảnh

[Data Overview Task 2](../../../Data/Task2/DSC2026_Task2_LegalQA_Data_Overview.docx)
mô tả mục tiêu là truy xuất văn bản liên quan rồi tạo câu trả lời và liệt kê
`selected-contexts.zip` trong gói Task 2. Quy định Task 1/Task 2 độc lập vẫn cấm
mọi data, context, label, index, runtime và checkpoint dùng chéo.

ADR-T2-0002 đã đúng khi chọn direct-generation làm first-run và cấm nối retrieval
Task 1. Tuy nhiên, câu chữ cũ chưa phân biệt retrieval Task 1 với retrieval được
xây độc lập trên corpus do BTC đặt trong Task 2.

## Quyết định

1. Giữ `E0 = Qwen Task2-only answer-only SFT, không retrieval` làm control bắt buộc.
2. Sau khi provenance/checksum của toàn bộ `selected-contexts` đạt gate, cho phép
   `E1 = BM25 Task2-only -> evidence packing -> cùng generator/checkpoint E0`.
3. E1 có data, code, config, index, test, trace và run bundle riêng; không import
   corpus/index/runtime/label/checkpoint Task 1 hoặc shared fixture.
4. Citation trong gold answer có thể tạo `citation-derived diagnostic qrels` để đo
   Recall@k, MRR và nDCG. Artifact này là proxy có confidence/coverage, không phải
   organizer gold, không tồn tại ở Public/Private và chưa được dùng train retriever.
5. Sau E1, dense-only chỉ dùng diagnostic; challenger neural chính là
   `BM25 + dense -> RRF`. Citation graph optional và disabled cho tới khi hybrid
   hợp lệ cùng gate riêng trong ADR-T2-0004. Model neural phải nằm trong allowlist,
   có revision/parameter audit và giữ tổng runtime dưới 4 tỷ tham số.
6. Promotion cuối cùng dựa vào paired METEOR/ROUGE-L end-to-end trên cùng split,
   generator, prompt budget và decoding; retrieval metric đơn lẻ không đủ.

## Acceptance gate

ADR được accept khi đồng thời có:

- người dùng/nhóm duyệt cách hiểu corpus Task 2;
- manifest Task 2 bao phủ 8.532 context, Data Overview và mọi derived chunk;
- `BanGiao.md` đạt cho profile E1;
- scorer contract được pin hoặc được duyệt rõ là provisional;
- cross-task scanner chứng minh không có Task 1 path/artifact;
- môi trường chạy được duyệt riêng. Duyệt ADR không tự động duyệt Kaggle/upload.

Trạng thái kiểm tra ngày 31-08-2026: người dùng đã chốt cách hiểu corpus Task 2;
manifest 8.532 document/316.100 chunk, các preflight E0/E1/Public và deep E1 release
audit đều PASS; cross-task hit bằng 0. Scorer parity, tokenizer đúng anchor và GPU
smoke vẫn là gate thực thi riêng. 150 manual citation audit pending không chặn build/
debug E1, nhưng chặn promotion dựa trên proxy qrels. ADR này không cho phép upload
organizer data lên Kaggle/Hugging Face hoặc dùng qrels để train retriever.

## Rollback

Nếu provenance không pass, E1 không thắng E0 hoặc gây regression ngoài ngưỡng đã
duyệt, dừng retrieval và quay về E0. Không cần rollback data release QA hoặc model
checkpoint E0.

## Hệ quả

- ADR-T2-0002 không bị xóa; nó trở thành quyết định canonical cho E0.
- `selected-contexts` preprocessing là nhánh song song, không thay QA preprocessing.
- BM25/FAISS/index build thuộc retrieval layer, không gọi là preprocessing.
- Hybrid, dynamic-k và neural retriever là roadmap, không phải baseline đã triển khai.
- Citation graph không được tích hợp trước hybrid; xem ADR-T2-0004.

## Bằng chứng

- [Contract và audit Task 2](../../References/Task2/organizer-contract-and-data-audit.md)
- [Kế hoạch phần việc hậu tiền xử lý](../TASK1.md)
- [VLSP 2025 MLQA-TSR overview](https://aclanthology.org/2025.vlsp-1.48/)
- [Official VLSP baseline repository](https://github.com/sonlam1102/VLSP2025-MLQA-TSR)
