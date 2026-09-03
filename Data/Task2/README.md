# Dữ liệu chính thức Task 2 LegalQA

Tập dữ liệu Task 2 do BTC cung cấp bao gồm:

- train.json: 7.000 cặp câu hỏi - câu trả lời tham chiếu chính thức phục vụ huấn luyện.
- public-official.json: 1.000 câu hỏi đánh giá giai đoạn Public Test.
- selected-contexts/: Kho văn bản căn cứ/ngữ cảnh pháp luật (context_*.json).
- DSC2026_Task2_LegalQA_Data_Overview.docx: Tài liệu tổng quan quy cách dữ liệu từ BTC.

## Quy tắc quản trị dữ liệu

Dữ liệu do BTC cung cấp được giữ cục bộ, không commit lên Git. Chỉ theo dõi
`manifest.json` và tài liệu này trong Git. Không tạo JSON giả để làm gate pass.

## Cây thư mục local

Các folder dưới đây là ranh giới lưu trữ; nội dung vẫn bị `.gitignore` loại khỏi Git:

```text
Data/Task2/
├── train.json
├── public-official.json
├── selected-contexts/
├── releases/
│   └── task2-data-v2/
│       ├── qa/
│       ├── corpus/
│       └── diagnostics/
├── control/
│   ├── approvals/
│   ├── imports/
│   ├── preflight/
│   ├── requests/
│   ├── model/
│   ├── tokenizer/
│   ├── training/
│   ├── scorer/
│   └── submission/
├── indexes/
│   └── e1-bm25/
├── runs/
├── submissions/
│   ├── public/
│   └── private/
└── cache/
```

Private Test chưa được BTC bàn giao trong workspace, nên không có private payload,
private preflight hay prediction giả. Khi có file chính thức, nó phải đi qua manifest
và phase gate riêng trước inference.

`releases/task2-data-v2` là release active được nhập chọn lọc từ `DSC2026.zip`, đã qua
E0/E1 deep preflight và tuân theo `Documents/Decision-making/BanGiao.md`. `indexes`, `runs`, `submissions`
và `cache` là output hậu preprocessing; không phải một phần của data release.
Tokenizer report cũng là control artifact của đúng model snapshot, không nằm trong
immutable release. Attachment sai backbone phải giữ ở `control/tokenizer/rejected/`
và không được dùng mở gate.

Kiểm tra cấu trúc project:

```powershell
uv run python skills/retrieval-delivery/scripts/validate_project.py
```
