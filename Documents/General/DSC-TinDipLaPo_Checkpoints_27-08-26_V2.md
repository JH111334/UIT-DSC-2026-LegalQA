# Checkpoint — Task 2 Official Data Integration & Offline Delivery Pipeline

## Định danh

- Project: DSC-TinDipLaPo
- Ngày: 27-08-26
- Phiên bản: V2
- Loại: checkpoint
- Owner: Hoàng Nguyễn Minh Khang
- Evidence class: measured|static_check
- Kết quả: PASS toàn bộ quality gates, tích hợp trọn vẹn dữ liệu chính thức Task 2 và bộ đánh giá offline.

## Cấu hình và lệnh thực thi

- Windows; Python 3.12; local CPU execution; không external API.
- Lệnh kiểm định dự án: uv run python skills/retrieval-delivery/scripts/validate_project.py -> PASS.
- Lệnh kiểm thử tự động:
  - uv run pytest -p no:cacheprovider --basetemp .tmp\pytest-run --cov=text_retrieval_agent --cov-report=term-missing -> 28/28 PASS, độ phủ 93.95%.
  - uv run python -m unittest discover -s DSC2026/tests -v -> 7/7 PASS.
- Lệnh chuẩn hóa mã nguồn:
  - uv lock --check -> PASS.
  - uv run ruff format --check . -> PASS.
  - uv run ruff check . -> PASS.
  - uv run mypy -> PASS.

## Kết quả kiểm toán và phân tích kỹ thuật Task 2

- Dữ liệu chính thức Data/Task2/:
  - 	rain.json: 7.000 bản ghi (SHA-256: 2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988), 0 rỗng, 99.4% có xuống dòng, 89.0% chứa trích dẫn Điều/Khoản.
  - public-official.json: 1.000 câu hỏi đánh giá (SHA-256: 5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5).
  - selected-contexts/: 8.532 tệp văn bản căn cứ pháp luật.
- Pipeline độc lập Task 2:
  - udit: Báo cáo phân bố độ dài, ký tự và mẫu cấu trúc.
  - preprocess: Chuẩn hóa Unicode NFC, bảo toàn ngắt dòng và nguồn gốc dữ liệu.
  - split: Group-aware split tỉ lệ 90/10 (6.300 train / 700 validation) loại trừ triệt để rò rỉ nhóm câu hỏi.
  - prepare-training: Tạo bộ dữ liệu SFT (	rain_sft.jsonl, alidation_sft.jsonl) cho mô hình < 4B tham số trong allowlist (Qwen/Qwen2.5-1.5B-Instruct).
  - evaluate: Bộ đánh giá cục bộ METEOR và ROUGE-L không phụ thuộc bên ngoài.
  - package: Đóng gói chuẩn submission.zip chứa duy nhất submission.json cho 1.000 câu hỏi public.

## Ranh giới và Next Gate

- Ranh giới: Tuyệt đối độc lập Task 1 và Task 2; cấm dùng API trong phương pháp; tổng tham số hệ thống < 4 tỷ.
- Next Gate: Thực hiện fine-tuning cục bộ/offline trên môi trường GPU được kiểm soát, đánh giá checkpoint qua METEOR/ROUGE-L và xuất file dự đoán chính thức cho phase Public Test.
