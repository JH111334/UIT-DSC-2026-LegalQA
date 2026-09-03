# ADR-T2-0002: Huấn luyện sinh độc lập cho Task 2

Status: accepted.

Clarification ngày 29-08-2026: ADR này tiếp tục khóa E0 direct-generation và cấm
mọi nối chéo từ Task 1. Đề xuất mở nhánh E1 trên corpus chính thức Task 2 nằm tại
[ADR-T2-0003](0003-task2-official-corpus-retrieval.md). Cho tới khi ADR-T2-0003
được duyệt, E1 không được chạy.

## Bối cảnh

BTC xác nhận Task 1 và Task 2 độc lập, không dùng dữ liệu chéo. Task 2 nộp answer
tiếng Việt và được xếp bằng METEOR, sau đó ROUGE-L. Quy định cấm API, dữ liệu ngoài,
augmentation và yêu cầu tổng tham số hệ thống nhỏ hơn 4 tỷ.

## Quyết định

- Luồng competition Task 2 là Task2-only preprocess → supervised fine-tuning →
  deterministic generation → METEOR/ROUGE-L → submission.
- Qwen/Qwen2.5-1.5B-Instruct là anchor first-run.
- ntphuc149/ViLegalQwen3-1.7B-Base là challenger chuyên ngành sau SFT.
- Loss chỉ tính answer; preserve cấu trúc target; group-aware split và tokenizer
  length audit là bắt buộc.
- Không nối Task 1 retrieval, không remote inference/API và không external data.
- Hugging Face Hub chỉ là nơi tải model/revision hợp lệ; execution do đội kiểm soát.

## Hệ quả

Baseline extractive/citation hiện có vẫn dùng cho fixture regression, nhưng không
được trình bày như phương pháp Task 2. ViLegal base không được so zero-shot với
instruction model. QLoRA là tối ưu bộ nhớ, không làm giảm số tham số danh nghĩa.

## Promotion gate

Scorer parity hoặc nhãn provisional rõ ràng; paired gain trên validation group-aware;
không missing/empty output; resource và failure log đầy đủ; submission ZIP replay
được từ môi trường sạch; model/license/parameter audit hợp lệ.
