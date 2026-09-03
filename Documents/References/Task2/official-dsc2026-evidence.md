# Bằng chứng chính thức DSC2026 Task 2

## Đã khóa

| Nhãn | Nguồn | Nội dung |
|---|---|---|
| official | [Codabench LegalQA](https://www.codabench.org/competitions/17716/) | Submission answer; METEOR chính, ROUGE-L phụ; phase public/private |
| official-local | Thông báo BTC do thành viên nhóm cung cấp 27-08-2026 | Mỗi task tổng tham số nhỏ hơn 4B; không API; chỉ data BTC; không augmentation; có thể tải open weights |
| official-local | Thông báo bổ sung BTC | Task 1/Task 2 độc lập và cấm dùng data chéo |
| official-local | Data Overview trong gói Task 2 | Liệt kê `selected-contexts` làm ngữ cảnh/căn cứ cho Task 2 |
| measured-local | Warm-up manifest/audit | 500 question-answer; chỉ là aggregate/schema evidence |

## Quyết định khi nguồn có vẻ xung đột

Hệ thống fail-closed ở Task2-only. `selected-contexts` chỉ được dùng sau khi manifest
chứng minh provenance Task 2 và ADR retrieval accepted; mọi Task 1/shared corpus
không phân định vẫn bị reject. Duyệt corpus không tự động duyệt GPU managed/upload.

## Open gates

- Exact scorer/version/tokenization của BTC.
- Schema train Task 2 cuối cùng và checksum.
- Xác nhận nếu muốn dùng GPU managed từ xa.
- License/revision/parameter count của candidate cuối.

Không biến open gate thành permission hoặc claim.
