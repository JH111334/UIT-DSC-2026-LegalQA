# Đối chiếu tài liệu trong `DSC2026`

Ngày đối chiếu: 01-09-2026. Phạm vi là `DSC2026/Documents`; thư mục nguồn và
`DSC2026.zip` không bị sửa hoặc xóa.

## Kết quả hợp nhất

| Nhóm nguồn trong `DSC2026/Documents` | Canonical hiện tại | Quyết định |
|---|---|---|
| Context và technical execution guide | `TASK1.md`, `Source/README.md`, kiến trúc Task 2 | Đã hấp thụ; không copy bản cũ |
| ADR evidence baseline / independent training | ADR-T2-0002, ADR-T2-0003 | Đã được thay thế bởi quyết định mới hơn |
| ADR remote compute | `SourceAPI/README.md`, ADR-T2-0005/0006 | HF inference API rejected; private Kaggle batch được duyệt có điều kiện |
| Official evidence và data/model register | Organizer contract audit, allowlist, TASK1 | Đã hấp thụ và cập nhật bằng release/model hash thực tế |
| Experiment record template | `Documents/General/RULE.md` | RULE hiện chặt hơn; không tạo source of truth thứ hai |
| DOCX, PDF, preview PNG | Markdown canonical ở root | Bản render dẫn xuất; không nhập lại |

Không có mệnh đề còn hiệu lực nào trong bộ tài liệu con cần ghi đè tài liệu root.
Những câu cũ như “scaffold chưa training” hoặc “E1 chưa accepted” đã lỗi thời và không
được merge nguyên văn.

## Đối chiếu artifact ZIP

Release canonical đã nhận QA/corpus/diagnostic data từ archive. Năm artifact không
được nhập vào immutable release là có chủ ý:

- `index/corpus.sqlite` và `index/retrieval_index_report.json`: index dẫn xuất chưa qua
  E1 parent/lineage gate; phải build lại bằng code canonical sau E0 validation.
- Hai preflight nguồn: giữ ở `Data/Task2/control/imports`, không đặt trong release.
- Tokenizer report nguồn: không đúng snapshot anchor đã pin; thay bằng tokenizer audit
  thực đo ở control root.

Các manifest/report canonical khác byte so với archive vì đã bỏ artifact ngoài boundary
và ghi nhận đầy đủ 30 document không indexable, gồm quarantine cùng exact-duplicate
aliases. Đây là migration có chủ ý, không phải missing extraction.

Kết luận: `Documents` ở root là source of truth. Người dùng có thể xóa bản sao
`DSC2026`/archive sau khi tự xác nhận không cần giữ nguồn rollback; project không tự xóa
hai artifact đó.
