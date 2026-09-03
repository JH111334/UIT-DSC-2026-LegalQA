# Error — DOCX QA lỗi trên tài liệu một trang

## Định danh

- Project: DSC-TinDipLaPo
- Ngày: 26-08-26
- Thời điểm: `2026-08-26T16:19:58+07:00` (`Asia/Bangkok`)
- Phiên bản: V1
- Git: `444a0db6bedda1c052b700cbbdffc065dcad7e0d`; branch `main`; worktree `dirty=72; untracked=65`
- System: Windows 11 Home Single Language 10.0.26200 build 26200; Intel Core i5-12450H 8C/12T; RAM 16.866.684.928 byte; NVIDIA GeForce RTX 2050 4.096 MiB, driver 596.36; Python 3.12.10
- Trạng thái: `RESOLVED`; log đã loại secret.


## Phạm vi và hiện tượng

- Expected: helper render/inspect hai DOCX một trang.
- Actual: `IndexError` khi tính median của danh sách các trang đứng trước trang cuối nhưng danh sách rỗng.
- `first_bad`: nhánh underfilled-last-page trong global `author-validate-docx`; `last_known_good`: structural/round-trip QA không render.

## Chẩn đoán và khắc phục

- Nguyên nhân: guard `result.pages > 1` đặt sau phép truy cập danh sách.
- Khắc phục: chỉ tính `preceding_lengths` và cảnh báo trang cuối khi PDF có hơn một trang; bổ sung title metadata bằng backup và thay thế nguyên tử.
- Regression: headless render/structural/visual QA 2/2 PASS, mỗi file 1 trang, không còn warning metadata.
- Không chạy remote model, organizer data, training hoặc benchmark; fixture CLI vẫn chỉ là smoke evidence.
