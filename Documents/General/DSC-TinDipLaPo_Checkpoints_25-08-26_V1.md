# Checkpoint nâng cấp quy ước repository

- Project: DSC-TinDipLaPo
- Ngày: 2026-08-25 19:13:19 +0700
- Phiên bản: V1
- Loại: checkpoint
- Evidence class: migration_applied
- Owner: local repository owner
- Git revision: NOT_CAPTURED — script không gọi Git
- System profile: RTX 2050 4 GB VRAM, RAM 16 GB là profile lịch sử; cần đo lại khi benchmark
- Data/corpus manifest: không đọc hoặc thay đổi dữ liệu
- Config hash: không áp dụng
- Lệnh dự kiến: `python scripts/upgrade_noncourse_repositories.py --apply`

## Thay đổi

  - `write` `Documents/General/RULE.md`: Khóa quy tắc General
  - `write` `skills/retrieval-delivery/SKILL.md`: Khóa local-skill logging rule

## Ranh giới kết quả

Đây là bằng chứng migration tài liệu/scaffold, không phải smoke test, benchmark,
kết quả mô hình hoặc mức sẵn sàng production. Script không cài package, tải
data/model, gọi service, commit hoặc push.

## Kết luận và bước kế tiếp

Chạy validator/test được liệt kê trong repository đích. Chỉ tạo benchmark log mới
khi có data/config/code revision, metric protocol và resource artifact thực đo.
