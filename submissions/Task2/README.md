# Task 2 submissions

Chỉ giữ contract và validator trong Git; package thật ghi vào `outputs/` và bị
`.gitignore` loại.

Contract đã khóa từ thông báo BTC/project contract:

```json
{
  "<question_id>": {
    "answer": "<câu trả lời tiếng Việt không rỗng>"
  }
}
```

- đủ và đúng toàn bộ ID của phase, không ID thừa/trùng;
- mỗi record chỉ có field `answer`;
- UTF-8 không BOM;
- ZIP chỉ có đúng `submission.json` ở root;
- Public và Private phải validate theo manifest phase riêng.

Builder canonical là `Online.RetrievingAnswer.submission.build_submission`; nó kiểm
prediction, tạo ZIP deterministic rồi mở lại archive để replay contract. Một ZIP
fixture chỉ chứng minh packaging, không phải candidate submission hoặc điểm BTC.

Candidate local ngày 01-09-2026 có SHA-256 `bf9ac32c...` và pass format 1.000/1.000,
nhưng lineage tới adapter E0 không đạt. Không dùng filename hoặc validator format để suy
ra model, quality, METEOR/ROUGE-L hay compliance; xem provenance audit của run.
