# Error V2 — Public artifact không có method provenance hợp lệ

- **Loại**: quality/provenance gate FAIL; không phải lỗi format submission.
- **Artifact**: `submissions/Task2/public_submission.zip`
- **SHA-256**: `bf9ac32c29ca5df621b4a96a54a62e4025b7e91f6fe6d6d4ccdfa432dcc04007`

## Quan sát

ZIP pass validator với 1.000 ID và answer không rỗng. Tuy nhiên code runner được dùng
làm bằng chứng chỉ nạp `Qwen/Qwen2.5-1.5B-Instruct`; không có `PeftModel`/adapter,
không pin revision và không ghi trace. `public_predictions.json` cũ có cùng ID nhưng
1.000/1.000 answer khác payload trong ZIP.

## Kết luận

Không thể quy artifact cho checkpoint `e0-full-v1`. Chỉ được claim format integrity;
METEOR, ROUGE-L và chất lượng không có reference để đo. Runner remote đã bị khóa, file
predictions gây nhầm lẫn đã xóa. Báo cáo máy đọc nằm tại
`Data/Task2/runs/public-external-artifact-audit-v1/provenance_audit.json`.

## Cách tránh lặp lại

Mọi inference run phải ghi input/model revision/adapter/code/config hashes, trace từng
sample, runtime approval và output hash. Không package một JSON không cùng lineage với
run manifest.
