# Error 01-09-2026 V1

- Checkpoint: `e0-full-v1`.
- Stage: Public inference attempt `e0-greedy-1536-v0`.
- Invariant: validation hoàn chỉnh phải precede Public; output không được bị cap/lặp có
  hệ thống trước khi package.
- Expected: 1.000 non-empty answers sau validation gate.
- Actual: dừng tại 69/1.000; 19/69 (27,5%) chạm đúng 1.536 output token; p90=1.536.
- First bad: validation artifact chưa tồn tại khi Public được khởi động.
- Evidence: local diagnostic tại
  `Data/Task2/runs/e0-full-v1/diagnostics/public-attempt-max1536-aborted/diagnostic.json`.
- Cause status: chưa kết luận. Hypothesis gồm decode cap quá rộng, một phần target train
  mất EOS do truncation, hoặc generator verbosity/repetition.
- Containment: dừng đúng process, giữ partial prediction/trace ngoài submission path,
  thêm validation-before-Public gate và tách decoding config khỏi training config.
- Regression proof: Public preflight hiện reject khi thiếu `validation_predictions.json`
  hoặc `metrics.json`.
- Rollback: checkpoint E0 không đổi; decode parent v0 không resume.
- Next experiment khi operator mở lại: full validation sạch với decode
  `e0-greedy-768-b8-v2`; không resume payload partial cũ.
  greedy và repetition penalty 1.0 giữ nguyên.
- Residual risk: 768 vẫn có thể cap long answer; quyết định chỉ sau metric/slice/case.
