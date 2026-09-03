# Checkpoint 01-09-2026 V1

- Project: DSC-TinDipLaPo, Task 2 LegalQA E0.
- Owner: training/integration owner.
- Git: HEAD `444a0db6bedda1c052b700cbbdffc065dcad7e0d`; worktree có thay đổi chưa commit.
- System: Windows, RTX 2050 4 GB, driver 596.36, PyTorch 2.7.0+cu128.
- Evidence class: `measured-local` cho execution/resource; không phải official score.
- Data: `task2-data-v1`, manifest SHA-256
  `f9ebfd9417da186c2292151ea233c1acc71f69b529b34c7574bb7e34488194b5`;
  split 6.300 train / 700 validation.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 1.543.714.304 parameters.
- Config: answer-only QLoRA NF4, sequence 1.024, micro-batch 1, accumulation 16,
  LoRA 16/32/0,05, LR 2e-4, seed 2026, một epoch.
- Command: `.venv-task2-ml\Scripts\python.exe Source\Task2\pipeline.py run
  --request Data\Task2\control\requests\train_e0_full.json`.
- Kết quả: status `READY`; runtime 12.860,145 s; loss 1,113865; không NaN/OOM.
- Resource: peak allocated 4.318.729.728 byte trên physical 4.294.443.008 byte;
  `PASS_WITH_WDDM_OVERSUBSCRIPTION`.
- Artifact: `Data/Task2/runs/e0-full-v1`; checkpoint manifest và adapter giữ local,
  ngoài Git.
- Giới hạn: train loss không chứng minh chất lượng; run manifest chỉ pin HEAD, chưa pin
  dirty diff; scorer parity vẫn provisional.
- Replay: không ghi đè `e0-full-v1`; run mới phải dùng ID mới và khóa source tree.
- Throughput gate: batch 8 hoàn tất 8 câu/311,93 giây trên RTX 2050; batch 16 bị
  operator dừng và không được promote.
- Operator decision: dừng inference nặng; full validation 700/700, metrics/slices/errors
  và Public đều `DEFERRED_NOT_RUN`.
- Decision: `REPEAT` chỉ khi validation hoặc replay chứng minh checkpoint không dùng được;
  hiện giữ làm E0 anchor.
