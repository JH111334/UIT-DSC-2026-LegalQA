# Remote inference của Task 2

Hugging Face model inference API vẫn bị khóa. Không đặt `HF_TOKEN`, Kaggle credential, dữ liệu
BTC hoặc checkpoint vào request/source code của Task 2.

Kiểm tra ranh giới:

```powershell
uv run python SourceAPI\pipeline.py contract
```

Chạy pipeline hợp lệ:

```powershell
uv run python Source\Task2\pipeline.py contract
uv run python Source\Task2\pipeline.py run --request <request.json>
```

Hugging Face chỉ là model registry để tải snapshot đã pin. Xem ADR-T2-0005 và
`SourceAPI/README.md`. Private Kaggle batch do đội kiểm soát được phép riêng theo
ADR-T2-0006; Hugging Face trong kernel chỉ tải public weights theo revision allowlist.
