# Task 2 Offline/Online

Đây là namespace riêng của Task 2 LegalQA. `Online` nghĩa là batch runtime trong
môi trường đội kiểm soát, không phải API hay endpoint từ xa.

## Cấu trúc

```text
Source/Task2/
├── pipeline.py                         # contract | run --request
├── Offline/
│   ├── QA/                             # QA audit/split/bundle
│   ├── Corpus/                         # corpus audit/parser/chunks/qrels
│   ├── core/                           # schema, I/O, normalize, metadata
│   ├── pipeline.py                     # build/preflight bằng JSON request
│   ├── release_builder.py
│   ├── release_audit.py
│   └── required_artifacts.json
└── Online/
    ├── cli.py                          # operation dispatcher
    ├── inventory.py                    # artifact inventory + next safe stage
    ├── request.py                      # compact request contract
    ├── evaluation.py                   # metric/slice/paired/error artifacts
    ├── Training/
    │   └── runtime.py                  # tokenizer audit + answer-only LoRA/QLoRA
    └── RetrievingAnswer/
        ├── batch.py                    # resumable deterministic batch
        ├── bm25.py                     # Task2-only SQLite FTS5 BM25
        ├── engine.py                   # E0/E1 evidence + Qwen generation
        ├── fusion.py                   # RRF seam for later E2b hybrid
        └── submission.py               # exact schema + deterministic ZIP replay
```

Tên package là `RetrievingAnswer`; ký tự `&` chỉ dùng trong nhãn tài liệu vì không
hợp lệ trong Python import. Phase A sở hữu data release và optional candidate index;
Phase B sở hữu SFT/generation dù code vẫn chạy batch, không phải API service.

## Trạng thái thật

Đã triển khai và kiểm thử:

- release/control/run preflight fail-closed;
- preprocessing QA/corpus và release audit;
- answer-only masking, tokenizer audit trên local snapshot;
- LoRA/QLoRA trainer và deterministic Qwen batch inference;
- Task2-only BM25 index/search, HIGH-confidence proxy Recall/MRR/nDCG;
- parent-context packing, provisional METEOR/ROUGE-L, slice/error/paired reports;
- exact Public/Private ID gate và ZIP chỉ chứa UTF-8 `submission.json`;
- RRF utility để nối dense candidate sau E1.

Chưa được gọi là kết quả ML/competition:

- exact model snapshot, CUDA ML environment, tokenizer audit và E0 full checkpoint đã có;
- full validation đã được operator dừng tại 11/700 vì throughput; chưa có complete
  validation metric;
- một Public ZIP 1.000 ID pass format nhưng không có method trace chứng minh đã nạp
  adapter E0, nên chỉ là external candidate, không phải canonical Public run;
- batch 8 là cấu hình lớn nhất đã benchmark hoàn tất (8 câu/311,93 giây, không OOM);
- scorer parity BTC còn mở;
- dense encoder/FAISS, metadata boost, reranker và dynamic-K chưa triển khai;
- Private payload chưa được BTC phát hành.

Citation graph giữ `OPTIONAL_DISABLED`. Dense-only chỉ là E2a diagnostic; challenger
neural được phép promote đầu tiên phải là E2b `BM25 + dense -> RRF` sau khi model
role/allowlist và parameter budget Task 2 được duyệt.

## CLI tối thiểu

```powershell
uv run python "Source\Task2\pipeline.py" contract
uv run python "Source\Task2\pipeline.py" run --request <pipeline_request.json>
```

CLI chỉ có hai command. Operation, stage, profile, path và output report nằm trong
JSON request để replay được. Online operations là:

```text
inventory -> advance one safe stage
audit-tokenizer -> train -> evaluate -> predict
build-index (E1, kế thừa READY E0) -> evaluate/predict E1
```

`inventory` chỉ đọc artifact. `advance` skip release/tokenizer/checkpoint đã hoàn tất và
chạy đúng một operation kế tiếp; Public bị chặn cho tới khi validation predictions và
metrics pass. Mỗi executable operation chạy preflight trước; thiếu artifact/hash/approval
thì exit code 2.
ML modules được import lazy, nên contract, preprocess, test và BM25 fixture không cần
cài model stack.

Code build index nằm trong `Online/RetrievingAnswer` để dùng chung runtime contract.
Phase A có thể chạy và bàn giao candidate index ngoài immutable release; Phase B vẫn phải
verify manifest hoặc rebuild trước khi ký canonical E1 run.

Archive `Source/ol.zip` được giữ nguyên local-only với SHA-256
`ADB6992499CB036CD004587B98597DDAF5AC8148AAE3CFA9C50B5DCE468A9860`; archive không
phải runtime và không vào Git. Handoff canonical nằm tại
[`BanGiao.md`](../../Documents/Decision-making/BanGiao.md).
