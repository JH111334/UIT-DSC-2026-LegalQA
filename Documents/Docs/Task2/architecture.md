# Task 2 — LegalQA

## Contract

Input: organizer question ID và câu hỏi Task 2. Output submission: answer tiếng Việt.
Training/inference chỉ nhận data và checkpoint có provenance Task 2.

## Kiến trúc đích

```text
PHASE A — OFFLINE DATA BUILD

train.json ── QA preprocess ── train/validation ── answer-only SFT generator
     └────── gold-answer citation extraction ── diagnostic qrels

selected-contexts
     └── integrity/normalization ── legal parser ── hierarchical chunks/metadata
                                  ├── BM25 index
                                  ├── dense embeddings/FAISS       [diagnostic]
                                  └── citation graph               [optional disabled]

PHASE B — ONLINE / BATCH INFERENCE

question ── BM25 ──┐
                    ├── RRF hybrid ── metadata/parent packing ── candidates
         dense ─────┘              ── reranker ── dynamic K
                                  ── evidence packing ── Qwen SFT generator ── answer

citation graph [OFF] ── chỉ nối sau hybrid bằng gate ADR-T2-0004
```

`Online` chỉ có nghĩa là runtime/batch inference trong môi trường đội kiểm soát;
không phải API hoặc endpoint Internet. Sơ đồ là kiến trúc đích. E0
chạy direct generation; E1 chỉ bật BM25 Task2-only; dense-only là diagnostic và
BM25+dense/RRF là hybrid challenger chính. Reranker/dynamic-K mở sau paired evidence.
Citation graph optional, mặc định tắt và không được tích hợp trước hybrid.

Code Task 2 nằm dưới một namespace để không trộn Task 1:

```text
Source/Task2/
├── Offline/{QA,Corpus}                 # data build và release contract
├── Online/Training                     # SFT runtime sau approval
├── Online/RetrievingAnswer             # retrieval, answer, submission boundary
└── pipeline.py                          # contract | run --request
```

Tên package kỹ thuật là `RetrievingAnswer`; dấu `&` chỉ dùng trong nhãn tài liệu.
Tokenizer hoặc index do teammate tạo là attachment/candidate, không tự trở thành
control/index artifact đã duyệt.

Không được lấy query, context, label hoặc checkpoint huấn luyện bằng Task 1. Không
dùng API, external data hoặc augmentation. Hugging Face chỉ là nguồn tải trọng số;
train/inference diễn ra trong môi trường đội kiểm soát.

E0 là baseline bắt buộc. E1 chỉ được build từ `selected-contexts` có manifest Task 2
sau khi ADR retrieval được duyệt; không gọi runtime Task 1/shared. Qrels suy từ
citation của gold answer chỉ dùng chẩn đoán Recall@k/MRR/nDCG, không dùng ở
Public/Private và không được gọi là gold.

## Gate

- Pin exact scorer/version trước model selection.
- Record METEOR, ROUGE-L, truncation, repetition, latency, memory, and failures.
- Thiết kế validation slices, trace và error taxonomy trước first run; sau first run
  mới diagnose bottleneck và thay một biến chính trong paired comparison.
- Treat warm-up answers as historical organizer references, not current legal advice.
- Giữ local provisional score tách khỏi official Codabench score.
