# Chiến lược thực nghiệm tiết kiệm compute cho Task 2

- **Trạng thái**: `ACCEPTED_DESIGN`.
- **Ngày**: 02-09-2026.
- **Phạm vi**: chọn và kiểm chứng thay đổi Phase B; không tự mở quyền dùng dịch vụ.
- **Mục tiêu**: tăng lượng thông tin thu được trên mỗi phút GPU mà vẫn giữ được khả năng
  quy nguyên nhân.

## 1. Kết luận nghiên cứu

Chiến lược đúng không phải “mỗi lần chỉ thử đúng một giá trị” và cũng không phải “gom
nhiều thay đổi vào một cấu hình”. Một vòng được phép khảo sát nhiều candidate, nhưng:

1. tất cả candidate trả lời cùng một câu hỏi thực nghiệm;
2. mỗi candidate độc lập chỉ đổi một scientific knob so với cùng parent;
3. candidate rẻ bị loại ở component/probe gate;
4. chỉ một hoặc rất ít candidate vào full validation;
5. chỉ full confirmation mới có quyền đề nghị promote.

Gọi đây là:

```text
wide-but-cheap screening -> narrow confirmation -> sequential integration
```

Hyperband cung cấp nguyên lý phân bổ dần tài nguyên và dừng sớm cấu hình yếu. Tuning
Playbook của Google nhấn mạnh mỗi round cần một mục tiêu hẹp, tách scientific, nuisance và
fixed parameters. NIST cho phép fractional factorial để screening, nhưng cảnh báo main
effect và interaction có thể bị confound. Vì vậy dự án chỉ mượn các nguyên lý này; không
cài một HPO framework hoặc chạy random sweep tự động.

## 2. Sửa lỗi trong strategy cũ

### Không dùng “orthogonal bundle” làm mặc định

BM25 và repetition penalty ở hai component khác nhau nhưng vẫn tương tác qua context và
output. Chạy cả hai trong một challenger không cho biết delta đến từ đâu. Performance knobs
chỉ được gộp khi:

- từng thay đổi đã thắng độc lập; hoặc
- có thiết kế ablation/factorial đủ để ước lượng interaction; hoặc
- thay đổi chỉ là observability/format và được chứng minh không đổi prediction hash.

### Wins lớn hơn Losses chưa đủ để promote

Quyết định cần đồng thời:

- delta METEOR primary và ROUGE-L secondary trên cùng ID;
- paired uncertainty, ưu tiên paired bootstrap confidence interval;
- wins/ties/losses để hiểu độ phủ;
- guardrail trên slice trọng yếu và invalid/empty rate;
- regression/counterexample review.

Không đặt trước các ngưỡng METEOR, Recall, evidence survival hoặc length ratio khi chưa có
anchor measured. Mục tiêu ban đầu là cải thiện tương đối so với parent và pass hard gates.

### Không dùng thời gian lịch sử làm hằng số

Các số “8–10 phút”, “batch 16”, “0,15 GPU giờ/run” chỉ có giá trị khi run manifest ghi GPU,
model/adapter, input count, batch, token cap và measured wall time. Nếu thiếu trace, chúng
chỉ là `unverified_operator_report`.

## 3. Cost classes

| Lớp | Ví dụ | Nơi chạy | Quyền quyết định |
|---|---|---|---|
| C0 | schema, checksum, leakage, tokenizer budget, submission contract | local | reject invalid candidate |
| C1 | BM25, qrels metric, packing simulation, scorer, cached trace analysis | local | screening |
| C2 | generation trên frozen probe panel | GPU được duyệt | shortlist |
| C3 | full 700 validation generation | GPU được duyệt | confirmation |
| C4 | full SFT/backbone/training recipe | GPU được duyệt | chỉ mở sau causal evidence |

Mọi hypothesis phải dùng hết bằng chứng C0/C1 trước khi xin C2–C4. Retrieval/index knobs
được đánh giá toàn validation locally trước; không tiêu GPU chỉ để đo Recall@k.

## 4. Multi-fidelity gates

### G0 — Contract gate

Kiểm scorer status, data/split/model/config hashes, output schema, expected IDs và resource
estimate. Fail thì không chạy candidate.

### G1 — Component-local screening

- Retrieval: sweep một grid nhỏ cho một knob trên cached corpus/qrels.
- Packing: replay cached ranked lists; đo proxy evidence survival và token budget.
- Decoding/prompt: static/token checks trước khi generation.
- Training: mask/length/OOM micro-probe; không dùng train loss làm quality gate.

Một round mặc định có tối đa ba candidate cộng parent. Con số này là internal operating
cap, không phải chân lý; tăng chỉ khi marginal information dự kiến lớn hơn chi phí.

### G2 — Frozen probe

Phase B tạo `evaluation_panel_manifest.json` một lần từ validation, gồm:

- `probe_ids`: stratified theo length, legal structure và question type;
- `confirmation_ids`: phần còn lại, không giao nhau;
- selection seed, slice registry và ID hashes.

Probe size được quyết định theo coverage và cost đã đo; không hardcode 50/100 vào contract.
Tất cả candidate trong round dùng cùng IDs, order, prompt cache và decode seed. Probe chỉ
shortlist, không promote.

### G3 — Confirmation

Chỉ candidate tốt nhất hoặc tối đa hai candidate bất phân định được chạy trên
`confirmation_ids`. Candidate thắng mới chạy full 700 để tạo báo cáo cuối. Nếu kết luận
thay đổi khi chuyển fidelity, giữ `INCONCLUSIVE` và điều tra sampling interaction.

### G4 — Integration

Tích hợp tuần tự:

```text
anchor A -> B thay knob X -> promote B
         -> C = B + thay knob Y -> compare C với B
```

Nếu X và Y nghi tương tác mạnh, chạy ma trận 2x2 trên probe/confirmation trước. Không đưa
một bundle chưa ablate thẳng vào full run.

## 5. Experiment batch contract

`experiment_batch.json` là kế hoạch, không phải một cấu hình trộn. Tối thiểu gồm:

```json
{
  "batch_id": "...",
  "goal": "...",
  "parent_run_id": "...",
  "scientific_knob": "...",
  "fixed_hashes": {},
  "candidate_values": [],
  "fidelity_ladder": [],
  "estimated_gpu_minutes": 0,
  "stop_conditions": [],
  "confirmation_rule": "..."
}
```

`nuisance_knobs` giữ cố định hoặc được tune công bằng cho mọi candidate. Nếu chúng thay
khác nhau không kiểm soát, comparison ghi `CONFOUNDED`.

## 6. Cache và replay

Không chạy lại stage có input/config hash không đổi:

- cache ranked candidates theo query/index/retrieval-config hash;
- cache packed prompt theo ranked-list/packer/tokenizer hash;
- cache prediction theo prompt/model/adapter/decode hash;
- metric và error report có thể tái tính local từ prediction cache.

Cache hit phải được xác minh bằng hash; không dùng path hoặc tên run làm bằng chứng.

## 7. Quota ledger

“30 giờ/tuần” là budget operator nêu, không phải quyền upload hay quota đã xác minh. Nếu
môi trường GPU được duyệt, mỗi request ghi `estimated_gpu_minutes`, `actual_gpu_minutes`,
`failure_reserve` và `information_goal`.

Internal policy v1:

- giữ ít nhất 25% budget tuần làm recovery/confirmation reserve;
- không đặt số run/ngày; chạy theo gate và information gain;
- C4 training phải có experiment card và stop condition riêng;
- job invalid do contract/config không được retry trên GPU trước khi local reproducer pass;
- dừng round khi candidate bị dominated hoặc uncertainty không thể giảm trong budget.

Tỷ lệ 25% là quyết định vận hành ban đầu, được sửa sau khi có resource history; không phải
khuyến nghị từ paper hoặc BTC.

## 8. Validation reuse và multiple comparisons

Lặp nhiều lần trên cùng 700 validation có thể làm nhóm overfit vào validation. Vì vậy:

- probe phục vụ exploration;
- confirmation IDs được giữ kín khỏi lựa chọn candidate cho tới G3;
- full 700 không dùng cho mọi thay đổi nhỏ;
- ghi số hypothesis/candidate đã thử;
- delta nhỏ sau nhiều lần thử giữ `REVIEW_REQUIRED` nếu paired interval chạm zero hoặc
  regression tập trung ở slice quan trọng.

Đây là biện pháp thực dụng lấy cảm hứng từ nghiên cứu adaptive holdout, không phải triển
khai thuật toán reusable holdout có bảo đảm hình thức.

## 9. Chuỗi knob theo causal reach

Chỉ mở nhóm knob gắn với bottleneck đã đo:

| Failure được hỗ trợ | Knob screening trước | Knob chưa mở |
|---|---|---|
| target/prompt truncation | sequence/output cap, pack budget | model mới |
| E1 retrieval miss | BM25 query/top-k/config | dense/reranker/graph |
| semantic miss sau BM25 | dense candidate + RRF | graph |
| evidence bị packing loại | dedup/order/final-k/parent budget | SFT recipe |
| evidence có nhưng model bỏ qua | prompt/SFT recipe | retrieval rewrite |
| repetition/cap hit | decode cap/repetition setting | backbone |

Không đổi Phase A representation từ Phase B. Nếu chunking là root cause, tạo change
request và release ID mới cho Phase A.

## 10. Evidence và nguồn

- [Hyperband, JMLR 2018](https://jmlr.org/papers/v18/16-558.html): adaptive resource
  allocation và early stopping; prior art cho multi-fidelity.
- [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook):
  incremental tuning, goal hẹp và scientific/nuisance/fixed parameters.
- [NIST fractional factorial guidance](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3345.htm):
  screening kinh tế và rủi ro interaction/confounding.
- [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/): lựa chọn kiểm định thống kê
  phải phù hợp metric và setup NLP.
- [Reusable Holdout, Science 2015](https://pubmed.ncbi.nlm.nih.gov/26250683/): rủi ro
  adaptive reuse của holdout; dự án chỉ áp dụng separation thực dụng.

Các nguồn là `prior-art`. Chúng không chứng minh candidate DSC nào tốt hoặc cho phép dùng
Kaggle/API. Promotion chỉ dựa trên organizer-compliant paired evidence của dự án.
