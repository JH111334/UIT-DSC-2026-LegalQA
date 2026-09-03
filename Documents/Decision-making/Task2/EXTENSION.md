# EXTENSION — cách tạo chênh lệch ở Task 2

Status: **accepted operating principle**; mọi claim chất lượng vẫn cần measured run.

## Execution fidelity trước performance diagnosis

Một score chỉ có giá trị chẩn đoán cho phương pháp khi artifact có trace chứng minh đúng
data/model revision, adapter/checkpoint, config, code và decode. ZIP đúng format và điểm do
leaderboard trả về vẫn có thể là bằng chứng của một runtime khác với phương pháp dự kiến.

Trường hợp Task 2 ngày 03-09-2026 là ví dụ: operator báo METEOR `0,18`, ROUGE-L `0,30` cho
ZIP SHA-256 `bf9ac32c...`, nhưng audit runner cho thấy base Qwen được nạp mà LoRA adapter
không được nạp. Do đó không được suy ra Phase A, SFT hoặc retrieval là bottleneck. Gate phải
được kiểm theo thứ tự:

```text
submission contract
  -> method execution fidelity
  -> validation/scorer fidelity
  -> system/slice/case/component diagnosis
  -> one bounded change
```

Đây là trường hợp đặc biệt của nguyên lý upstream-first: trước khi tối ưu chất lượng, phải
chứng minh hệ thống đã thực thi đúng phương pháp cần đánh giá.

Từ 03-09-2026, private Kaggle batch được mở lại bằng ADR-T2-0006 sau xác nhận của operator
với BTC. Thay đổi compute venue không hạ bất kỳ evidence gate nào: payload/input/runtime
phải có hash, base revision được pin, adapter phải được nạp lại cho generation, validation
đi trước Public và output phải qua local replay validator.

## Kết luận điều hành

Khi các đội có quyền dùng các backbone mạnh tương tự, model ID đơn lẻ không còn là
lợi thế bền. Backbone vẫn ảnh hưởng ceiling; nhưng lợi thế đội kiểm soát được chủ
yếu đến từ việc hiểu distribution, scorer và failure mode của chính DSC, rồi biến
hiểu biết đó thành data contract, split, retrieval/context, training và decoding
được kiểm chứng. Chênh lệch có khả năng đến từ năm lớp theo thứ tự:

1. hiểu đúng scorer và submission contract;
2. phát hiện cấu trúc, lỗi và leakage của dữ liệu sớm hơn;
3. thiết kế split phản ánh private test tốt hơn;
4. huấn luyện/decoding theo failure mode thay vì thử model ngẫu nhiên;
5. tái lập và submit ổn định, không mất điểm vì lỗi kỹ thuật.

Kiến trúc đột phá hợp lệ ở đây không nhất thiết là thêm module. Một training system
nhỏ nhưng có data lineage, length-aware batching, answer-only loss, metric-aware
error analysis và candidate selection ổn định có thể mạnh hơn pipeline nhiều model
không giải thích được.

Trạng thái 01-09-2026 không làm thay đổi nguyên lý này: preprocessing release,
tokenizer audit và checkpoint E0 đã có; submission ZIP Public đã pass format nhưng
method provenance fail vì runner lịch sử không chứng minh đã nạp adapter. Validation
predictions, METEOR/ROUGE-L, slice/case metrics và scorer parity vẫn thiếu. Vì vậy
first **valid quality result** chưa tồn tại. Bước tiếp theo là E0 validation canonical,
không phải mở E1 theo cảm giác. Dense chỉ chẩn đoán trước E2b hybrid
BM25+dense/RRF; citation graph tiếp tục `OPTIONAL_DISABLED`.

## Nguyên lý triển khai

> Tối đa hóa mức tăng end-to-end đã đo trên mỗi đơn vị phức tạp, VRAM, latency và
> rủi ro debug; chỉ thêm cấu trúc khi failure artifact chỉ ra cấu trúc đó cần thiết.

`Simple-first` không có nghĩa là giữ hệ thống đơn giản vĩnh viễn. Nó là thứ tự mở
complexity có kiểm soát:

1. chạy vòng nhỏ nhất từ input tới metric và giữ nó làm control;
2. khóa data/split/scorer/generator rồi chỉ đổi một biến chính;
3. mọi con số config phải mang nhãn `official`, `measured`, `paired-result` hoặc
   `hypothesis`; default từ README chỉ là hypothesis;
4. tune bottleneck hiện tại bằng coarse-to-fine grid nhỏ, không sweep tích Descartes
   của hàng chục knob;
5. promote bằng paired METEOR/ROUGE-L và case evidence; retrieval proxy chỉ chẩn đoán;
6. giữ trace và rollback một bước. Component hợp lệ nhưng không thắng là
   `REJECT|REPEAT`, không phải Error.

Với RAG, có thể dùng phân rã chẩn đoán gần đúng:

$$
P(\text{answer tốt}) \approx
P(\text{evidence được retrieve})
\times P(\text{evidence sống qua packing})
\times P(\text{generator dùng đúng})
\times P(\text{output hợp scorer}).
$$

Phân rã này không phải công thức chấm của BTC; nó dùng để định vị bottleneck trước
khi thêm model hoặc mở sweep.

## Framework từ bài toán tới performance engineering

Validation và observability là thành phần kiến trúc, không phải báo cáo thêm sau khi
code xong. Quy trình canonical gồm bốn lớp trước implementation và một vòng tối ưu
sau first valid result.

### 1. Problem formulation và system logic

Khóa trước: input, output, metric, data được phép, parameter/resource limit và
submission contract. Với Task 2:

```text
question + Task2 organizer data
    -> locate sufficient legal evidence khi profile cho phép
    -> generate Vietnamese answer
    -> METEOR primary, ROUGE-L secondary
    -> exact answer-only submission contract
```

Ở lớp này chưa chọn model, top-k hay threshold. Những giá trị đó chỉ là hypothesis
kỹ thuật sau khi objective và constraints đã rõ.

### 2. Technical formulation có validation và observability

Pipeline kỹ thuật phải đồng thời thiết kế đường dữ liệu và cách nhìn thấy lỗi:

```text
integrity -> preprocess -> leakage-safe split -> tokenizer/length audit
          -> index/retrieval -> context packing -> SFT/generation
          -> metric + trace + slice/error report -> submission replay
```

Validation cố định là group-aware, không random theo row. Ngoài overall metric phải
có slice tối thiểu:

- answer ngắn/trung bình/dài;
- có/không có cấu trúc Điều/Khoản;
- list/paragraph, multiline/single-line;
- loại câu hỏi như condition, procedure, penalty, authority, definition;
- với retrieval: HIGH-confidence judged subset, evidence hit/miss và rank bucket.

Mỗi validation case phải nối được `question_id -> prediction -> reference -> metric
-> generation trace -> retrieval IDs/ranks/scores -> length/truncation -> error
labels`. Text nhạy cảm chỉ nằm trong run local; submission không chứa trace.

### 3. Technical considerations và failure taxonomy

Trước first run phải dự đoán và gắn signal cho các failure có thể xảy ra:

| Lớp | Failure cần thấy được |
|---|---|
| Data | Unicode/newline bị đổi, duplicate leakage, mất Điều/Khoản, wrong split |
| Tokenizer | target bị cắt, context chiếm hết budget, p95/p99 vượt cấu hình |
| Retrieval | miss, đúng document sai section, rank thấp, duplicate/noisy context |
| Packing | evidence đúng bị loại/cắt/đảo thứ tự, budget lãng phí |
| Training | answer mask sai, NaN/OOM, overfit, batch padding quá lớn |
| Generation | empty, repeat, hallucinated number/article, thiếu kết luận, verbosity |
| Evaluation | scorer drift, missing/extra ID, sai encoding hoặc serialization |

Automatic labels chỉ là `HEURISTIC_REVIEW_REQUIRED`, không phải kết luận pháp lý.
Reviewer dùng case table để xác nhận bottleneck trước khi đổi component.

### 4. Implementation tới first valid result

Mục tiêu đầu tiên là một vòng replay được, không phải top score:

```text
release gate -> tokenizer gate -> train -> checkpoint -> validation generation
             -> provisional/exact scorer -> case/slice/error artifacts
             -> submission dry-run -> clean replay
```

First result chỉ hợp lệ khi loss hữu hạn, không OOM/NaN, đủ ID và answer không rỗng,
trace/resource log có mặt, scorer status rõ và ZIP replay pass. Smoke/fixture không
được đổi tên thành trained result hoặc official score.

### 5. Performance engineering sau first result

Vòng tối ưu bắt buộc:

```text
RESULT -> OBSERVE -> DIAGNOSE -> HYPOTHESIS -> ONE MAIN CHANGE
       -> PAIRED COMPARE -> PROMOTE | REJECT | REPEAT -> NEXT BOTTLENECK
```

Observation diễn ra ở bốn mức: system (metric/latency/VRAM/failure), slice, case và
component (retrieval survival/packing/generator use). Diagnosis phải chỉ ra lớp chịu
trách nhiệm. Ví dụ Recall@20 cao nhưng Recall@5 sau rerank thấp thì không đổi
generator; evidence có mặt nhưng answer sai mới mở hypothesis training/prompt.

Mỗi run challenger phải ghi parent, evaluation role, hypothesis, expected failure,
một biến chính, stop condition, hashes, paired wins/ties/losses, slice delta, 20 gain
lớn nhất, 20 regression/counterexample lớn nhất và quyết định. Average metric tăng
nhưng regression chưa giải thích thì giữ `REVIEW_REQUIRED`.

### 6. Chuỗi suy luận dùng cho mọi config

```text
internal observation
    -> evidence-backed bottleneck
    -> bounded hypothesis
    -> coarse-to-fine candidate values
    -> paired validation
    -> case/slice regression review
    -> promote hoặc rollback
```

Do đó `top_k=5`, `max_new_tokens=1024`, LR hay LoRA rank không phải chân lý. Mỗi giá
trị phải mang provenance `official|measured|paired-result|hypothesis`; Public Test
chỉ là check ngoài có chủ đích, không thay validation nội bộ.

## Điều đã kiểm chứng về Precision/Recall

Không có bằng chứng công khai để nói leaderboard hiện có Recall cao, Precision thấp
hay ngược lại. Endpoint kết quả chi tiết của Codabench trả HTTP 403 khi chưa đăng
nhập, nên mọi mô tả hành vi của đội khác là suy đoán.

Task 1 thực sự có Macro Recall chính và Macro Precision tie-break. Vì chỉ được trả
tối đa 5 ID, tăng k tạo trade-off quen thuộc giữa độ phủ và độ chính xác.

Task 2 không công bố hai cột Precision/Recall riêng; metric là METEOR chính và
ROUGE-L phụ. Tuy vậy, METEOR căn chỉnh unigram có thành phần precision/recall và
phạt alignment bị phân mảnh; ROUGE-L dựa vào longest common subsequence. Suy luận
cần kiểm chứng:

- answer quá ngắn có thể precision cao nhưng bỏ thiếu cụm tham chiếu;
- answer quá dài có thể phủ nhiều từ hơn nhưng thêm token không khớp và boilerplate;
- đúng ý nhưng đảo thứ tự, paraphrase mạnh hoặc mất cấu trúc có thể hại ROUGE-L;
- cắt phần cuối có thể mất điều kiện/ngoại lệ quan trọng và hại cả hai metric.

Không tối ưu theo suy luận này cho tới khi scorer implementation và case-level
experiment xác nhận.

Prior art COLIEE 2025 củng cố cách làm multi-stage, không cấp một recipe cho DSC.
Overview báo cả tám đội Task 1 dùng biến thể multi-stage retrieval, thường gồm
candidate retrieval, neural reranking và post-processing. Một paper hệ thống cũng
thử các candidate pool khác nhau và hard-negative reranking. Đây là bằng chứng để
tune system theo dữ liệu; task, metric và corpus khác nên không chuyển threshold,
top-k hoặc score sang DSC. Xem [COLIEE overview](https://link.springer.com/article/10.1007/s12626-026-00199-9)
và [Hybrid Legal Reasoning](https://link.springer.com/article/10.1007/s12626-026-00208-x).

## Data insights đang có

Audit `measured-local` trên 7.000 train records có SHA-256
`2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988` cho thấy
answer min/p50/p95/max là 129/1.410/3.143/10.755 ký tự; 6.961/7.000 (99,4%) có
xuống dòng và 6.230/7.000 (89,0%) chứa mẫu Điều/Khoản/Nghị định. Nguồn và phương
pháp audit nằm tại
[`organizer-contract-and-data-audit.md`](../../References/Task2/organizer-contract-and-data-audit.md).
Các số warm-up chỉ còn vai trò lịch sử, không thay audit full train.

Các hypothesis ưu tiên:

| ID | Hypothesis | Thí nghiệm tối thiểu | Dấu hiệu promote |
|---|---|---|---|
| H1 | Giữ newline/danh sách giúp ROUGE-L | raw format vs flatten | paired ROUGE-L tăng, METEOR không giảm |
| H2 | Answer-only loss tốt hơn full-sequence loss | cùng seed/config | cả hai metric và valid output tăng |
| H3 | Truncation làm mất kết luận/ngoại lệ | max token grid theo bucket | p95 dài tăng mà ngắn không regression |
| H4 | Group-aware split giảm optimistic leakage | random vs grouped | score giảm hợp lý, private/public ổn định hơn |
| H5 | Legal base cần SFT nhưng có transfer tốt | ViLegalQwen3-1.7B vs Qwen2.5-1.5B | thắng trên legal-structure slices |
| H6 | Verbosity calibration tăng cả hai metric | length penalty/max tokens grid | Pareto gain METEOR và ROUGE-L |

## Knob registry tối thiểu

Không đưa mọi knob vào một sweep. Mỗi vòng chỉ chọn nhóm gắn với bottleneck đã đo:

| Lớp | Knob ưu tiên | Artifact quyết định |
|---|---|---|
| Data/split | newline, group rule, duplicate threshold | data/leakage/split report |
| Index/retrieval | legal boundary, BM25 k, dense k, RRF weights | qrels coverage, Recall@k, rank trace |
| Context | fixed-k, ordering, dedup, token budget | evidence-survival và truncation report |
| Generator | max input/output, deterministic decode, repetition | case-level METEOR/ROUGE-L và length slices |
| Training | LR, epoch, effective batch, LoRA, answer mask | held-out paired delta và resource log |

Thứ tự retrieval giữ nguyên: E1 BM25; E2a dense-only để chẩn đoán; E2b
BM25+dense/RRF để promote. Reranker, dynamic-K và citation graph không cùng xuất
hiện trong first hybrid run.

## Model ladder hợp lệ

### Anchor

Qwen/Qwen2.5-1.5B-Instruct: tạo vòng đầu nhanh, khoảng 1,54B tham số và đã
instruction-tuned.

### Challenger chuyên ngành

ViLegalQwen3-1.7B-Base: khoảng 1,72B tham số, continual pretraining trên tiếng Việt
pháp luật. Đây là base model, không dùng zero-shot như instruction model; phải SFT
Task 2 rồi mới so.

### Challenger kiến trúc

VietAI/vit5-base rồi vit5-large nếu tài nguyên cho phép. Encoder-decoder có thể có
lợi thế khi mapping question-to-long-answer, nhưng phải đo context/target truncation.

### Hold hoặc loại

- Các model thực tế từ 4 tỷ tham số trở lên bị blocked dù tên ghi 4B và dù quantize.
- PhoGPT-4B-Chat bị hold cho tới khi xác minh tổng tham số.
- Retrieval/reranker/embedding không tự động có ích cho Task 2; E1 BM25 trên corpus
  Task 2 phải thắng E0 direct control trước khi thêm neural component. Mọi thành
  phần neural đều ăn vào tổng parameter budget.
- Nếu mở neural retrieval, dense-only chỉ là diagnostic; target promote đầu tiên là
  hybrid BM25+dense/RRF. Citation graph optional disabled cho tới gate ADR-T2-0004.
- Không ensemble ở runtime nếu tổng các thành phần vượt giới hạn.

Danh sách và parameter audit canonical nằm tại configs/Shared/model_allowlist.json.

## Training architecture nên xây

    immutable Task2 raw
      -> QA release: schema/group split
      -> tokenizer audit: optional handoff, mandatory control trước training
      -> corpus release: selected-contexts/parser/chunks/provenance
      -> E0 answer-only SFT direct control
      -> optional E1 Task2-only BM25 evidence
      -> deterministic generation
      -> exact local scorer
      -> case/slice error table
      -> submission validator + ZIP replay

Mỗi cạnh phải có checksum hoặc version. Đây là kiến trúc cạnh tranh vì nó rút ngắn
feedback loop và loại silent failure, không phải vì nhiều service.

## Chiến lược thí nghiệm

Mỗi run ghi:

- parent run và một biến thay đổi;
- hypothesis, expected failure mode và stop condition;
- data/split/model/tokenizer/revision/config/code/seed;
- METEOR, ROUGE-L, metric theo slice, latency, peak VRAM và failure count;
- paired case delta, best/worst cases và counterexample;
- quyết định PROMOTE, REJECT hoặc REPEAT.

Ưu tiên Pareto improvement. Nếu METEOR tăng nhưng ROUGE-L giảm, chỉ promote khi
metric chính tăng ổn định qua seed/fold và regression đã được giải thích.

## Các nguồn giúp tiếp tục tạo insight

### Nguồn chính thức

- Thông báo BTC, Codabench phase/metric/submission contract và scorer bundle.
- Organizer train/public test chỉ dùng đúng phạm vi được phép.
- Public submission history của chính đội: score, config và hash; không suy từ
  leaderboard không truy cập được.

### Prior art chỉ để tạo hypothesis

- [METEOR](https://aclanthology.org/W05-0909/): alignment, precision/recall và
  fragmentation penalty.
- [ROUGE](https://aclanthology.org/W04-1013/): LCS/sequence overlap.
- [ViLegalLM](https://aclanthology.org/2026.findings-acl.1801/): Vietnamese legal
  continual pretraining và các base checkpoint 1.5B/1.7B.
- [ViT5](https://aclanthology.org/2022.naacl-srw.18/): Vietnamese text-to-text.
- [QLoRA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html):
  tối ưu bộ nhớ fine-tuning, không thay đổi luật đếm tham số.
- [COLIEE 2025 overview](https://link.springer.com/article/10.1007/s12626-026-00199-9):
  multi-stage legal retrieval; không chuyển metric hoặc threshold sang DSC.

Không đưa corpus hoặc synthetic datasets của prior art vào train DSC.

### Nguồn nội bộ có giá trị nhất

- tokenizer_report và truncation samples;
- duplicate/leakage graph;
- case-level metric diff;
- gradient/VRAM/throughput profile;
- error taxonomy có reviewer;
- rejected-run ledger.

Đây là phần AI không tự cung cấp nếu không có dữ liệu và kỷ luật thực nghiệm của đội.

## Câu hỏi đột phá cần trả lời bằng experiment

1. Private-like split nên nhóm theo normalized question, answer structure hay legal
   topic để tương quan Public tốt nhất mà không dùng public labels?
2. Phần nào của long answer đóng góp nhiều nhất cho METEOR/ROUGE-L: định nghĩa,
   điều kiện, danh sách hay kết luận?
3. Model đang thiếu nội dung hay chỉ sai độ dài/thứ tự/format?
4. ViLegal continual pretraining giúp slice nào và làm hại instruction following nào?
5. Có thể chọn max_new_tokens theo bucket câu hỏi bằng rule học từ Task 2 mà không
   tạo thêm model/tham số không?
6. Checkpoint nào nằm trên Pareto frontier của metric, VRAM và latency?

Mỗi câu phải kết thúc bằng data artifact và quyết định, không chỉ bằng nhận xét.

## Stop rules

- Không thêm model mới khi scorer parity chưa đạt.
- Không chạy sweep lớn khi chưa có first trained end-to-end run.
- Không thay nhiều hơn một biến chính mỗi comparison.
- Không sweep nhiều chiều khi chưa xác định bottleneck và budget run.
- Không dùng Public Test như validation loop dày.
- Không triển khai API/agent/UI. Không mở E1 retrieval trước ADR/corpus gate hoặc
  dense/reranker/graph trước paired blocker evidence.
- Không gọi một run là tốt hơn nếu chỉ train loss giảm nhưng metric/case evidence
  không cải thiện.

## Bổ sung 02-09-2026 — measurement envelope và causal debugging

Validation, observability và error taxonomy phải được thiết kế trước run dùng để ra quyết
định. Mỗi metric phải có population, denominator, profile, implementation hash và evidence
status; giá trị không áp dụng ghi `not_applicable`, không ghi zero. Aggregate metric chỉ là
điểm vào của quá trình quan sát:

```text
system -> slice -> case -> component trace
       -> earliest failed upstream gate
       -> one hypothesis -> one main change -> paired comparison
```

Phân biệt ba lớp bằng chứng:

- `DETERMINISTIC`: schema, checksum, missing ID, cap hit hoặc trace trực tiếp;
- `HEURISTIC_REVIEW_REQUIRED`: verbosity, omission, number/citation mismatch;
- `HUMAN_REQUIRED`: correctness pháp lý, faithfulness bất định và root-cause adjudication.

Nhãn quan sát không tự là nguyên nhân. Causal status phải đi qua
`OBSERVED -> HYPOTHESIS -> SUPPORTED|REFUTED|INCONCLUSIVE`. Hard gate về scorer,
leakage, provenance và submission được xử lý trước mọi priority score. Sau đó mới triage
theo frequency, impact, causal reach, actionability và confidence.

Agent-first chỉ nhằm giảm khối lượng đọc: đọc summary trước, mở một tập case bounded có
seed, đề xuất một hypothesis và giao validator độc lập kiểm. Không auto-promote, không
auto-rollback bằng ghi đè và không coi heuristic là kết luận pháp lý. Khi chưa có human
review, quyết định giữ `REVIEW_REQUIRED`.

Cơ sở prior art cho cách nhìn này gồm [WILDS](https://proceedings.mlr.press/v139/koh21a.html)
về evaluation dưới distribution shift,
[CheckList](https://aclanthology.org/2020.acl-main.442/) về giới hạn của aggregate
held-out metrics, [Slice-based Learning](https://papers.neurips.cc/paper_files/paper/2019/hash/351869bde8b9d6ad1e3090bd173f600d-Abstract.html)
về critical slices, [RAGChecker](https://arxiv.org/abs/2408.08067) về fine-grained RAG
diagnostics và [Hidden Technical Debt in ML Systems](https://proceedings.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
về dependency/configuration debt. Đây là prior art để thiết kế; chỉ artifact Task 2 hợp lệ
và paired local validation mới là bằng chứng promote của DSC.

## Bổ sung 02-09-2026 — thực nghiệm theo chi phí

“Một thay đổi chính” áp dụng cho từng candidate, không buộc mỗi round chỉ có một
candidate. Một round có thể screening một grid nhỏ của cùng scientific knob trên cùng
parent, nhưng từng candidate vẫn độc lập và mọi fixed/nuisance knob phải được khóa.

```text
local contract/component screen
  -> frozen probe
  -> disjoint confirmation
  -> full validation cho candidate thắng
  -> tích hợp tuần tự
```

Không coi hai knob ở hai component là tự động trực giao. Chỉ bundle sau khi từng thay đổi
đã thắng độc lập hoặc có ablation/factorial design ước lượng interaction. Wins/Losses chỉ
là diagnostic; promote cần paired metric delta, uncertainty, slice guardrail và regression
review. Dùng cache theo input/config hash để không chạy lại stage bất biến.

Phân tầng này lấy nguyên lý adaptive resource allocation từ
[Hyperband](https://jmlr.org/papers/v18/16-558.html), cách tách scientific/nuisance/fixed
parameter từ [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook),
cảnh báo confounding từ [NIST design of experiments](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3345.htm)
và paired significance từ [Dror et al.](https://aclanthology.org/P18-1128/). Chi tiết vận
hành nằm tại
[COMPUTE-AWARE-EXPERIMENT-STRATEGY.md](COMPUTE-AWARE-EXPERIMENT-STRATEGY.md).
