# MAP — đường chính để thi DSC 2026

## Mục đích và phân bổ

Project này tối ưu cho competition: kết quả đúng contract, chạy được, tái lập được
và cải thiện metric. Không dùng nó để trình diễn kiến trúc sản phẩm.

Phân bổ cho giai đoạn hiện tại:

- 85% Task 2: hai nhánh QA/corpus preprocessing, E0 generator control, E1 BM25
  Task2-only, scorer parity, error analysis và submission replay;
- 10% integration: đóng gói, checksum, seed, resource log, Codabench rehearsal;
- 5% Task 1: chỉ lưu contract và hướng tiền xử lý, không thực nghiệm thay phần việc
  của thành viên khác.

Hai công việc đầu tiên và chiếm nhiều thời gian nhất là tiền xử lý dữ liệu Task 2
và tạo first trained submission. Mọi model/architecture mới phải xếp sau hai mốc này.

Validation design và error/observability design phải được khóa trước first run:
group-aware split, slice registry, trace schema, failure taxonomy và paired report là
thành phần architecture. Performance engineering chỉ bắt đầu sau first valid result
theo vòng `observe -> diagnose -> hypothesis -> one change -> paired review`.

## Mốc 0 — khóa luật, scorer và dữ liệu

Hoàn thành trước bất kỳ train dài nào.

### 0.1 Contract

- Chỉ nhận artifact có task_id=Task2 và provenance từ gói BTC.
- Không dùng dữ liệu Task 1, dữ liệu ngoài hoặc data augmentation.
- Không gọi API; chỉ tải trọng số mở hợp lệ rồi train/inference trong môi trường
  đội kiểm soát.
- Tổng tham số mọi thành phần đang chạy nhỏ hơn 4 tỷ; không dùng quantization/LoRA
  để biện minh cho model vốn từ 4 tỷ trở lên.
- Khóa schema submission, UTF-8, đủ question_id, không field thừa ngoài contract.
- `selected-contexts` chỉ được dùng khi manifest xác định là Task 2. Task 1/shared
  corpus, index, runtime, qrels và checkpoint luôn bị reject.
- E0 direct-generation là control. E1 retrieval chỉ mở sau ADR-T2-0003 và corpus gate.

### 0.2 Scorer parity

METEOR và ROUGE-L có nhiều implementation/tokenization khác nhau. Trước tối ưu:

1. lấy scorer hoặc hướng dẫn chính xác của BTC;
2. pin package/version/config/tokenizer;
3. tạo 8–12 golden cases: exact match, thừa câu, thiếu câu, đổi thứ tự, xuống dòng,
   dấu câu, Điều/Khoản và answer rỗng;
4. so local score với Codabench trên một submission probe;
5. chỉ dùng local metric làm promotion gate sau khi parity đạt.

Nếu chưa lấy được scorer, ghi metric là provisional và không tinh chỉnh quá sát nó.

## Mốc 1 — hai nhánh tiền xử lý Task 2

### 1.1 Raw layer bất biến

- Lưu gói BTC ngoài Git, đặt quyền đọc phù hợp.
- Ghi filename, byte size, SHA-256, ngày nhận, task, split và schema.
- Không sửa raw tại chỗ. Mọi bản dẫn xuất có transform version và parent checksum.

### 1.2 Nhánh QA: schema và integrity audit

Validator phải fail nếu:

- thiếu hoặc trùng question_id;
- question/answer không phải chuỗi hoặc rỗng sau chuẩn hóa;
- bản ghi Task 1 lẫn vào Task 2;
- encoding không phải UTF-8 hoặc JSON/JSONL lỗi;
- split có ID giao nhau;
- output không thể truy ngược tới raw record.

Không tự động xóa bản ghi lạ. Xuất quarantine report rồi quyết định theo từng loại lỗi.

### 1.3 Chuẩn hóa tối thiểu

Giữ hai view:

- raw_text: nguyên văn để audit và serialization;
- model_text: Unicode NFC, thống nhất CRLF/LF và loại khoảng trắng vô nghĩa ở biên.

Không lower-case answer, không bỏ dấu tiếng Việt, không flatten xuống dòng, không
xóa số thứ tự, Điều, Khoản hoặc dấu câu nếu chưa có paired ablation. Warm-up cho
thấy gần như toàn bộ answer có xuống dòng và phần lớn có cấu trúc pháp lý.

### 1.4 Duplicate, near-duplicate và leakage

- Exact duplicate: nhóm theo hash của nội dung chuẩn hóa; giữ provenance của mọi ID.
- Near-duplicate: dùng đặc trưng lexical chỉ tạo từ Task 2 để nhóm câu gần nhau.
- Chia train/validation theo group, không chia ngẫu nhiên từng row.
- So n-gram overlap và normalized similarity giữa split; lưu top suspicious pairs.
- Không dùng public/private labels để chọn model.

### 1.5 Tokenization và length budget

Chạy tokenizer của từng candidate trên prompt và answer, báo p50/p90/p95/p99/max:

- input tokens;
- target tokens;
- tổng sequence;
- tỷ lệ bị cắt ở từng max_length;
- tỷ lệ mất phần cuối chứa Điều/Khoản hoặc kết luận.

Chọn max_length bằng coverage và VRAM đo được, không đoán từ số ký tự. Với mẫu quá
dài, ưu tiên gradient accumulation và length bucketing trước khi cắt target.

### 1.6 Split và data card

Tạo tối thiểu một validation cố định, group-aware và có các slice:

- độ dài answer;
- có/không xuống dòng, danh sách, Điều/Khoản;
- loại câu hỏi được gán bằng rule có kiểm tra mẫu;
- mức overlap question–answer;
- outlier và record từng bị quarantine.

### 1.7 Nhánh corpus `selected-contexts`

- Giữ raw bất biến; empty passage chỉ quarantine với `indexable=false`.
- Missing name chỉ tạo `name_derived` và `name_source`, không overwrite raw.
- Very-long document là audit flag, không phải lý do xóa.
- Exact duplicate có content group/alias provenance; near duplicate không auto-delete.
- Parser giữ `Document -> Chapter -> Section -> Article -> Clause -> Point`, source
  offsets và mọi `unparsed_span`.
- Retrieval chunk ưu tiên Article; Article dài tạo child clause/sentence và parent
  expansion có token budget.
- Giữ `raw_text`, `canonical_text` và `retrieval_text` riêng; citation canonical là
  metadata, không rewrite surface text.

### 1.8 Diagnostic bridge

Citation từ gold answer có thể match sang corpus metadata để tạo
`citation-derived diagnostic qrels`. Qrels phải ghi confidence, parser/matcher
revision, judged coverage và unjudged policy. Đây là proxy cho Recall@k/MRR/nDCG,
không phải gold hoặc input Public/Private; dùng nó để train retriever vẫn bị block
cho tới rule gate.

Đầu ra bắt buộc của Mốc 1:

- manifest.json;
- data_report.json;
- split_manifest.json;
- leakage_report.json;
- tokenizer_report theo model;
- processed train/validation có checksum;
- corpus_manifest.json, documents.jsonl, chunks.jsonl và corpus_report.json;
- citation_qrels.jsonl và citation_match_report.json với nhãn proxy;
- unit tests cho schema, transform idempotence và split disjointness.

Contract chi tiết và ma trận theo phase nằm ở
[BanGiao.md](../Decision-making/BanGiao.md).

## Mốc 2 — first trained submission trong 72 giờ sau khi có train

Mục tiêu không phải model tốt nhất; mục tiêu là một vòng hoàn chỉnh có thể replay.

### 2.1 Anchor

Ưu tiên Qwen/Qwen2.5-1.5B-Instruct làm anchor vì:

- khoảng 1,54 tỷ tham số, còn khoảng an toàn dưới giới hạn;
- đã instruction-tuned, phù hợp để tạo baseline sớm hơn base model;
- giấy phép/model revision phải được pin lại trước khi tải.

AITeamVN/Vi-Qwen2-1.5B-RAG không là anchor mặc định. E0 phải tách khỏi retrieval để
làm control; E1 bắt đầu bằng BM25 0 neural parameter trên corpus Task 2. Chỉ chọn
neural retriever sau allowlist/parameter audit và paired evidence.

### 2.2 Training recipe v0

- Một template prompt ngắn, versioned; không nhồi hướng dẫn pháp luật ngoài data.
- Supervised fine-tuning chỉ trên Task 2.
- Loss mask toàn bộ system/user prompt, chỉ học answer tokens.
- QLoRA/LoRA được dùng để vừa VRAM nhưng model danh nghĩa vẫn được tính đủ tham số.
- Bắt đầu một seed, epoch ngắn, deterministic validation và checkpoint nhỏ nhất đủ
  replay; lưu effective batch size, optimizer, scheduler, precision và peak VRAM.
- Greedy decoding hoặc temperature=0 làm anchor; pin max_new_tokens và stop rule.
- Không thêm retrieval, reranker, ensemble hoặc self-consistency trong E0.

### 2.3 Gate first run

Một run chỉ hợp lệ khi:

- train loss hữu hạn, không NaN/OOM;
- sample decode không rỗng và không lặp vô hạn;
- prediction đủ validation IDs, không có answer null;
- scorer local chạy;
- model/revision, code, config, data/split checksum và seed đầy đủ;
- submission dry-run tạo đúng một submission.json trong ZIP;
- inference replay từ môi trường sạch trên một sample cố định.

Đầu ra: checkpoint hoặc adapter, run manifest, predictions, metrics, resource log,
error table và candidate submission. Không gọi đây là competition-quality nếu chưa
có public/private evidence.

## Mốc 3 — nghiên cứu có thứ tự

Mỗi run chỉ đổi một biến chính và so paired trên cùng split.

| Thứ tự | Thí nghiệm | Câu hỏi |
|---:|---|---|
| E0 | Qwen2.5-1.5B answer-only SFT direct | First trained control có replay và valid output không? |
| E0.1 | target-format preservation | Giữ newline/danh sách có tăng ROUGE-L không? |
| E0.2 | length bucket và max token grid | Truncation/verbosity ảnh hưởng metric ra sao? |
| E0.3 | decoding grid nhỏ | Decoding có cải thiện cả hai metric không? |
| E1 | Task2-only BM25 + cùng generator E0 | Evidence có tăng end-to-end metric so với E0 không? |
| E2a | Dense-only diagnostic | Dense cứu semantic miss nào và tốn bao nhiêu tài nguyên? |
| E2b | BM25+dense/RRF hybrid | Hai tín hiệu bổ sung có thắng cả E1 và E2a end-to-end không? |
| E3 | Parent-child/metadata packing | Context hữu ích hơn trong cùng token budget không? |
| E4 | Reranker | Candidate recall đủ nhưng ranking sai có được sửa không? |
| E5 | Dynamic-k | Fixed-k có phải blocker đã đo không? |
| E6 | Citation graph optional | Chỉ relation/multi-hop miss sau hybrid có được cứu không? |
| G1 | ViLegalQwen3-1.7B-Base SFT | Domain pretraining thắng instruction anchor không? |

E2b là neural retrieval target đầu tiên. E2a bắt buộc để hiểu contribution nhưng
không thay hybrid. Citation graph mặc định disabled và chỉ mở theo ADR-T2-0004.

ViLegalQwen3-1.7B là base model, vì vậy phải post-train trước khi so công bằng.
ViLegalLM là prior art đáng thử vì pretraining tiếng Việt pháp luật, nhưng các
synthetic dataset của paper tuyệt đối không được nhập vào DSC.

Model có tên 4B nhưng số tham số thực tế từ 4 tỷ trở lên bị loại, gồm các ID đã
đánh dấu blocked trong model_allowlist.json. PhoGPT-4B-Chat bị hold cho tới khi có
parameter count đáng tin cậy.

## Mốc 4 — error analysis tạo khác biệt

Sau mỗi run, không chỉ xem trung bình. Xuất case-level table:

- METEOR, ROUGE-L, độ dài prediction/reference và tỷ lệ độ dài;
- missing key phrase, thừa boilerplate, đảo thứ tự, lặp, cắt cuối, answer rỗng;
- lỗi Điều/Khoản/số, phủ định, chủ thể, điều kiện và ngoại lệ;
- slice delta so với anchor;
- 20 best gains, 20 worst regressions và counterexample cho hypothesis.

Lợi thế cạnh tranh nằm ở vòng lặp:

data audit → hypothesis hẹp → paired run → case analysis → fix đúng failure mode.

Không mở thêm model khi chưa giải thích được regression của model hiện tại.

## Mốc 5 — Public Test rồi mới Private Test

Không thể “up thẳng private” trước khi phase Private mở. Quy trình an toàn:

1. dùng validation local để chọn ít candidate;
2. submit Public Test có chủ đích, ghi submission hash và config;
3. không hill-climb mù theo public score;
4. chọn candidate bằng local robustness và public evidence;
5. freeze code/model/config trước 18-09-2026;
6. khi Private mở, chỉ submit artifact đã clean replay, không sửa khẩn cấp thiếu test.

Submission rehearsal phải kiểm:

- ZIP chỉ có submission.json;
- UTF-8, đúng key, đủ ID, không duplicate/missing;
- deterministic serialization và hash;
- inference command từ README tạo lại cùng output;
- không có raw data, token, cache, secret hoặc artifact Task 1 trong package.

## Task 1 — phần lưu để bàn giao

Task 1 train có passage rỗng/trùng nên owner Task 1 cần raw checksum, empty flag,
duplicate grouping và group-aware split. Anchor vẫn là lexical rồi hybrid/reranker.
Metric chính thức là Macro Recall, Precision tie-break, tối đa 5 ID. Không dùng dữ
liệu hay checkpoint Task 1 cho Task 2.

## Definition of done

Project competition hoàn thành khi:

- preprocessing và split Task 2 có manifest, audit, checksum và test;
- corpus Task 2 có manifest/chunks/provenance; nếu E1 được mở thì index/trace tách
  khỏi Task 1 và paired comparison với E0;
- có ít nhất một trained model hợp lệ chạy end-to-end;
- local scorer parity được chứng minh hoặc hạn chế được ghi rõ;
- error analysis dẫn đến ít nhất một cải tiến paired được tái lập;
- submission Public và candidate Private replay được từ môi trường sạch;
- README, model/data card, license/parameter audit và resource log khớp code;
- không có API, dữ liệu ngoài, augmentation, cross-task leakage hoặc model tổng từ
  4 tỷ tham số trở lên.
