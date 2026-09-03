# ADR-T2-0004: Hybrid retrieval trước citation graph

Status: **accepted — khóa thứ tự nghiên cứu, không cấp quyền triển khai model**  
Ngày chốt: 29-08-2026  
Phạm vi: Task 2 LegalQA, corpus chính thức `selected-contexts`.

## Nguyên lý quyết định

Ưu tiên thành phần tạo **mức tăng end-to-end kỳ vọng lớn nhất trên mỗi đơn vị phức
tạp và rủi ro**, với điều kiện không vi phạm contract cuộc thi. Diễn đạt vận hành:

> Giữ baseline có đối chứng, kết hợp tín hiệu bổ sung có thể trace, và chỉ thêm cấu
> trúc mới khi một failure mode đã đo chứng minh cấu trúc đó cần thiết.

Đây là heuristic nội bộ, không phải công thức chấm của BTC. Một component chỉ được
promote khi paired METEOR/ROUGE-L trên cùng data, generator, prompt budget và decode
tăng đủ ổn định; retrieval metric đơn lẻ chỉ dùng chẩn đoán.

## Bằng chứng và giới hạn chuyển giao

| Lớp bằng chứng | Phát hiện | Điều được phép kết luận |
|---|---|---|
| Official-local: Data Overview Task 2 | Bài toán yêu cầu truy xuất văn bản liên quan và tạo câu trả lời; `selected-contexts` là kho ngữ cảnh/căn cứ | Retrieval Task2-only là nhánh kiến trúc hợp lý sau provenance gate; tài liệu không chỉ định BM25, dense, graph hay thứ tự tối ưu |
| Prior art: [ViDRILL](https://aclanthology.org/2025.vlsp-1.17/) | Pipeline tìm kiếm pháp luật tiếng Việt kết hợp BM25, dense retrieval và reranker | BM25 và dense có failure mode bổ sung; hybrid là challenger đáng benchmark, không phải bằng chứng sẽ tăng METEOR của DSC |
| Prior art: [VLSP MLQA-TSR overview](https://aclanthology.org/2025.vlsp-1.48/) | Benchmark đa phương thức có retrieval labels, F2 và QA Accuracy | Chỉ chuyển pattern retrieval; không chuyển metric, label hoặc kết quả sang DSC |
| Prior art: [graph-based MLQA-TSR](https://aclanthology.org/2025.vlsp-1.49/) | Paper báo cáo hệ thống dùng graph dị thể nối article, image và table xếp hạng 2 với F2 0,611 trong traffic-law multimodal retrieval | Paper không cô lập hiệu ứng nhân quả của graph; task/corpus/metric khác nên graph chỉ là hypothesis cho DSC text LegalQA |
| Measured-local 31-08-2026 | Release canonical có 8.532 document và 316.100 chunk; deep E1 audit PASS, chưa có index/hybrid benchmark và 150 manual citation audit còn pending | Đủ data contract để đề xuất build E1 sau approval; vẫn không có evidence mở citation graph hoặc tuyên bố graph benefit |
| Policy local | Neural retriever phải có Task 2 role/allowlist và tổng runtime dưới 4B | Chưa được chọn hoặc tải dense model trong ADR này |

HyperAI và Papers with Code archive không trả về exact discovery record hữu ích cho
quyết định này; bằng chứng kỹ thuật được xác minh lại ở nguồn sơ cấp ACL Anthology.

## Quyết định

1. Giữ `E0` direct SFT làm control và `E1` BM25 Task2-only làm sparse control.
2. Khi E1 chỉ ra semantic miss và model policy pass, chạy `E2a` dense-only như
   **diagnostic ablation** để đo phần semantic được cứu và chi phí neural.
3. `E2b = BM25 + dense -> RRF` là **neural retrieval challenger chính đầu tiên**.
   RRF là fusion anchor vì không yêu cầu hai score scale đã calibrated; fusion khác
   chỉ được thử bằng paired ablation.
4. Không promote dense-only thành kiến trúc đích trước khi so trực tiếp với E1 và
   E2b. Mọi hybrid run phải giữ component ranks/scores và trace provenance.
5. Citation graph ở trạng thái `OPTIONAL_DISABLED`. Không tạo graph schema, graph
   index, graph dependency hoặc graph runtime trong baseline/hybrid implementation.
6. Parent-child expansion, metadata packing và reranker vẫn là các ablation độc lập;
   chúng không tự cấp quyền mở graph.
7. Citation graph chỉ được đề xuất lại sau khi hybrid đã hợp lệ và error table chỉ
   ra lỗi quan hệ/multi-hop mà BM25+dense/reranker không giải quyết được.

## Gate mở citation graph

Tất cả điều kiện sau phải đồng thời đạt:

- E2b hybrid đã có run replay được và là parent checkpoint được chấp nhận;
- corpus parser/citation matcher có version, precision/coverage audit và edge
  provenance; unmatched/ambiguous relation không bị biến thành edge chắc chắn;
- slice quan hệ pháp lý/multi-document được định nghĩa trước, có denominator đủ và
  cho thấy hybrid miss có hệ thống;
- graph expansion có token/candidate budget, latency/memory budget và deterministic
  trace;
- paired ablation `hybrid` so với `hybrid + graph` tăng end-to-end METEOR, không gây
  regression không giải thích được về ROUGE-L/output integrity;
- rollback chỉ cần tắt graph, không thay corpus release hoặc generator parent.

Thiếu một điều kiện thì quyết định là `REJECT|REPEAT`, không tích hợp graph.

## Hệ quả thực thi

- Scaffold executable hiện vẫn chỉ có `e0-direct` và `e1-bm25`.
- Không thêm FAISS, embedding model, reranker hoặc graph package ở lượt planning.
- Khi mở neural retrieval, dense-only và hybrid dùng cùng corpus/split/query set để
  giữ comparison hợp lệ; hybrid là điểm promote, dense-only là diagnostic parent.
- Không hứa thứ hạng. Lợi thế cạnh tranh đến từ vòng lặp nhỏ, trace được, rollback
  được và tối ưu trực tiếp theo metric Task 2.

## Quan hệ ADR

- ADR-T2-0002 giữ E0 direct-generation và cấm cross-task retrieval.
- ADR-T2-0003 quyết định điều kiện dùng corpus Task 2; vẫn chờ acceptance gate.
- ADR này chỉ khóa **thứ tự và stop rule** của retrieval challengers.
