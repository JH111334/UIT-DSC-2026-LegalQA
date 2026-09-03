# Quy tắc Documents/General — DSC-TinDipLaPo

## Phạm vi và tên file

General chỉ chứa RULE.md và hai họ log bất biến:

- DSC-TinDipLaPo_Checkpoints_DD-MM-YY_Vn.md
- DSC-TinDipLaPo_Error_DD-MM-YY_Vn.md

DD-MM-YY thay DD/MM/YY vì Windows dùng dấu gạch chéo làm ký tự đường dẫn. Vn bắt
đầu từ V1, tăng theo ngày và loại; không ghi đè hoặc sửa log cũ. Tri thức, ADR,
hướng dẫn, meeting note và báo cáo thuộc Decision-Making, References hoặc Docs.

## Checkpoint tối thiểu

Ghi project, ngày/version, owner, Git/worktree, system profile, lệnh/exit code,
data-corpus/config/model/code revision, seed, evidence class, artifact/hash, kết
quả, giới hạn, replay/rollback và next gate. Thiếu trường nào ghi không thu thập,
không suy đoán.

Với run so sánh, khóa anchor_checkpoint, parent_checkpoint, hypothesis_id,
evaluation_role=development|error_analysis|sealed_test, gold_status và
evaluation_contract_hash. Contract bao gồm data manifest, split, gold/qrels,
evaluator, metric, parser và slice registry. Khác contract hoặc đổi nhiều biến
không tách được phải ghi comparison=confounded.

Báo paired delta theo case/query/group so với anchor và parent: overall, slice,
denominator, seed/CI và wins|ties|losses. Artifact chi tiết giữ output/rank/score
cũ-mới, slice ID, gold/proxy, invalid/abstention và error category; log chỉ dẫn
URI/hash.

## Học từ hành vi dữ liệu

Khóa case budget và quy tắc chọn trước run: regression lớn nhất, ca được cứu, lỗi
dai dẳng, low-margin/high-variability, counterexample và mẫu random có seed trên
lát cắt trọng yếu. Slice khởi điểm: intent, lexical-semantic gap, length, domain, judged coverage, hard-negative, citation và abstention. Mỗi slice phải có definition,
revision, count và metadata; slice post-hoc là exploratory cho tới run xác nhận.

Tách observation khỏi cause. Kết luận hypothesis bằng
supported|refuted|inconclusive, kèm dự đoán có thể bác bỏ, minimal next experiment
và stop/rollback gate. Gold partial phải ghi judged coverage và unjudged policy.
Gold absent chỉ cho phép quality, stability, behavioral test hoặc human audit;
pseudo-label, synthetic label và LLM judge là proxy, không phải gold.

## Error

Error chỉ dành cho run không hợp lệ hoặc quality gate định trước chuyển FAIL.
Ghi checkpoint liên quan, first_bad, last_known_good, invariant, stage, ca/lát
cắt, expected/actual, reproducer, log đã lọc secret, bằng chứng ủng hộ/phản bác
nguyên nhân, containment/rollback, regression và residual risk.

Run hợp lệ nhưng metric kém hơn là Checkpoint decision=REJECT|REPEAT. Giả thuyết
bị bác bỏ không tự tạo Error.

## Ngôn ngữ và bảo mật

Viết tiếng Việt có dấu, viết hoa đầu câu và dùng câu ngắn. Không chép debug
narrative; chỉ dẫn artifact đã lọc và có hash. Không ghi token, secret, PII,
organizer/private payload, model weights hoặc transcript dài.
