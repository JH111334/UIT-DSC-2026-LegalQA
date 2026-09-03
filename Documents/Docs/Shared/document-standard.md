# Document standard

## Required shape

- One purpose per file.
- Outcome first; then evidence, decision, command, result, blocker.
- Distinguish `official`, `measured`, `prior-art`, `decision`, and `roadmap`.
- Cite a direct URL, file checksum, config version, and Git revision when applicable.
- No greetings, repeated background, debug transcript, decorative text, or unverified score.

## Formats

- Markdown is canonical and tracked.
- DOCX/PDF are delivery renders, local-only unless explicitly approved.
- Tables only for exact comparisons; diagrams only for non-trivial flow.
- Vietnamese uses full diacritics; English names and identifiers remain exact.

## DOCX gate

Build from reviewed Markdown; apply compact heading/table styles; render to PDF;
inspect pagination, clipped text, fonts, and tables before delivery. A generated file
without structural and rendered QA is not complete.

<!-- BEGIN research-refinement:v1 -->
## Chuẩn viết refine

Viết tiếng Việt có dấu, viết hoa đầu câu, dùng câu ngắn và nguồn gần mệnh đề. Không chép debug narrative. Tài liệu lịch sử và General logs bất biến; tạo version mới theo RULE.md.
<!-- END research-refinement:v1 -->
