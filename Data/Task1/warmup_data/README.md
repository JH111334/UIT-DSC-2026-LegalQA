# Task 1 warm-up data

Expected local file: `warmup_Task1.json`.

Schema: object keyed by query ID; each value contains `question: string` and
`answer: list[document_id]`. Organizer data is ignored by Git. Track only this note
and `manifest.json`; update the checksum after an authorized source change.

Validate:

```powershell
uv run egta-warmup Task1 Data/Task1/warmup_data/warmup_Task1.json
```
