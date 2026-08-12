# Task 2 warm-up data

Expected local file: `warmup_Task2.json`.

Schema: object keyed by query ID; each value contains `question: string` and
`answer: string`. Answers are reference labels, not current legal advice. Organizer
data is ignored by Git. Track only this note and `manifest.json`.

Validate:

```powershell
uv run egta-warmup Task2 Data/Task2/warmup_data/warmup_Task2.json
```
