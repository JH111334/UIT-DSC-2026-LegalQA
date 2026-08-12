"""Test organizer warm-up validation without depending on private files."""

import json
from pathlib import Path

import pytest

from text_retrieval_agent.contracts import ContractError
from text_retrieval_agent.warmup import (
    load_task1_warmup,
    load_task2_warmup,
    main,
    summarize_warmup,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_task1_warmup_preserves_query_and_ranked_labels(tmp_path: Path) -> None:
    """Validate Task 1 shape and preserve organizer identifiers."""
    path = _write_json(
        tmp_path / "task1.json",
        {"q-1": {"question": "Văn bản nào áp dụng?", "answer": ["doc-2", "doc-1"]}},
    )
    records = load_task1_warmup(path)
    summary = summarize_warmup(path, "Task1")
    assert records[0].query_id == "q-1"
    assert records[0].answer_document_ids == ("doc-2", "doc-1")
    assert summary.records == summary.unique_questions == 1
    assert len(summary.sha256) == 64


def test_task2_warmup_and_cli_emit_only_aggregate_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validate Task 2 and avoid printing private questions or answers."""
    path = _write_json(
        tmp_path / "task2.json",
        {"q-2": {"question": "Ai có thẩm quyền?", "answer": "Theo điều khoản được dẫn."}},
    )
    assert load_task2_warmup(path)[0].answer == "Theo điều khoản được dẫn."
    assert main(["Task2", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "Task2"
    assert payload["records"] == 1
    assert "thẩm quyền" not in str(payload)


@pytest.mark.parametrize(
    ("task", "payload"),
    [
        ("Task1", {}),
        ("Task1", {"q": {"question": "", "answer": ["doc"]}}),
        ("Task1", {"q": {"question": "Q", "answer": []}}),
        ("Task1", {"q": {"question": "Q", "answer": ["doc", "doc"]}}),
        ("Task2", {"q": []}),
        ("Task2", {"q": {"question": "Q", "answer": ""}}),
    ],
)
def test_invalid_warmup_contracts_fail(tmp_path: Path, task: str, payload: object) -> None:
    """Reject malformed or empty organizer records."""
    path = _write_json(tmp_path / "invalid.json", payload)
    with pytest.raises(ContractError):
        if task == "Task1":
            load_task1_warmup(path)
        else:
            load_task2_warmup(path)


def test_summary_rejects_unknown_task(tmp_path: Path) -> None:
    """Reject task names outside the two competition contracts."""
    path = _write_json(tmp_path / "payload.json", {"q": {}})
    with pytest.raises(ValueError):
        summarize_warmup(path, "Task3")
