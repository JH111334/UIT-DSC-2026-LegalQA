"""Test the public answer command."""

import json
from pathlib import Path

import pytest
from text_retrieval_agent.cli import main

ROOT = Path(__file__).parents[2]


def test_cli_answer_emits_cited_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Emit an evidence-grounded answer through the installed command."""
    exit_code = main(
        [
            "--corpus",
            str(ROOT / "Data/Shared/fixtures/documents.jsonl"),
            "--config",
            str(ROOT / "configs/Shared/smoke.toml"),
            "ask",
            "--text",
            "How should search enforce access scope?",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert not payload["abstained"]
    assert payload["citations"][0]["document_id"] == "doc_access_scope"
