"""Tests for dependency-light Task 2 fusion, batching, and packaging."""

from __future__ import annotations

import json
import os
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "Source" / "Task2"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from Online.evaluation import evaluate_run
from Online.inventory import inventory_pipeline
from Online.output_audit import audit_unlabeled_output
from Online.request import PipelineRequest
from Online.RetrievingAnswer.batch import (
    AnswerResult,
    BatchInferenceRunner,
)
from Online.RetrievingAnswer.bm25 import (
    BM25Index,
    build_bm25_index,
    evaluate_bm25,
)
from Online.RetrievingAnswer.fusion import reciprocal_rank_fusion
from Online.RetrievingAnswer.submission import (
    build_submission,
    package_submission,
    validate_predictions,
    validate_submission_zip,
)
from Online.Training.runtime import (
    configure_offline_transformers,
    encode_answer_only,
)


class FixtureEngine:
    """Return deterministic answers without loading an ML dependency."""

    def __init__(self) -> None:
        """Initialize a visible call counter for resume assertions."""
        self.calls = 0

    def __repr__(self) -> str:
        """Return a stable test representation."""
        return f"FixtureEngine(calls={self.calls})"

    def answer(self, question: str) -> AnswerResult:
        """Echo one fixture answer and an ID-only trace."""
        self.calls += 1
        return AnswerResult(answer=f"Trả lời: {question}", trace={"engine": "fixture"})


class FixtureBatchedEngine:
    """Expose deterministic micro-batching without model dependencies."""

    inference_batch_size = 2

    def __init__(self) -> None:
        self.batch_calls: list[tuple[str, ...]] = []

    def answer(self, question: str) -> AnswerResult:
        raise AssertionError(f"Single-answer fallback must not run: {question}")

    def answer_many(self, questions: Sequence[str]) -> tuple[AnswerResult, ...]:
        batch = tuple(questions)
        self.batch_calls.append(batch)
        return tuple(
            AnswerResult(answer=f"Trả lời: {question}", trace={"batch": len(batch)})
            for question in batch
        )


class FixtureTokenizer:
    """Expose a prefix-stable character tokenizer for answer-mask tests."""

    pad_token_id: int | None = 0
    eos_token_id: int | None = 1
    pad_token: str | None = "<pad>"
    eos_token: str | None = "<eos>"
    padding_side = "right"

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize
        text = "".join(f"<{row['role']}>{row['content']}" for row in conversation)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text: str, *, add_special_tokens: bool) -> Mapping[str, Sequence[int]]:
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def save_pretrained(self, save_directory: str | Path) -> object:
        del save_directory
        return ()


def _write_questions(path: Path, *, split: str = "public") -> None:
    records = [
        {
            "question_id": "q1",
            "source_task": "Task2",
            "question_model": "Câu hỏi một",
            "split": split,
        },
        {
            "question_id": "q2",
            "source_task": "Task2",
            "question_model": "Câu hỏi hai",
            "split": split,
        },
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_rrf_combines_bm25_and_dense_with_trace() -> None:
    """Hybrid RRF must reward shared high-ranked chunks deterministically."""
    result = reciprocal_rank_fusion(
        {
            "bm25": [("c1", 8.0), ("c2", 7.0)],
            "dense": [("c2", 0.9), ("c3", 0.8)],
        },
        constant=60,
        limit=3,
    )
    assert [item.chunk_id for item in result] == ["c2", "c1", "c3"]
    assert result[0].component_ranks == {"bm25": 2, "dense": 1}
    assert result[0].component_scores == {"bm25": 7.0, "dense": 0.9}


def test_batch_resume_and_submission_package(tmp_path: Path) -> None:
    """Batch inference must resume and produce an answer-only deterministic ZIP."""
    questions = tmp_path / "public.jsonl"
    predictions = tmp_path / "submission.json"
    trace = tmp_path / "trace.json"
    archive = tmp_path / "submission.zip"
    _write_questions(questions)
    engine = FixtureEngine()
    runner = BatchInferenceRunner(engine)

    first = runner.run(questions, predictions, expected_split="public", trace_path=trace)
    second = runner.run(
        questions,
        predictions,
        expected_split="public",
        trace_path=trace,
        resume=True,
    )

    assert first == {"questions": 2, "generated": 2, "complete": True}
    assert second == {"questions": 2, "generated": 0, "complete": True}
    assert engine.calls == 2
    assert validate_predictions(questions, predictions, expected_split="public")["status"] == "PASS"
    first_hash = package_submission(predictions, archive)
    with zipfile.ZipFile(archive) as value:
        assert value.namelist() == ["submission.json"]
    archive.unlink()
    assert package_submission(predictions, archive) == first_hash


def test_batch_runner_uses_bounded_engine_micro_batch(tmp_path: Path) -> None:
    """A batch-capable engine must receive questions in stable bounded groups."""
    questions = tmp_path / "public.jsonl"
    predictions = tmp_path / "predictions.json"
    trace = tmp_path / "trace.json"
    _write_questions(questions)
    engine = FixtureBatchedEngine()

    result = BatchInferenceRunner(engine).run(
        questions,
        predictions,
        expected_split="public",
        trace_path=trace,
    )

    assert result == {"questions": 2, "generated": 2, "complete": True}
    assert engine.batch_calls == [("Câu hỏi một", "Câu hỏi hai")]
    assert {row["batch"] for row in json.loads(trace.read_text(encoding="utf-8")).values()} == {2}


def test_batch_resume_rejects_mixed_decoding_config(tmp_path: Path) -> None:
    """Resume must not combine answers generated by different decode controls."""
    questions = tmp_path / "public.jsonl"
    predictions = tmp_path / "predictions.json"
    trace = tmp_path / "trace.json"
    _write_questions(questions)
    BatchInferenceRunner(FixtureEngine()).run(
        questions,
        predictions,
        expected_split="public",
        trace_path=trace,
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        BatchInferenceRunner(FixtureEngine()).run(
            questions,
            predictions,
            expected_split="public",
            trace_path=trace,
            resume=True,
            resume_trace_requirements={"decode_id": "different"},
        )


def test_answer_only_mask_keeps_prompt_as_context() -> None:
    """Only assistant answer characters may contribute to causal loss."""
    encoded = encode_answer_only(
        FixtureTokenizer(),
        "Câu hỏi",
        "Câu trả lời",
        max_sequence_length=4096,
    )
    assert encoded.prompt_tokens > 0
    assert encoded.target_tokens > 0
    assert encoded.labels[: encoded.prompt_tokens] == (-100,) * encoded.prompt_tokens
    assert encoded.labels[encoded.prompt_tokens :] == encoded.input_ids[encoded.prompt_tokens :]
    assert encoded.truncated_target_tokens == 0


def _write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "source_task": "Task2",
            "parent_chunk_id": "p1",
            "retrieval_text": "điều kiện bán lẻ rượu",
            "parent_text": "Điều 13. Điều kiện bán lẻ rượu.",
            "doc_number_canonical": "105/2017/NĐ-CP",
            "article": "13",
            "indexable": True,
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "source_task": "Task2",
            "parent_chunk_id": "p2",
            "retrieval_text": "kiểm dịch động vật",
            "parent_text": "Điều 17. Kiểm dịch động vật.",
            "doc_number_canonical": "90/2017/NĐ-CP",
            "article": "17",
            "indexable": True,
        },
    ]
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_bm25_build_search_and_proxy_metrics(tmp_path: Path) -> None:
    """E1 must index only Task2 chunks and retain diagnostic-only metrics."""
    release = tmp_path / "release"
    run = tmp_path / "run"
    chunks = release / "corpus" / "chunks.jsonl"
    _write_chunks(chunks)
    (release / "corpus" / "corpus_manifest.json").write_text("{}", encoding="utf-8")
    (release / "manifest.json").write_text(
        json.dumps({"release_id": "task2-data-v3"}), encoding="utf-8"
    )
    manifest = build_bm25_index(
        release,
        run,
        index_id="task2-data-v3-bm25-v1",
        manifest_status="CANDIDATE",
    )
    assert manifest["indexed_chunks"] == 2
    assert manifest["status"] == "CANDIDATE"
    assert manifest["release_id"] == "task2-data-v3"
    assert manifest["sqlite_integrity_check"] == "ok"
    index = BM25Index(run / "index" / "bm25.sqlite3")
    assert index.search("Điều kiện kinh doanh bán lẻ rượu", limit=1)[0].chunk_id == "c1"

    questions = release / "qa" / "validation.jsonl"
    questions.parent.mkdir(parents=True)
    questions.write_text(
        json.dumps({"question_id": "q1", "question_model": "điều kiện bán lẻ rượu"}) + "\n",
        encoding="utf-8",
    )
    qrels = release / "diagnostics" / "citation_qrels.jsonl"
    qrels.parent.mkdir(parents=True)
    qrels.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "chunk_id": "c1",
                "confidence": "HIGH",
                "qrels_status": "diagnostic_proxy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metrics, traces = evaluate_bm25(index, questions, qrels, cutoffs=(1, 2))
    assert metrics["recall_at_k"]["1"] == 1.0
    assert metrics["ndcg_at_k"]["1"] == 1.0
    assert metrics["primary_proxy_scope"] == "HIGH_confidence"
    assert metrics["promotion_allowed"] is False
    assert traces[0]["ranked"][0]["chunk_id"] == "c1"

    with pytest.raises(FileExistsError, match="already exist"):
        build_bm25_index(release, run)


def test_evaluation_writes_case_slice_and_error_artifacts(tmp_path: Path) -> None:
    """Validation must be observable beyond aggregate metrics."""
    release = tmp_path / "release"
    run = tmp_path / "run"
    validation = release / "qa" / "validation.jsonl"
    validation.parent.mkdir(parents=True)
    record = {
        "question_id": "q1",
        "question_model": "Điều kiện là gì?",
        "answer_model": "Theo Điều 13.\n- Điều kiện một.",
    }
    validation.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    run.mkdir()
    predictions = {"q1": {"answer": record["answer_model"]}}
    (run / "validation_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics = evaluate_run(release, run)
    assert metrics["meteor"] > 0.99
    assert metrics["metric_status"] == "PROVISIONAL_UNTIL_ORGANIZER_SCORER_PARITY"
    assert (run / "case_metrics.jsonl").is_file()
    assert (run / "slice_metrics.json").is_file()
    assert (run / "error_table.jsonl").is_file()


def test_paired_evaluation_reports_slice_deltas_and_counterexamples(tmp_path: Path) -> None:
    """A challenger comparison must expose slice-level gains and regressions."""
    release = tmp_path / "release"
    run = tmp_path / "run"
    validation = release / "qa" / "validation.jsonl"
    validation.parent.mkdir(parents=True)
    validation.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_model": "Điều kiện là gì?",
                "answer_model": "Theo Điều 13, cần giấy phép.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    run.mkdir()
    (run / "validation_predictions.json").write_text(
        json.dumps({"q1": {"answer": "Theo Điều 13, cần giấy phép."}}, ensure_ascii=False),
        encoding="utf-8",
    )
    parent = run / "parent.json"
    parent.write_text(
        json.dumps({"q1": {"answer": "Không rõ."}}, ensure_ascii=False),
        encoding="utf-8",
    )
    evaluate_run(release, run, parent_predictions=parent)
    paired = json.loads((run / "paired_comparison.json").read_text(encoding="utf-8"))
    assert paired["candidate_wins"] == 1
    assert paired["slice_deltas"]["overall"]["meteor_delta"] > 0
    assert paired["counterexamples"]


def test_submission_replay_rejects_extra_fields(tmp_path: Path) -> None:
    """The final archive contract allows only one answer field per expected ID."""
    questions = tmp_path / "public.jsonl"
    _write_questions(questions)
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps({"q1": {"answer": "A"}, "q2": {"answer": "B"}}),
        encoding="utf-8",
    )
    archive = tmp_path / "submission.zip"
    report = build_submission(
        questions,
        predictions,
        archive,
        expected_split="public",
    )
    assert report["status"] == "PASS"
    assert validate_submission_zip(archive, expected_ids={"q1", "q2"})["status"] == "PASS"
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("submission.json", json.dumps({"q1": {"answer": "A", "id": "x"}}))
    assert validate_submission_zip(archive, expected_ids={"q1", "q2"})["status"] == "FAIL"

    with zipfile.ZipFile(archive, "w") as value:
        value.writestr(
            "submission.json",
            '{"q1":{"answer":"A"},"q1":{"answer":"B"},"q2":{"answer":"C"}}',
        )
    duplicate_report = validate_submission_zip(archive, expected_ids={"q1", "q2"})
    assert duplicate_report["status"] == "FAIL"
    assert "Duplicate JSON key" in duplicate_report["errors"][0]


def test_unlabeled_output_audit_is_id_only_and_does_not_claim_metrics(tmp_path: Path) -> None:
    predictions = tmp_path / "public_predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "q1": {"answer": "Theo Điều 1. Nội dung hợp lệ."},
                "q2": {"answer": "lặp lại lặp lại lặp lại lặp lại lặp lại lặp lại"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    case_path = tmp_path / "cases.jsonl"

    report = audit_unlabeled_output(
        predictions,
        report_path,
        case_path,
        expected_ids={"q1", "q2"},
    )

    assert report["status"] == "PASS"
    assert report["gold_status"] == "absent"
    assert report["quality_claim_allowed"] is False
    case_text = case_path.read_text(encoding="utf-8")
    assert "Nội dung hợp lệ" not in case_text
    assert "q1" in case_text and "q2" in case_text


def test_runtime_forces_offline_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime guard must override contradictory network-enabled values."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    configure_offline_transformers()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_visible_answer_is_rejected(tmp_path: Path) -> None:
    """Batch runtime must fail closed if Public input exposes an answer."""
    questions = tmp_path / "public.jsonl"
    payload: dict[str, Any] = {
        "question_id": "q1",
        "source_task": "Task2",
        "question_model": "Câu hỏi",
        "split": "public",
        "answer": "Không được phép",
    }
    questions.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exposes an answer"):
        BatchInferenceRunner(FixtureEngine()).run(
            questions,
            tmp_path / "predictions.json",
            expected_split="public",
        )


def test_inventory_skips_ready_training_and_selects_evaluation(tmp_path: Path) -> None:
    """A READY checkpoint without validation must advance to evaluation, not Public."""
    release = tmp_path / "release"
    control = tmp_path / "control"
    run = tmp_path / "run"
    for path, status in (
        (release / "manifest.json", "READY"),
        (control / "tokenizer" / "tokenizer_report_qwen.json", "PASS"),
        (run / "run_manifest.json", "READY"),
        (run / "checkpoint_manifest.json", "READY"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": status}), encoding="utf-8")
    request = PipelineRequest(
        operation="inventory",
        stage="auto",
        profile="e0-direct",
        scope="full",
        release_root=release,
        control_root=control,
        run_root=run,
        report=tmp_path / "inventory.json",
    )

    inventory = inventory_pipeline(request)

    assert inventory["status"] == "READY_TO_ADVANCE"
    assert inventory["next_operation"] == "evaluate"
    assert inventory["next_stage"] == "evaluation"

    (run / ".pipeline.lock").write_text(
        json.dumps({"operation": "evaluate", "pid": 123, "started_at": "fixture"}),
        encoding="utf-8",
    )
    assert inventory_pipeline(request)["status"] == "BLOCKED"
