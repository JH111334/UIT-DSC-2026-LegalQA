from __future__ import annotations

import ast
import json
from pathlib import Path

from legalqa.core.config import GenerationConfig, PipelineConfig
from legalqa.core.io import load_qa
from legalqa.core.manifest import config_hash, write_run_manifest
from legalqa.generation import generators
from legalqa.preprocessing.sanitation import (
    AD_TEXT,
    sanitize_answer_raw,
    sanitize_corpus_document,
)


def test_generation_knob_reaches_all_three_model_generators() -> None:
    source = Path(generators.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    for name in ("TransformersGenerator", "Seq2SeqGenerator", "QwenVLTextGenerator"):
        rendered = ast.unparse(classes[name])
        assert "self.generation.no_repeat_ngram_size" in rendered
    assert source.count('"no_repeat_ngram_size"') == 3
    assert GenerationConfig().no_repeat_ngram_size == 4
    assert GenerationConfig().repetition_penalty == 1.20


def test_manifest_captures_config_code_input_and_output_hashes(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    manifest_path = tmp_path / "output.json.manifest.json"
    source.write_text("{}", encoding="utf-8")
    output.write_text("{}", encoding="utf-8")
    config = PipelineConfig()
    manifest = write_run_manifest(
        config=config,
        input_path=source,
        output_path=output,
        environment="local_cpu",
        adapter_path=None,
        manifest_path=manifest_path,
        execution_mode="schema_only",
    )
    assert manifest["config_hash"] == config_hash(config)
    assert manifest["code_hash"]
    assert manifest["input_sha256"]
    assert manifest["output_sha256"]
    assert manifest["quality_status"] == "NOT_APPLICABLE_SCHEMA_ONLY"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_release_jsonl_is_accepted_by_shared_qa_loader(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_model": "Câu hỏi",
                "answer_model": "Câu trả lời",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_qa(path)
    assert [(value.id, value.question, value.answer) for value in examples] == [
        ("q1", "Câu hỏi", "Câu trả lời")
    ]


def test_answer_sanitation_removes_only_reviewed_trailing_patterns() -> None:
    answer = "Nội dung pháp lý hợp lệ. (Hình từ Internet) phần quảng bá"
    assert sanitize_answer_raw(answer) == "Nội dung pháp lý hợp lệ."
    related = "Kết luận chính. ……… Câu hỏi liên đới là gì?"
    assert sanitize_answer_raw(related) == "Kết luận chính."
    form = "Mẫu đơn: Họ tên .......... Địa chỉ .........."
    assert sanitize_answer_raw(form) == form


def test_corpus_sanitation_preserves_nonterminal_form_noi_nhan() -> None:
    form = "Nơi nhận: trường thông tin trong mẫu đơn.\nHọ tên ..........\nNội dung hợp lệ."
    assert sanitize_corpus_document(form) == form
    document = (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Độc lập - Tự do - Hạnh phúc\n"
        "Điều 1. Nội dung hợp lệ.\n"
        + AD_TEXT
    )
    cleaned = sanitize_corpus_document(document)
    assert "CỘNG HÒA" not in cleaned
    assert "TVPL Pro" not in cleaned
    assert "Điều 1." in cleaned
