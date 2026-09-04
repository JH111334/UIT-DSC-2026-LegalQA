from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.config import PipelineConfig
from ..core.io import load_qa, write_json_atomic
from ..retrieval.engine import HybridRetriever
from ..retrieval.evidence import EvidenceAssembler
from .generators import create_generator


class LegalQAPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        config.validate()
        self.config = config
        self.retriever = HybridRetriever(config)
        self.evidence = EvidenceAssembler(config)
        self.generator = create_generator(config.models.generator, config.generation)

    def answer(self, question: str) -> dict[str, Any]:
        candidates = self.retriever.retrieve(question)
        evidence, decision = self.evidence.assemble(question, candidates)
        answer = self.generator.generate(question, evidence)
        return {
            "answer": answer,
            "decision": decision,
            "evidence": [
                {
                    "id": item.id,
                    "document_id": item.document_id,
                    "source_name": item.source_name,
                    "score": item.score,
                    "provenance": list(item.provenance),
                }
                for item in evidence
            ],
        }

    def predict_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        trace_path: str | Path | None = None,
        limit: int | None = None,
        resume: bool = False,
        checkpoint_every: int = 1,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        examples = load_qa(input_path)
        if limit is not None:
            examples = examples[:limit]
        output = Path(output_path)
        trace_output = Path(trace_path) if trace_path else None
        predictions = self._load_checkpoint(output) if resume else {}
        traces = self._load_checkpoint(trace_output) if resume and trace_output else {}
        target_ids = {example.id for example in examples}
        predictions = {key: value for key, value in predictions.items() if key in target_ids}
        traces = {key: value for key, value in traces.items() if key in target_ids}
        notify = progress or (lambda _message: None)
        generated = 0
        reused = 0
        pending_checkpoint = 0
        total = len(examples)
        for index, example in enumerate(examples, 1):
            prediction = predictions.get(example.id)
            prediction_complete = (
                isinstance(prediction, dict)
                and prediction.get("question") == example.question
                and isinstance(prediction.get("answer"), str)
            )
            trace_complete = trace_output is None or example.id in traces
            if prediction_complete and trace_complete:
                reused += 1
                notify(f"Reused predicted answer {index}/{total}")
                continue
            result = self.answer(example.question)
            predictions[example.id] = {"question": example.question, "answer": result["answer"]}
            traces[example.id] = {"decision": result["decision"], "evidence": result["evidence"]}
            generated += 1
            pending_checkpoint += 1
            if pending_checkpoint >= checkpoint_every:
                self._write_checkpoints(output, predictions, trace_output, traces)
                pending_checkpoint = 0
            notify(f"Generated predicted answer {index}/{total}")
        self._write_checkpoints(output, predictions, trace_output, traces)
        return {
            "input": str(input_path),
            "output": str(output),
            "total": total,
            "generated": generated,
            "reused": reused,
            "complete": len(predictions) == total,
        }

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Prediction checkpoint must be a JSON object: {path}")
        return value

    @staticmethod
    def _write_checkpoints(
        output: Path,
        predictions: dict[str, Any],
        trace_output: Path | None,
        traces: dict[str, Any],
    ) -> None:
        write_json_atomic(output, predictions)
        if trace_output:
            write_json_atomic(trace_output, traces)
