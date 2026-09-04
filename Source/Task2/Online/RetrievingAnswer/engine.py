"""Compose approved local BM25 evidence with a deterministic Qwen generator."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from Online.Training.runtime import SYSTEM_PROMPT, configure_offline_transformers

from .batch import AnswerResult
from .bm25 import BM25Hit, BM25Index


class EvidenceProvider(Protocol):
    """Provide bounded context and trace metadata for one question."""

    def evidence(self, question: str) -> tuple[str, Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class NoEvidence:
    """Represent the E0 direct-generation control."""

    def evidence(self, question: str) -> tuple[str, Mapping[str, Any]]:
        del question
        return "", {"retrieval_profile": "e0-direct", "evidence": []}


@dataclass(frozen=True, slots=True)
class BM25Evidence:
    """Pack de-duplicated parent contexts from the E1 lexical index."""

    index: BM25Index
    top_k: int = 5
    candidate_k: int = 20
    max_context_chars: int = 8000

    def evidence(self, question: str) -> tuple[str, Mapping[str, Any]]:
        hits = self.index.search(question, limit=self.candidate_k)
        selected = _select_parent_contexts(
            hits,
            top_k=self.top_k,
            max_context_chars=self.max_context_chars,
        )
        text = "\n\n".join(
            f"[Căn cứ {index}]\n{hit.parent_text}" for index, hit in enumerate(selected, start=1)
        )
        trace = {
            "retrieval_profile": "e1-bm25",
            "candidate_k": self.candidate_k,
            "selected_k": len(selected),
            "context_chars": len(text),
            "evidence": [
                {
                    "chunk_id": hit.chunk_id,
                    "parent_chunk_id": hit.parent_chunk_id,
                    "doc_id": hit.doc_id,
                    "rank": rank,
                    "bm25_score": hit.score,
                }
                for rank, hit in enumerate(selected, start=1)
            ],
        }
        return text, trace


class TransformersAnswerEngine:
    """Load an offline checkpoint and generate answer-only Vietnamese text."""

    def __init__(
        self,
        control_root: Path,
        run_root: Path,
        *,
        evidence_provider: EvidenceProvider | None = None,
    ) -> None:
        configure_offline_transformers()
        transformers = _require_module("transformers")
        torch = _require_module("torch")
        peft = _require_module("peft")
        model_manifest = _load_object(control_root / "model" / "model_snapshot_manifest.json")
        checkpoint_manifest = _load_object(run_root / "checkpoint_manifest.json")
        training = _load_object(run_root / "config.json")
        decoding_path = control_root / "inference" / "decoding_config.json"
        decoding = _load_object(decoding_path)
        _validate_decoding(decoding)
        snapshot = Path(str(model_manifest.get("snapshot_path", ""))).resolve()
        checkpoint = Path(str(checkpoint_manifest.get("checkpoint_path", ""))).resolve()
        if not snapshot.is_dir() or not checkpoint.is_dir():
            raise FileNotFoundError("Local base snapshot and checkpoint adapter are required.")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs: dict[str, Any] = {"local_files_only": True}
        if bool(torch.cuda.is_available()):
            model_kwargs.update({"device_map": "auto", "dtype": torch.float16})
        if bool(training.get("use_4bit", True)) and bool(torch.cuda.is_available()):
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        base_model = transformers.AutoModelForCausalLM.from_pretrained(snapshot, **model_kwargs)
        self.model = peft.PeftModel.from_pretrained(base_model, checkpoint, local_files_only=True)
        self.model.eval()
        self.model.config.use_cache = True
        self.torch = torch
        self.max_input_tokens = int(training.get("max_sequence_length", 2048))
        self.max_new_tokens = int(decoding["max_new_tokens"])
        self.repetition_penalty = float(decoding.get("repetition_penalty", 1.0))
        self.no_repeat_ngram_size = int(decoding.get("no_repeat_ngram_size", 0))
        self.inference_batch_size = int(decoding.get("inference_batch_size", 1))
        self.decode_id = str(decoding["decode_id"])
        self.decoding_config_sha256 = _sha256(decoding_path)
        self.evidence_provider = evidence_provider or NoEvidence()

    def trace_contract(self) -> dict[str, str]:
        """Return fields that must match before resuming persisted predictions."""
        return {
            "decode_id": self.decode_id,
            "decoding_config_sha256": self.decoding_config_sha256,
        }

    def answer(self, question: str) -> AnswerResult:
        """Generate one greedy answer through the same bounded batch path."""
        return self.answer_many((question,))[0]

    def answer_many(self, questions: Sequence[str]) -> tuple[AnswerResult, ...]:
        """Generate a deterministic micro-batch while preserving per-question traces."""
        if not questions:
            return ()
        prompts: list[str] = []
        retrieval_traces: list[Mapping[str, Any]] = []
        for question in questions:
            evidence, retrieval_trace = self.evidence_provider.evidence(question)
            user = question if not evidence else f"Câu hỏi:\n{question}\n\nCăn cứ:\n{evidence}"
            messages = (
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            )
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            retrieval_traces.append(retrieval_trace)
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        input_width = int(encoded["input_ids"].shape[-1])
        gen_kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **gen_kwargs)
        results: list[AnswerResult] = []
        for index, retrieval_trace in enumerate(retrieval_traces):
            raw_generated = output[index][input_width:].tolist()
            generated, stopped = _trim_generated_tokens(
                raw_generated,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            answer = str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()
            trace = {
                **dict(retrieval_trace),
                "generator": "Qwen/Qwen2.5-1.5B-Instruct+adapter",
                "decoding": "greedy",
                "decode_id": self.decode_id,
                "decoding_config_sha256": self.decoding_config_sha256,
                "inference_batch_size": self.inference_batch_size,
                "actual_batch_size": len(questions),
                "input_tokens": int(encoded["attention_mask"][index].sum().item()),
                "output_tokens": len(generated),
                "max_new_tokens": self.max_new_tokens,
                "hit_max_new_tokens": not stopped and len(raw_generated) >= self.max_new_tokens,
            }
            results.append(AnswerResult(answer=answer, trace=trace))
        return tuple(results)


def _select_parent_contexts(
    hits: Sequence[BM25Hit],
    *,
    top_k: int,
    max_context_chars: int,
) -> list[BM25Hit]:
    if top_k <= 0 or max_context_chars <= 0:
        raise ValueError("Evidence top_k and context budget must be positive.")
    selected: list[BM25Hit] = []
    seen: set[str] = set()
    used = 0
    for hit in hits:
        parent_id = hit.parent_chunk_id or hit.chunk_id
        if parent_id in seen:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        text = hit.parent_text.strip()
        if not text:
            continue
        if len(text) > remaining:
            fallback = hit.retrieval_text.strip()
            bounded = fallback if len(fallback) <= remaining else fallback[:remaining]
            if not bounded:
                continue
            hit = replace(hit, parent_text=bounded)
            text = bounded
        selected.append(hit)
        seen.add(parent_id)
        used += min(len(text), remaining)
        if len(selected) >= top_k:
            break
    return selected


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_decoding(value: Mapping[str, Any]) -> None:
    """Reject sampling, unbounded length, or unsupported decoding controls."""
    if (
        value.get("task_id") != "Task2"
        or value.get("status") != "APPROVED"
        or value.get("deterministic") is not True
        or value.get("do_sample") is not False
    ):
        raise ValueError("Decoding config must be approved deterministic Task 2 greedy decode.")
    max_new_tokens = int(value.get("max_new_tokens", 0))
    repetition_penalty = float(value.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(value.get("no_repeat_ngram_size", 0))
    inference_batch_size = int(value.get("inference_batch_size", 1))
    if not 1 <= max_new_tokens <= 2048:
        raise ValueError("max_new_tokens must be between 1 and 2048.")
    if not 0.5 <= repetition_penalty <= 2.0:
        raise ValueError("repetition_penalty must be between 0.5 and 2.0.")
    if not 0 <= no_repeat_ngram_size <= 10:
        raise ValueError("no_repeat_ngram_size must be between 0 and 10.")
    if not 1 <= inference_batch_size <= 32:
        raise ValueError("inference_batch_size must be between 1 and 32.")


def _trim_generated_tokens(
    token_ids: Sequence[int],
    *,
    eos_token_id: int | Sequence[int] | None,
    pad_token_id: int | None,
) -> tuple[list[int], bool]:
    """Remove batch padding and report whether generation emitted a stop token."""
    if eos_token_id is None:
        stop_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
        stop_ids = {eos_token_id}
    else:
        stop_ids = {int(token_id) for token_id in eos_token_id}
    if pad_token_id is not None:
        stop_ids.add(int(pad_token_id))
    kept: list[int] = []
    for token_id in token_ids:
        if int(token_id) in stop_ids:
            return kept, True
        kept.append(int(token_id))
    return kept, False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(f"Missing locked runtime dependency: {name}") from exc
