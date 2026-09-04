from __future__ import annotations

import re
from typing import Protocol

from ..core.config import GenerationConfig, ModelSpec
from ..core.runtime import configure_transformers_runtime
from ..core.schema import Evidence
from ..retrieval.engine import tokenize

SYSTEM_PROMPT = """Bạn là trợ lý pháp luật Việt Nam. Chỉ trả lời dựa trên các chứng cứ được cung cấp.
Nêu căn cứ pháp lý chính xác, trả lời trực tiếp bằng văn xuôi ngắn gọn và không bịa thông tin.
Không trình bày chuỗi suy luận nội bộ. Nếu chứng cứ không đủ, nói rõ giới hạn đó."""

QWEN_LEGALQA_PROMPT = """Bạn là trợ lý pháp luật Việt Nam. Chỉ dùng chứng cứ được cung cấp và tuyệt đối không bịa thông tin.
Viết theo phong cách đáp án pháp luật: mở đầu bằng căn cứ văn bản/điều/khoản nếu xác định được; trình bày đầy đủ quy định trực tiếp liên quan, gồm điều kiện, mức phạt, biện pháp bổ sung hoặc khắc phục nếu có; sau đó kết luận thẳng vào câu hỏi.
Ưu tiên giữ nguyên số hiệu, con số và nội dung pháp lý quan trọng. Khi chứng cứ đủ, trả lời khoảng 200-350 từ; không trình bày chuỗi suy luận nội bộ. Nếu chứng cứ không đủ, nói rõ giới hạn."""


def format_prompt(question: str, evidence: list[Evidence]) -> list[dict[str, str]]:
    blocks = []
    for index, item in enumerate(evidence, 1):
        blocks.append(f"[Chứng cứ {index} — {item.source_name}]\n{item.text}")
    context = "\n\n".join(blocks) if blocks else "Không có chứng cứ phù hợp."
    user = f"CÂU HỎI:\n{question}\n\nCHỨNG CỨ:\n{context}\n\nTRẢ LỜI:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class Generator(Protocol):
    def generate(self, question: str, evidence: list[Evidence]) -> str: ...


class ExtractiveGenerator:
    """Dependency-free smoke-test backend; not intended as the final submission model."""

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        if not evidence:
            return "Không đủ căn cứ trong kho văn bản được cung cấp để trả lời câu hỏi."
        query = set(tokenize(question))
        candidates: list[tuple[float, int, str]] = []
        order = 0
        for item in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", item.text):
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                tokens = set(tokenize(sentence))
                score = len(query & tokens) / max(1, len(query))
                candidates.append((score, order, sentence))
                order += 1
        best = sorted(candidates, key=lambda value: (-value[0], value[1]))[:6]
        return " ".join(sentence for _score, _order, sentence in sorted(best, key=lambda value: value[1]))


class TransformersGenerator:
    def __init__(self, spec: ModelSpec, generation: GenerationConfig) -> None:
        configure_transformers_runtime(local_files_only=spec.local_files_only)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the generation extra: pip install -e .[generation]") from error
        self.torch = torch
        self.generation = generation
        self.tokenizer = AutoTokenizer.from_pretrained(spec.name_or_path, local_files_only=spec.local_files_only)
        kwargs = {"device_map": "auto", "local_files_only": spec.local_files_only}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(spec.name_or_path, dtype="auto", **kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(spec.name_or_path, torch_dtype="auto", **kwargs)

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        messages = format_prompt(question, evidence)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, dict):
            model_inputs = {key: value.to(self.model.device) for key, value in encoded.items()}
            input_ids = model_inputs["input_ids"]
        else:
            input_ids = encoded.to(self.model.device)
            model_inputs = {"input_ids": input_ids}
        kwargs = {
            "max_new_tokens": self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
            "repetition_penalty": self.generation.repetition_penalty,
            "no_repeat_ngram_size": self.generation.no_repeat_ngram_size,
        }
        if self.generation.do_sample:
            kwargs["temperature"] = self.generation.temperature
        with self.torch.inference_mode():
            output = self.model.generate(**model_inputs, **kwargs)
        generated = output[:, input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


class Seq2SeqGenerator:
    def __init__(self, spec: ModelSpec, generation: GenerationConfig) -> None:
        configure_transformers_runtime(local_files_only=spec.local_files_only)
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the generation extra: pip install -e .[generation]") from error
        self.torch = torch
        self.generation = generation
        self.max_length = spec.max_length
        self.device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.name_or_path, local_files_only=spec.local_files_only
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            spec.name_or_path, local_files_only=spec.local_files_only
        ).to(self.device)
        self.model.eval()

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        blocks = [f"Chứng cứ {index} ({item.source_name}):\n{item.text}" for index, item in enumerate(evidence, 1)]
        context = "\n\n".join(blocks) if blocks else "Không có chứng cứ phù hợp."
        prompt = (
            "Trả lời câu hỏi pháp luật Việt Nam chỉ dựa trên chứng cứ. "
            "Nêu căn cứ chính xác, trả lời trực tiếp và không bịa thông tin.\n\n"
            f"Câu hỏi: {question}\n\n{context}\n\nTrả lời:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        kwargs = {
            "max_new_tokens": self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
            "repetition_penalty": self.generation.repetition_penalty,
            "no_repeat_ngram_size": self.generation.no_repeat_ngram_size,
        }
        if self.generation.do_sample:
            kwargs["temperature"] = self.generation.temperature
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


class QwenVLTextGenerator:
    """Use a locally cached Qwen2.5-VL checkpoint for text-only generation."""

    def __init__(self, spec: ModelSpec, generation: GenerationConfig) -> None:
        configure_transformers_runtime(local_files_only=spec.local_files_only)
        try:
            import torch
            from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Qwen2.5-VL requires a recent Transformers installation") from error
        self.torch = torch
        self.generation = generation
        self.max_length = spec.max_length
        self.device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.name_or_path, local_files_only=spec.local_files_only
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            spec.name_or_path,
            local_files_only=spec.local_files_only,
            dtype="auto",
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        evidence_budget = max(512, int(self.max_length * 1.5))

        def packed_messages(character_budget: int) -> list[dict[str, str]]:
            blocks = []
            remaining = character_budget
            for index, item in enumerate(evidence, 1):
                if remaining <= 0:
                    break
                text = item.text[:remaining]
                if not text.strip():
                    continue
                blocks.append(f"[Chứng cứ {index} — {item.source_name}]\n{text}")
                remaining -= len(text)
            context = "\n\n".join(blocks) if blocks else "Không có chứng cứ phù hợp."
            # Put the question after evidence so it and the assistant prefix are
            # always retained while fitting long legal context.
            user = f"CHỨNG CỨ:\n{context}\n\nCÂU HỎI:\n{question}\n\nTRẢ LỜI:"
            return [
                {"role": "system", "content": QWEN_LEGALQA_PROMPT},
                {"role": "user", "content": user},
            ]

        while True:
            messages = packed_messages(evidence_budget)
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            if inputs["input_ids"].shape[1] <= self.max_length or evidence_budget <= 512:
                break
            evidence_budget = max(512, int(evidence_budget * 0.8))
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError("Qwen prompt scaffolding exceeds configured max_length")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_ids = inputs["input_ids"]
        kwargs = {
            "max_new_tokens": self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
            "repetition_penalty": self.generation.repetition_penalty,
            "no_repeat_ngram_size": self.generation.no_repeat_ngram_size,
        }
        if self.generation.do_sample:
            kwargs["temperature"] = self.generation.temperature
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        generated = output[:, input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def create_generator(spec: ModelSpec, generation: GenerationConfig) -> Generator:
    if not spec.enabled or spec.backend == "extractive":
        return ExtractiveGenerator()
    if spec.backend == "transformers":
        return TransformersGenerator(spec, generation)
    if spec.backend == "seq2seq_transformers":
        return Seq2SeqGenerator(spec, generation)
    if spec.backend == "qwen_vl_text":
        return QwenVLTextGenerator(spec, generation)
    raise ValueError(f"Unsupported generator backend: {spec.backend}")
