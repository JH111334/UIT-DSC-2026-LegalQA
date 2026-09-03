"""Run local-only tokenizer audits and answer-only SFT for Task 2."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi pháp luật Việt Nam. "
    "Trả lời trực tiếp bằng tiếng Việt và bảo toàn cấu trúc cần thiết."
)


class TrainingRuntimeError(RuntimeError):
    """Report a missing dependency, invalid control, or unsafe runtime."""


class ChatTokenizer(Protocol):
    """Describe the tokenizer methods needed by the governed runtime."""

    pad_token_id: int | None
    eos_token_id: int | None
    pad_token: str | None
    eos_token: str | None
    padding_side: str

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Mapping[str, Sequence[int]]: ...

    def save_pretrained(self, save_directory: str | Path) -> object: ...


@dataclass(frozen=True, slots=True)
class EncodedAnswer:
    """Carry one causal-LM record with prompt labels masked."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    target_tokens: int
    truncated_target_tokens: int


@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Normalize the bounded training knobs stored in the control artifact."""

    seed: int
    max_sequence_length: int
    max_new_tokens: int
    learning_rate: float
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    use_4bit: bool
    gradient_checkpointing: bool
    max_steps: int
    run_mode: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SFTConfig:
        """Load safe defaults while rejecting contradictory core settings."""
        if value.get("profile") != "e0-direct" or value.get("answer_only_loss") is not True:
            raise TrainingRuntimeError(
                "Training config must be e0-direct with answer_only_loss=true."
            )
        config = cls(
            seed=int(value.get("seed", 2026)),
            max_sequence_length=int(value.get("max_sequence_length", 2048)),
            max_new_tokens=int(value.get("max_new_tokens", 1024)),
            learning_rate=float(value.get("learning_rate", 2e-4)),
            epochs=float(value.get("epochs", 1.0)),
            batch_size=int(value.get("batch_size", 1)),
            gradient_accumulation_steps=int(value.get("gradient_accumulation_steps", 16)),
            lora_rank=int(value.get("lora_rank", 16)),
            lora_alpha=int(value.get("lora_alpha", 32)),
            lora_dropout=float(value.get("lora_dropout", 0.05)),
            use_4bit=bool(value.get("use_4bit", True)),
            gradient_checkpointing=bool(value.get("gradient_checkpointing", True)),
            max_steps=int(value.get("max_steps", -1)),
            run_mode=str(value.get("run_mode", "full")),
        )
        if (
            min(
                config.max_sequence_length,
                config.max_new_tokens,
                config.batch_size,
                config.gradient_accumulation_steps,
                config.lora_rank,
            )
            <= 0
        ):
            raise TrainingRuntimeError(
                "Training length, batch, accumulation, and LoRA rank must be positive."
            )
        if not 0.0 < config.learning_rate < 1.0 or not 0.0 <= config.lora_dropout < 1.0:
            raise TrainingRuntimeError("Invalid learning rate or LoRA dropout.")
        if config.run_mode not in {"full", "smoke"}:
            raise TrainingRuntimeError("run_mode must be full or smoke.")
        if config.run_mode == "smoke" and config.max_steps <= 0:
            raise TrainingRuntimeError("Smoke training requires a positive max_steps.")
        return config


def configure_offline_transformers() -> None:
    """Disable remote inference, Hub access, and telemetry before model imports."""
    os.environ.update(
        {
            "USE_TF": "0",
            "USE_TORCH": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )


def encode_answer_only(
    tokenizer: ChatTokenizer,
    question: str,
    answer: str,
    *,
    max_sequence_length: int,
    evidence: str | None = None,
) -> EncodedAnswer:
    """Serialize chat data and mask every non-assistant target token."""
    user_content = question if not evidence else f"Câu hỏi:\n{question}\n\nCăn cứ:\n{evidence}"
    prompt_messages = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    )
    full_messages = (*prompt_messages, {"role": "assistant", "content": answer})
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = _token_ids(tokenizer, prompt_text)
    full_ids = _token_ids(tokenizer, full_text)
    prompt_length = _common_prefix_length(prompt_ids, full_ids)
    if prompt_length == 0 or prompt_length >= len(full_ids):
        raise TrainingRuntimeError("Chat template does not expose a non-empty assistant target.")
    kept = full_ids[:max_sequence_length]
    kept_prompt = min(prompt_length, len(kept))
    target_tokens = len(full_ids) - prompt_length
    kept_target = max(0, len(kept) - kept_prompt)
    if kept_target == 0:
        raise TrainingRuntimeError("max_sequence_length truncates the entire assistant answer.")
    labels = [-100] * kept_prompt + kept[kept_prompt:]
    return EncodedAnswer(
        input_ids=tuple(kept),
        attention_mask=(1,) * len(kept),
        labels=tuple(labels),
        prompt_tokens=prompt_length,
        target_tokens=target_tokens,
        truncated_target_tokens=max(0, target_tokens - kept_target),
    )


def audit_tokenizer(release_root: Path, control_root: Path) -> dict[str, Any]:
    """Measure real Qwen token lengths and write the canonical tokenizer control."""
    configure_offline_transformers()
    model_manifest = _load_object(control_root / "model" / "model_snapshot_manifest.json")
    snapshot = _validated_snapshot(model_manifest)
    transformers = _require_module("transformers")
    tokenizer = cast(
        ChatTokenizer,
        transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True),
    )
    training = _load_object(control_root / "training" / "training_config.json")
    config = SFTConfig.from_mapping(training)
    rows = list(_read_jsonl(release_root / "qa" / "train.jsonl"))
    rows.extend(_read_jsonl(release_root / "qa" / "validation.jsonl"))
    question_lengths: list[int] = []
    target_lengths: list[int] = []
    total_lengths: list[int] = []
    truncation_counts = {
        str(length): 0 for length in _candidate_lengths(config.max_sequence_length)
    }
    generation_limits = (512, 768, 1024, 1280, 1536, 2048)
    target_truncation_counts = {str(length): 0 for length in generation_limits}
    longest: list[tuple[int, str]] = []
    for row in rows:
        encoded = encode_answer_only(
            tokenizer,
            str(row["question_model"]),
            str(row["answer_model"]),
            max_sequence_length=max(config.max_sequence_length, 65536),
        )
        total = len(encoded.input_ids)
        question_lengths.append(encoded.prompt_tokens)
        target_lengths.append(encoded.target_tokens)
        total_lengths.append(total)
        longest.append((total, str(row["question_id"])))
        for length in truncation_counts:
            truncation_counts[length] += int(total > int(length))
        for length in target_truncation_counts:
            target_truncation_counts[length] += int(encoded.target_tokens > int(length))
    report = {
        "schema_version": "task2-tokenizer-report-v1",
        "task_id": "Task2",
        "release_id": _load_object(release_root / "manifest.json")["release_id"],
        "status": "PASS",
        "model_id": model_manifest["model_id"],
        "model_revision": model_manifest["model_revision"],
        "tokenizer_revision": model_manifest["tokenizer_revision"],
        "tokenizer_files_sha256": _tree_sha256(snapshot),
        "prompt_version": str(training.get("prompt_version", "direct-v1")),
        "question_tokens": _distribution(question_lengths),
        "target_tokens": _distribution(target_lengths),
        "retrieval_chunk_tokens": {"status": "DEFERRED_TO_E1"},
        "packed_context_tokens": {"status": "DEFERRED_TO_E1"},
        "total_sequence_tokens": _distribution(total_lengths),
        "candidate_max_lengths": [int(value) for value in truncation_counts],
        "truncation_counts": truncation_counts,
        "candidate_max_new_tokens": list(generation_limits),
        "target_truncation_counts": target_truncation_counts,
        "sample_ids": [question_id for _length, question_id in sorted(longest, reverse=True)[:20]],
    }
    output = control_root / "tokenizer" / "tokenizer_report_qwen.json"
    _write_json(output, report)
    return {"status": "PASS", "report": str(output), "records": len(rows)}


def train_e0(release_root: Path, control_root: Path, run_root: Path) -> dict[str, Any]:
    """Fine-tune the approved local snapshot with LoRA/QLoRA and save a replay bundle."""
    if (run_root / "checkpoint_manifest.json").exists():
        raise TrainingRuntimeError("Run root already contains a checkpoint; use a new run_id.")
    configure_offline_transformers()
    started = time.perf_counter()
    transformers = _require_module("transformers")
    torch = _require_module("torch")
    peft = _require_module("peft")
    model_manifest = _load_object(control_root / "model" / "model_snapshot_manifest.json")
    snapshot = _validated_snapshot(model_manifest)
    config_path = control_root / "training" / "training_config.json"
    raw_config = _load_object(config_path)
    config = SFTConfig.from_mapping(raw_config)
    _seed_everything(config.seed, torch)

    tokenizer = cast(
        ChatTokenizer,
        transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True),
    )
    _ensure_padding(tokenizer)
    train_rows = list(_read_jsonl(release_root / "qa" / "train.jsonl"))
    validation_rows = list(_read_jsonl(release_root / "qa" / "validation.jsonl"))
    if not train_rows or not validation_rows:
        raise TrainingRuntimeError("Train and validation splits must both be non-empty.")
    if config.run_mode == "smoke":
        smoke_question_id = str(raw_config.get("smoke_question_id", ""))
        train_rows = [row for row in train_rows if str(row["question_id"]) == smoke_question_id]
        if len(train_rows) != 1:
            raise TrainingRuntimeError("Smoke config must select exactly one train question_id.")
    train_dataset = _AnswerOnlyDataset(tokenizer, train_rows, config.max_sequence_length)

    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if bool(torch.cuda.is_available()):
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["dtype"] = torch.float16
    if config.use_4bit:
        if not bool(torch.cuda.is_available()):
            raise TrainingRuntimeError(
                "4-bit QLoRA requires a CUDA runtime in this implementation."
            )
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(snapshot, **model_kwargs)
    if config.use_4bit:
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora = peft.LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    model = peft.get_peft_model(model, lora)
    model.config.use_cache = False
    output = run_root / "checkpoint-or-adapter"
    arguments = transformers.TrainingArguments(
        output_dir=str(run_root / "trainer-state"),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        optim="paged_adamw_8bit" if config.use_4bit else "adamw_torch",
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        fp16=bool(torch.cuda.is_available()),
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = transformers.Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        data_collator=_AnswerOnlyCollator(torch, int(tokenizer.pad_token_id or 0)),
    )
    result = trainer.train()
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    _write_run_artifacts(
        release_root,
        run_root,
        model_manifest,
        config_path,
        config,
        checkpoint_path=output,
        train_metrics=dict(result.metrics),
        elapsed_seconds=time.perf_counter() - started,
        torch=torch,
    )
    return {
        "status": "PASS",
        "run_mode": config.run_mode,
        "run_root": str(run_root),
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
    }


class _AnswerOnlyDataset:
    """Tokenize one record on demand to keep host RAM bounded."""

    def __init__(
        self,
        tokenizer: ChatTokenizer,
        rows: Sequence[Mapping[str, Any]],
        max_sequence_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.rows = tuple(rows)
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        value = encode_answer_only(
            self.tokenizer,
            str(row["question_model"]),
            str(row["answer_model"]),
            max_sequence_length=self.max_sequence_length,
        )
        return {
            "input_ids": list(value.input_ids),
            "attention_mask": list(value.attention_mask),
            "labels": list(value.labels),
        }


class _AnswerOnlyCollator:
    """Right-pad causal inputs while retaining -100 answer masking."""

    def __init__(self, torch: Any, pad_token_id: int) -> None:
        self.torch = torch
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        width = max(len(value["input_ids"]) for value in features)
        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        labels: list[list[int]] = []
        for value in features:
            padding = width - len(value["input_ids"])
            input_ids.append([*value["input_ids"], *([self.pad_token_id] * padding)])
            attention.append([*value["attention_mask"], *([0] * padding)])
            labels.append([*value["labels"], *([-100] * padding)])
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


def _validated_snapshot(manifest: Mapping[str, Any]) -> Path:
    if manifest.get("model_id") != "Qwen/Qwen2.5-1.5B-Instruct":
        raise TrainingRuntimeError("E0 runtime only accepts the approved Qwen anchor.")
    if int(manifest.get("parameter_count", 4_000_000_000)) >= 4_000_000_000:
        raise TrainingRuntimeError("Model parameter count violates the under-4B rule.")
    raw = manifest.get("snapshot_path")
    if not isinstance(raw, str) or not raw.strip():
        raise TrainingRuntimeError("Model manifest must provide a local snapshot_path.")
    snapshot = Path(raw).resolve()
    if not snapshot.is_dir():
        raise TrainingRuntimeError(f"Local model snapshot is missing: {snapshot}")
    return snapshot


def _token_ids(tokenizer: ChatTokenizer, text: str) -> list[int]:
    payload = tokenizer(text, add_special_tokens=False)
    values = payload.get("input_ids")
    if not isinstance(values, Sequence):
        raise TrainingRuntimeError("Tokenizer did not return input_ids.")
    return [int(value) for value in values]


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for first, second in zip(left, right, strict=False):
        if first != second:
            break
        length += 1
    return length


def _ensure_padding(tokenizer: ChatTokenizer) -> None:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise TrainingRuntimeError("Tokenizer has neither pad_token_id nor usable eos token.")


def _candidate_lengths(selected: int) -> tuple[int, ...]:
    return tuple(sorted({512, 1024, 1536, 2048, selected}))


def _distribution(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}

    def percentile(fraction: float) -> int:
        index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TrainingRuntimeError(f"Expected object at {path}:{line_number}.")
            yield value


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingRuntimeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise TrainingRuntimeError(
            f"Missing approved runtime dependency {name!r}; "
            "install the locked ML environment first."
        ) from exc


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else "WORKTREE_UNKNOWN"


def _write_run_artifacts(
    release_root: Path,
    run_root: Path,
    model_manifest: Mapping[str, Any],
    config_path: Path,
    config: SFTConfig,
    *,
    checkpoint_path: Path,
    train_metrics: Mapping[str, Any],
    elapsed_seconds: float,
    torch: Any,
) -> None:
    release_manifest = release_root / "manifest.json"
    run_root.mkdir(parents=True, exist_ok=True)
    run_status = "READY" if config.run_mode == "full" else "SMOKE_ONLY"
    _write_json(run_root / "config.json", _load_object(config_path))
    _write_json(
        run_root / "run_manifest.json",
        {
            "schema_version": "task2-run-manifest-v1",
            "task_id": "Task2",
            "run_id": run_root.name,
            "profile": "e0-direct",
            "status": run_status,
            "data_release_id": _load_object(release_manifest)["release_id"],
            "data_manifest_sha256": _sha256(release_manifest),
            "model_revision": model_manifest["model_revision"],
            "config_sha256": _sha256(config_path),
            "code_revision": _git_revision(),
            "seed": config.seed,
            "evaluation_role": "anchor" if config.run_mode == "full" else "memory_probe",
            "hypothesis_id": "H-E0" if config.run_mode == "full" else "H-HW1",
            "main_variable": "anchor" if config.run_mode == "full" else "hardware_fit",
            "metric_status": "PROVISIONAL_UNTIL_SCORER_PARITY",
        },
    )
    _write_json(
        run_root / "checkpoint_manifest.json",
        {
            "schema_version": "task2-checkpoint-manifest-v1",
            "task_id": "Task2",
            "status": run_status,
            "base_model_id": model_manifest["model_id"],
            "base_model_revision": model_manifest["model_revision"],
            "checkpoint_sha256": _tree_sha256(checkpoint_path),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "adapter_type": "QLORA" if config.use_4bit else "LORA",
        },
    )
    _write_json(run_root / "train_metrics.json", dict(train_metrics))
    cuda_available = bool(torch.cuda.is_available())
    peak_vram = int(torch.cuda.max_memory_allocated()) if cuda_available else 0
    total_vram = int(torch.cuda.get_device_properties(0).total_memory) if cuda_available else 0
    _write_json(
        run_root / "resource_log.json",
        {
            "schema_version": "task2-resource-log-v1",
            "elapsed_seconds": elapsed_seconds,
            "cuda_available": cuda_available,
            "total_vram_bytes": total_vram,
            "peak_vram_bytes": peak_vram,
            "within_physical_vram": not cuda_available or peak_vram <= total_vram,
            "resource_status": (
                "PASS"
                if not cuda_available or peak_vram <= total_vram
                else "PASS_WITH_WDDM_OVERSUBSCRIPTION"
            ),
        },
    )
