"""Typed configuration for the LegalQA pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    data_dir: str = "LegalQA - Public Test"
    train_file: str = "train.json"
    test_file: str = "public-official.json"
    contexts_zip: str = "selected-contexts.zip"
    artifacts_dir: str = "artifacts/default"
    index_file: str = "bm25.sqlite3"
    index_manifest_file: str = "index_manifest.json"


@dataclass(slots=True)
class AuditConfig:
    very_long_chars: int = 200_000
    extreme_chars: int = 1_000_000
    min_structure_density_per_10k: float = 0.25
    near_duplicate_enabled: bool = True
    near_duplicate_threshold: float = 0.94
    near_duplicate_num_perm: int = 64


@dataclass(slots=True)
class SanitationConfig:
    enabled: bool = False
    version: str = "v1"
    blacklist_path: str = "configs/preprocessing/boilerplate_blacklist.yaml"
    protected_patterns_path: str = "configs/preprocessing/protected_patterns.yaml"
    review_policy_path: str = "configs/preprocessing/boilerplate_review_policy.yaml"
    terminal_noi_nhan_min_ratio: float = 0.85
    block_min_chars: int = 100
    overlap_min_tokens: int = 12
    overlap_max_tokens: int = 50_000
    remove_retrieval_boilerplate: bool = True
    recover_paywall_documents: bool = True
    df_line_min_chars: int = 20
    df_line_max_chars: int = 500
    df_min_ratio: float = 0.01
    df_max_candidates: int = 500
    local_repetition_flag_ratio: float = 0.20
    local_max_tf_flag: int = 5


@dataclass(slots=True)
class ChunkingConfig:
    legal_aware: bool = True
    max_chars: int = 1800
    overlap_chars: int = 220
    min_chars: int = 60
    preamble_chars: int = 1400


@dataclass(slots=True)
class ComplexityConfig:
    enabled: bool = True
    entity_weight: float = 0.8
    condition_weight: float = 1.2
    citation_weight: float = 1.0
    conjunction_weight: float = 0.6
    length_weight: float = 0.8
    low_threshold: float = 2.2
    high_threshold: float = 5.0
    complexity_top_k: tuple[int, int, int] = (3, 5, 8)
    confidence_enabled: bool = True
    confident_gap: float = 0.20
    ambiguous_gap: float = 0.05
    confident_min_k: int = 2
    confident_max_k: int = 3
    ambiguous_min_k: int = 5


@dataclass(slots=True)
class RetrievalConfig:
    bm25_enabled: bool = True
    dense_enabled: bool = True
    bm25_top_k: int = 100
    dense_top_k: int = 100
    rrf_top_k: int = 50
    rrf_constant: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    legal_metadata_boost: float = 0.12
    rerank_enabled: bool = True
    rerank_top_k: int = 50


@dataclass(slots=True)
class ExpansionConfig:
    parent_enabled: bool = True
    graph_enabled: bool = False
    graph_max_per_seed: int = 2
    graph_decay: float = 0.72


@dataclass(slots=True)
class EvidenceConfig:
    max_chars: int = 12000
    max_per_document: int = 3
    parent_max_chars: int = 5000
    min_score_ratio: float = 0.35


@dataclass(slots=True)
class ModelSpec:
    name_or_path: str = ""
    parameters_b: float = 0.0
    backend: str = "none"
    enabled: bool = False
    local_files_only: bool = True
    batch_size: int = 16
    max_length: int = 2048
    query_prefix: str = ""
    document_prefix: str = ""
    device: str = ""
    revision: str | None = None
    adapter_path: str | None = None


@dataclass(slots=True)
class ModelsConfig:
    dense: ModelSpec = field(default_factory=ModelSpec)
    reranker: ModelSpec = field(default_factory=ModelSpec)
    generator: ModelSpec = field(default_factory=ModelSpec)
    parameter_limit_b: float = 4.0

    @property
    def total_parameters_b(self) -> float:
        return sum(
            spec.parameters_b
            for spec in (self.dense, self.reranker, self.generator)
            if spec.enabled
        )

    def validate_budget(self) -> None:
        if self.total_parameters_b >= self.parameter_limit_b:
            raise ValueError(
                f"Model budget is {self.total_parameters_b:.3f}B; it must be "
                f"strictly below {self.parameter_limit_b:.3f}B"
            )
        for role, spec in (
            ("dense", self.dense),
            ("reranker", self.reranker),
            ("generator", self.generator),
        ):
            local_backend = spec.backend in {"lexical", "extractive", "tfidf", "none"}
            if spec.enabled and not spec.name_or_path and not local_backend:
                raise ValueError(f"models.{role}.name_or_path is required")

    @staticmethod
    def _requires_real_model(role: str, spec: ModelSpec, fallback_backends: set[str]) -> None:
        if not spec.enabled:
            raise ValueError(f"models.{role}.enabled must be true for this configuration")
        if spec.backend in fallback_backends:
            raise ValueError(
                f"models.{role}.backend={spec.backend!r} is a fallback backend; "
                "configure a real model backend for this configuration"
            )
        if not spec.name_or_path:
            raise ValueError(f"models.{role}.name_or_path is required")


@dataclass(slots=True)
class DiagnosticConfig:
    enabled: bool = True
    allow_training_use: bool = False
    manual_sample_size: int = 150
    seed: int = 2026


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.2
    repetition_penalty: float = 1.20
    no_repeat_ngram_size: int = 4


@dataclass(slots=True)
class PipelineConfig:
    seed: int = 2026
    data: DataConfig = field(default_factory=DataConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    sanitation: SanitationConfig = field(default_factory=SanitationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    diagnostic: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def validate(self) -> None:
        if self.chunking.max_chars <= self.chunking.overlap_chars:
            raise ValueError("chunking.max_chars must exceed overlap_chars")
        if any(k <= 0 for k in self.complexity.complexity_top_k):
            raise ValueError("complexity.complexity_top_k values must be positive")
        if tuple(sorted(self.complexity.complexity_top_k)) != self.complexity.complexity_top_k:
            raise ValueError("complexity.complexity_top_k must be non-decreasing")
        if self.diagnostic.allow_training_use:
            raise ValueError(
                "Compliance guard: citation-derived labels are diagnostic-only. "
                "Keep diagnostic.allow_training_use=false unless the rules and code are "
                "explicitly revised after written organizer confirmation."
            )
        if self.retrieval.dense_enabled:
            self.models._requires_real_model("dense", self.models.dense, {"none", "tfidf"})
        if self.retrieval.rerank_enabled:
            self.models._requires_real_model("reranker", self.models.reranker, {"none", "lexical"})
            if self.models.reranker.backend != "cross_encoder":
                raise ValueError("models.reranker.backend must be cross_encoder when rerank is enabled")
        self.models.validate_budget()

    def validate_submission(self) -> None:
        if not self.sanitation.enabled:
            raise ValueError("sanitation.enabled must be true before generating predictions")
        if not self.retrieval.dense_enabled:
            raise ValueError("retrieval.dense_enabled must be true before generating predictions")
        if not self.retrieval.rerank_enabled:
            raise ValueError("retrieval.rerank_enabled must be true before generating predictions")
        self.models._requires_real_model("generator", self.models.generator, {"none", "extractive"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], value: dict[str, Any] | None) -> Any:
    value = dict(value or {})
    if cls is ComplexityConfig and "complexity_top_k" in value:
        value["complexity_top_k"] = tuple(int(x) for x in value["complexity_top_k"])
    return cls(**value)


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models_raw = raw.get("models") or {}
    config = PipelineConfig(
        seed=int(raw.get("seed", 2026)),
        data=_construct(DataConfig, raw.get("data")),
        audit=_construct(AuditConfig, raw.get("audit")),
        sanitation=_construct(SanitationConfig, raw.get("sanitation")),
        chunking=_construct(ChunkingConfig, raw.get("chunking")),
        retrieval=_construct(RetrievalConfig, raw.get("retrieval")),
        complexity=_construct(ComplexityConfig, raw.get("complexity")),
        expansion=_construct(ExpansionConfig, raw.get("expansion")),
        evidence=_construct(EvidenceConfig, raw.get("evidence")),
        models=ModelsConfig(
            dense=_construct(ModelSpec, models_raw.get("dense")),
            reranker=_construct(ModelSpec, models_raw.get("reranker")),
            generator=_construct(ModelSpec, models_raw.get("generator")),
            parameter_limit_b=float(models_raw.get("parameter_limit_b", 4.0)),
        ),
        diagnostic=_construct(DiagnosticConfig, raw.get("diagnostic")),
        generation=_construct(GenerationConfig, raw.get("generation")),
    )
    config.validate()
    return config


def artifact_dir(config: PipelineConfig) -> Path:
    return Path(config.data.artifacts_dir)


def data_path(config: PipelineConfig, filename: str) -> Path:
    return Path(config.data.data_dir) / filename


def retrieval_store_path(config: PipelineConfig) -> Path:
    artifacts_dir = Path(config.data.artifacts_dir)
    index_file = getattr(config.data, "index_file", "bm25.sqlite3")
    return artifacts_dir / index_file
