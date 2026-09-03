"""Cấu hình tối thiểu cho preprocessing Task 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    """Đường dẫn dữ liệu BTC và vùng làm việc local-only."""

    data_dir: str = "Data/Task2"
    train_file: str = "train.json"
    test_file: str = "public-official.json"
    contexts_path: str = "selected-contexts"
    artifacts_dir: str = "Data/Task2/work/preprocessing"


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
    blacklist_path: str = "configs/Task2/preprocessing/boilerplate_blacklist.yaml"
    protected_patterns_path: str = "configs/Task2/preprocessing/protected_patterns.yaml"
    review_policy_path: str = "configs/Task2/preprocessing/boilerplate_review_policy.yaml"
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
class DiagnosticConfig:
    enabled: bool = True
    allow_training_use: bool = False
    manual_sample_size: int = 150
    seed: int = 2026


@dataclass(slots=True)
class PipelineConfig:
    """Chỉ chứa knob data build; retrieval và training có config riêng."""

    seed: int = 2026
    data: DataConfig = field(default_factory=DataConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    sanitation: SanitationConfig = field(default_factory=SanitationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    diagnostic: DiagnosticConfig = field(default_factory=DiagnosticConfig)

    def validate(self) -> None:
        if self.chunking.max_chars <= self.chunking.overlap_chars:
            raise ValueError("chunking.max_chars must exceed overlap_chars")
        if self.chunking.min_chars <= 0:
            raise ValueError("chunking.min_chars must be positive")
        if self.diagnostic.allow_training_use:
            raise ValueError(
                "Citation-derived labels are diagnostic-only; "
                "diagnostic.allow_training_use must remain false."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], value: dict[str, Any] | None) -> Any:
    return cls(**dict(value or {}))


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Preprocessing config must be a YAML object")
    config = PipelineConfig(
        seed=int(raw.get("seed", 2026)),
        data=_construct(DataConfig, raw.get("data")),
        audit=_construct(AuditConfig, raw.get("audit")),
        sanitation=_construct(SanitationConfig, raw.get("sanitation")),
        chunking=_construct(ChunkingConfig, raw.get("chunking")),
        diagnostic=_construct(DiagnosticConfig, raw.get("diagnostic")),
    )
    config.validate()
    return config


def artifact_dir(config: PipelineConfig) -> Path:
    return Path(config.data.artifacts_dir)


def data_path(config: PipelineConfig, filename: str) -> Path:
    return Path(config.data.data_dir) / filename
