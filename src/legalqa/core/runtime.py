"""Runtime configuration for optional ML dependencies."""

from __future__ import annotations

import os


def configure_transformers_runtime(*, local_files_only: bool) -> None:
    """Configure optional Transformers stacks before importing them."""

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    if local_files_only:
        # Some nested auto-processor lookups do not consistently inherit
        # local_files_only from Sentence Transformers. Hub offline mode closes
        # that gap and avoids slow network retries on an air-gapped runner.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
