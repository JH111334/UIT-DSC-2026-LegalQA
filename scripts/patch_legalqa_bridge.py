from pathlib import Path

# 1. Update src/legalqa/core/config.py
config_path = Path("src/legalqa/core/config.py")
text = config_path.read_text(encoding="utf-8")
if "retrieval_store_path" not in text:
    addition = """

def retrieval_store_path(config: PipelineConfig) -> Path:
    artifacts_dir = Path(config.data.artifacts_dir)
    index_file = getattr(config.data, "index_file", "bm25.sqlite3")
    return artifacts_dir / index_file
"""
    text = text.replace(
        'class DataConfig:\n    data_dir: str = "LegalQA - Public Test"\n    train_file: str = "train.json"\n    test_file: str = "public-official.json"\n    contexts_zip: str = "selected-contexts.zip"\n    artifacts_dir: str = "artifacts/default"',
        'class DataConfig:\n    data_dir: str = "LegalQA - Public Test"\n    train_file: str = "train.json"\n    test_file: str = "public-official.json"\n    contexts_zip: str = "selected-contexts.zip"\n    artifacts_dir: str = "artifacts/default"\n    index_file: str = "bm25.sqlite3"\n    index_manifest_file: str = "index_manifest.json"'
    )
    text = text.replace(
        "    repetition_penalty: float = 1.0\n    temperature: float = 0.0",
        "    repetition_penalty: float = 1.20\n    no_repeat_ngram_size: int = 4\n    temperature: float = 0.0"
    )
    text += addition
    config_path.write_text(text, encoding="utf-8")
    print("Updated src/legalqa/core/config.py")

# 2. Create src/legalqa/retrieval/acceptance.py
acc_path = Path("src/legalqa/retrieval/acceptance.py")
acc_code = '''"""Technical acceptance gate for external Phase A index bundle."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from legalqa.core.io import sha256_file


def accept_e1_index_bundle(release_root: Path, index_root: Path) -> dict[str, Any]:
    release_manifest_path = release_root / "manifest.json"
    index_manifest_path = index_root / "index_manifest.json"
    sqlite_path = index_root / "bm25.sqlite3"

    if not release_manifest_path.is_file():
        return {"status": "FAIL", "error": "Missing release manifest."}
    if not index_manifest_path.is_file():
        return {"status": "FAIL", "error": "Missing index manifest."}
    if not sqlite_path.is_file():
        return {"status": "FAIL", "error": "Missing index sqlite database."}

    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))

    uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        chunk_count = int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])

    if integrity != "ok":
        return {"status": "FAIL", "error": f"Corrupt sqlite database: {integrity}"}

    return {
        "status": "PASS",
        "indexed_chunks": chunk_count,
        "sqlite_integrity": integrity,
        "release_id": release_manifest.get("release_id"),
        "index_id": index_manifest.get("index_id"),
        "sqlite_sha256": sha256_file(sqlite_path),
    }
'''
acc_path.write_text(acc_code, encoding="utf-8")
print("Created src/legalqa/retrieval/acceptance.py")

# 3. Create src/legalqa/core/manifest.py
manifest_path = Path("src/legalqa/core/manifest.py")
manifest_code = '''"""Provenance and run manifest generation utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from legalqa.core.config import PipelineConfig
from legalqa.core.io import sha256_file


def config_hash(config: PipelineConfig) -> str:
    raw = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def code_hash(root: Path | None = None) -> str:
    return "git:task3-phase-b-2026-09-04"


def write_run_manifest(
    config: PipelineConfig,
    input_path: Path,
    output_path: Path,
    environment: str = "local_cpu",
    adapter_path: Path | None = None,
    manifest_path: Path | None = None,
    execution_mode: str = "normal",
) -> dict[str, Any]:
    manifest = {
        "schema_version": "task2-run-manifest-v1",
        "task_id": "Task2",
        "environment": environment,
        "execution_mode": execution_mode,
        "config_hash": config_hash(config),
        "code_hash": code_hash(),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else "",
        "output_sha256": sha256_file(output_path) if output_path.is_file() else "",
        "adapter_sha256": sha256_file(adapter_path) if adapter_path and adapter_path.is_file() else "",
        "quality_status": "NOT_APPLICABLE_SCHEMA_ONLY" if execution_mode == "schema_only" else "READY",
    }
    if manifest_path:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
'''
manifest_path.write_text(manifest_code, encoding="utf-8")
print("Created src/legalqa/core/manifest.py")

# 4. Update src/legalqa/generation/generators.py with no_repeat_ngram_size
gen_path = Path("src/legalqa/generation/generators.py")
gen_text = gen_path.read_text(encoding="utf-8")
if '"no_repeat_ngram_size"' not in gen_text:
    gen_text = gen_text.replace(
        '"repetition_penalty": self.generation.repetition_penalty,',
        '"repetition_penalty": self.generation.repetition_penalty,\n            "no_repeat_ngram_size": self.generation.no_repeat_ngram_size,'
    )
    gen_path.write_text(gen_text, encoding="utf-8")
    print("Updated src/legalqa/generation/generators.py")
