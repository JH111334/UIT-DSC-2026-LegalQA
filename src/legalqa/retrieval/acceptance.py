"""Technical acceptance gate for external Phase A index bundle."""

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
