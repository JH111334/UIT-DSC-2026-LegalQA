from __future__ import annotations

import json
from pathlib import Path

from Online.RetrievingAnswer.bm25 import build_bm25_index

from legalqa.core.config import load_config, retrieval_store_path
from legalqa.retrieval.acceptance import accept_e1_index_bundle
from legalqa.retrieval.store import CorpusStore


def _write_release(root: Path) -> None:
    chunks = root / "corpus" / "chunks.jsonl"
    chunks.parent.mkdir(parents=True)
    chunks.write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "source_task": "Task2",
                "parent_chunk_id": "p1",
                "retrieval_text": "Văn bản > Điều 13\nđiều kiện bán lẻ rượu",
                "parent_text": "Điều 13. Điều kiện bán lẻ rượu.",
                "doc_number_canonical": "105/2017/NĐ-CP",
                "article": "13",
                "indexable": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "corpus" / "corpus_manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "task2-data-release-v1",
                "task_id": "Task2",
                "release_id": "task2-data-v3",
                "status": "READY",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_phase_a_candidate_index_is_accepted_and_queryable(tmp_path: Path) -> None:
    release = tmp_path / "release"
    index_run = tmp_path / "index-run"
    _write_release(release)
    build_bm25_index(
        release,
        index_run,
        index_id="task2-data-v3-bm25-v1",
        manifest_status="CANDIDATE",
    )

    report = accept_e1_index_bundle(release, index_run / "index")

    assert report["status"] == "PASS"
    assert report["indexed_chunks"] == 1
    store = CorpusStore(index_run / "index" / "bm25.sqlite3")
    try:
        chunk_id, score = store.search_bm25("bán lẻ rượu", 1)[0]
        assert chunk_id == "c1"
        assert score > 0
        assert store.get_chunk(chunk_id).parent_id == "p1"
        assert store.get_parent("p1").article == "13"
    finally:
        store.close()


def test_e1_config_selects_phase_a_index_filename() -> None:
    config = load_config("configs/e1_rag.yaml")
    assert retrieval_store_path(config).as_posix().endswith(
        "task2-data-v3-bm25-candidate/index/bm25.sqlite3"
    )

