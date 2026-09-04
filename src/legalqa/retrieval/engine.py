from __future__ import annotations

import gzip
import heapq
import math
import pickle
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from ..core.config import ModelSpec, PipelineConfig, artifact_dir
from ..core.io import read_jsonl, write_json
from ..core.metadata import extract_citations
from ..core.normalize import normalize_text
from ..core.runtime import configure_transformers_runtime
from ..core.schema import Candidate, LegalChunk

_TOKEN_RE = re.compile(r"\d{1,5}/\d{4}/[A-ZĐ0-9-]+|[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(normalize_text(text, preserve_newlines=False))]


class BM25Index:
    def __init__(
        self,
        chunk_ids: list[str],
        postings: dict[str, list[tuple[int, int]]],
        document_lengths: list[int],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunk_ids = chunk_ids
        self.postings = postings
        self.document_lengths = document_lengths
        self.average_length = sum(document_lengths) / max(1, len(document_lengths))
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, chunks: list[LegalChunk]) -> BM25Index:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for index, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.retrieval_text))
            lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                postings[token].append((index, frequency))
        return cls([chunk.id for chunk in chunks], dict(postings), lengths)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        corpus_size = len(self.chunk_ids)
        for token in dict.fromkeys(tokenize(query)):
            posting = self.postings.get(token)
            if not posting:
                continue
            document_frequency = len(posting)
            idf = math.log(1.0 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5))
            for index, frequency in posting:
                length_norm = 1.0 - self.b + self.b * self.document_lengths[index] / max(1e-9, self.average_length)
                scores[index] += idf * frequency * (self.k1 + 1.0) / (frequency + self.k1 * length_norm)
        best = heapq.nlargest(top_k, scores.items(), key=lambda item: (item[1], -item[0]))
        return [(self.chunk_ids[index], float(score)) for index, score in best]

    def save(self, path: str | Path) -> None:
        # This cache contains no executable code and must only be loaded from the
        # self-built artifacts directory; pickle itself is not a trust boundary.
        with gzip.open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        with gzip.open(path, "rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, cls):
            raise TypeError("Unexpected BM25 cache type")
        return value


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        values = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        self.embeddings = values / np.maximum(norms, 1e-12)
        self.chunk_ids = chunk_ids

    def search_vector(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        scores = self.embeddings @ query[0]
        count = min(top_k, len(scores))
        indices = np.argpartition(scores, -count)[-count:] if count < len(scores) else np.arange(len(scores))
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [(self.chunk_ids[int(index)], float(scores[index])) for index in indices]


class FaissDenseIndex:
    def __init__(self, path: str | Path, chunk_ids: list[str]) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("FAISS index exists but faiss-cpu is not installed") from error
        self.index = faiss.read_index(str(path))
        self.chunk_ids = chunk_ids
        if self.index.ntotal != len(chunk_ids):
            raise ValueError("FAISS vector count does not match dense_chunk_ids.json")

    def search_vector(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        count = min(top_k, len(self.chunk_ids))
        scores, indices = self.index.search(query, count)
        return [
            (self.chunk_ids[int(index)], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]


class DenseEncoder:
    def __init__(self, spec: ModelSpec) -> None:
        # The dense stack is PyTorch-only. Prevent an unrelated optional
        # TensorFlow installation from being imported by Transformers.
        configure_transformers_runtime(local_files_only=spec.local_files_only)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install the retrieval extra: pip install -e .[retrieval]") from error
        load_kwargs = {"local_files_only": spec.local_files_only}
        if spec.device:
            load_kwargs["device"] = spec.device
        self.model = SentenceTransformer(spec.name_or_path, **load_kwargs)
        if spec.max_length > 0:
            self.model.max_seq_length = spec.max_length
        self.batch_size = spec.batch_size
        self.query_prefix = spec.query_prefix
        self.document_prefix = spec.document_prefix

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        values = [f"{self.document_prefix}{text}" for text in texts]
        method = self.model.encode if self.document_prefix else getattr(
            self.model, "encode_document", self.model.encode
        )
        return np.asarray(
            method(
                values,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        values = [f"{self.query_prefix}{query}" for query in queries]
        method = self.model.encode if self.query_prefix else getattr(
            self.model, "encode_query", self.model.encode
        )
        return np.asarray(
            method(
                values,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


def benchmark_dense_throughput(
    config: PipelineConfig,
    max_chunks: int = 64,
) -> dict[str, object]:
    """Measure local dense encoding throughput without mutating the index."""

    if not config.models.dense.enabled:
        raise ValueError("models.dense.enabled must be true")
    output = artifact_dir(config)
    texts = []
    for value in read_jsonl(output / "chunks.jsonl"):
        texts.append(str(value["retrieval_text"]))
        if len(texts) >= max_chunks:
            break
    if not texts:
        raise ValueError("No chunks available for dense benchmark")
    # Production encoding orders chunks by length so each batch wastes less
    # CPU time on padding. Sorting this representative sample mirrors that path.
    texts.sort(key=len)
    import torch

    load_started = time.perf_counter()
    encoder = DenseEncoder(config.models.dense)
    load_seconds = time.perf_counter() - load_started
    warmup_count = min(2, len(texts))
    encoder.encode_documents(texts[:warmup_count])
    started = time.perf_counter()
    embeddings = encoder.encode_documents(texts)
    encode_seconds = time.perf_counter() - started
    chunks_per_second = len(texts) / max(encode_seconds, 1e-9)
    build = __import__("json").loads((output / "build_report.json").read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "model": config.models.dense.name_or_path,
        "backend": config.models.dense.backend,
        "device": str(encoder.model.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "max_length": config.models.dense.max_length,
        "batch_size": config.models.dense.batch_size,
        "query_prefix": config.models.dense.query_prefix,
        "document_prefix": config.models.dense.document_prefix,
        "batch_order": "length_ascending",
        "chunks": len(texts),
        "embedding_shape": list(embeddings.shape),
        "load_seconds": round(load_seconds, 4),
        "encode_seconds": round(encode_seconds, 4),
        "chunks_per_second": round(chunks_per_second, 4),
        "estimated_full_hours": round(
            int(build["chunks"]) / max(chunks_per_second, 1e-9) / 3600.0, 3
        ),
        "mean_l2_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
        "mutated_index": False,
    }
    write_json(output / "dense_throughput_report.json", report)
    return report


class Reranker(Protocol):
    def score(self, query: str, chunks: list[LegalChunk]) -> list[float]: ...


class LexicalReranker:
    def score(self, query: str, chunks: list[LegalChunk]) -> list[float]:
        query_tokens = set(tokenize(query))
        result = []
        for chunk in chunks:
            tokens = set(tokenize(chunk.retrieval_text))
            overlap = len(query_tokens & tokens)
            result.append(overlap / max(1.0, math.sqrt(len(query_tokens) * len(tokens))))
        return result


class CrossEncoderReranker:
    def __init__(self, spec: ModelSpec) -> None:
        configure_transformers_runtime(local_files_only=spec.local_files_only)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError("Install the retrieval extra: pip install -e .[retrieval]") from error
        self.model = CrossEncoder(
            spec.name_or_path,
            max_length=spec.max_length,
            local_files_only=spec.local_files_only,
            device=spec.device or None,
        )
        self.batch_size = spec.batch_size

    def score(self, query: str, chunks: list[LegalChunk]) -> list[float]:
        return self.score_pairs([(query, chunk.retrieval_text) for chunk in chunks])

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        raw = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        # Qwen3 CrossEncoder returns raw logit differences. Sigmoid makes its
        # scores comparable and suitable for confidence/elbow analysis.
        values = np.where(values >= 0, 1 / (1 + np.exp(-values)), np.exp(values) / (1 + np.exp(values)))
        return [float(value) for value in values]


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    *,
    weights: dict[str, float] | None = None,
    constant: int = 60,
    top_k: int = 50,
) -> list[Candidate]:
    weights = weights or {}
    candidates: dict[str, Candidate] = {}
    for channel, ranking in rankings.items():
        weight = weights.get(channel, 1.0)
        for rank, (chunk_id, raw_score) in enumerate(ranking, 1):
            candidate = candidates.setdefault(chunk_id, Candidate(chunk_id))
            candidate.score += weight / (constant + rank)
            candidate.channel_scores[channel] = raw_score
            candidate.channel_ranks[channel] = rank
    return sorted(candidates.values(), key=lambda item: (-item.score, item.chunk_id))[:top_k]


def apply_legal_metadata_boost(
    question: str,
    candidates: list[Candidate],
    chunk_by_id: dict[str, LegalChunk],
    boost: float,
) -> list[Candidate]:
    citations = extract_citations(question)
    for candidate in candidates:
        chunk = chunk_by_id[candidate.chunk_id]
        matches = 0
        for citation in citations:
            if citation.document_number and citation.document_number == chunk.document_number:
                matches += 2
            if citation.article and citation.article == chunk.article:
                matches += 1
            if citation.clause and citation.clause == chunk.clause:
                matches += 1
        if matches:
            candidate.score *= 1.0 + boost * matches
            candidate.channel_scores["metadata_matches"] = float(matches)
    return sorted(candidates, key=lambda item: (-item.score, item.chunk_id))


def build_retrieval_indexes(
    config: PipelineConfig,
    build_dense: bool = True,
    *,
    rebuild_store: bool = True,
    resume_dense: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    output = artifact_dir(config)
    from .store import CorpusStore

    notify = progress or (lambda message: None)
    store_path = output / "corpus.sqlite"
    if rebuild_store or not store_path.exists():
        notify("Building SQLite FTS5 BM25 store")
        store_report = CorpusStore.build(
            store_path, output / "chunks.jsonl", output / "parents.jsonl"
        )
    else:
        build = __import__("json").loads((output / "build_report.json").read_text(encoding="utf-8"))
        store_report = {
            "backend": "sqlite_fts5_bm25",
            "chunks": int(build["chunks"]),
            "parents": int(build["parents"]),
        }
        notify("Reusing frozen SQLite FTS5 BM25 store")
    report: dict[str, object] = dict(store_report)
    if config.retrieval.bm25_enabled:
        report["bm25"] = "SQLite FTS5 bm25 (disk-backed)"
    if build_dense and config.retrieval.dense_enabled and config.models.dense.enabled:
        final_embeddings_path = output / "dense_embeddings.npy"
        partial_embeddings_path = output / "dense_embeddings.partial.npy"
        progress_path = output / "dense_build_progress.json"
        if final_embeddings_path.exists() and not resume_dense:
            raise FileExistsError(
                f"{final_embeddings_path} already exists; keep it or explicitly resume an incomplete build"
            )
        encoder = DenseEncoder(config.models.dense)
        chunk_count = int(store_report["chunks"])
        ids: list[str] = []
        memory_map = None
        offset = 0
        initial_offset = 0
        try:
            import faiss
        except ImportError:
            faiss = None  # type: ignore[assignment]
        signature = {
            "model": config.models.dense.name_or_path,
            "max_length": config.models.dense.max_length,
            "query_prefix": config.models.dense.query_prefix,
            "document_prefix": config.models.dense.document_prefix,
            "chunk_count": chunk_count,
            "batch_order": "sqlite_length_ascending_rowid",
        }
        if resume_dense and partial_embeddings_path.exists() and progress_path.exists():
            saved = __import__("json").loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("signature") != signature:
                raise ValueError("Dense checkpoint does not match current model/corpus configuration")
            memory_map = np.load(partial_embeddings_path, mmap_mode="r+")
            offset = int(saved["offset"])
            initial_offset = offset
            notify(f"Resuming dense encoding at {offset}/{chunk_count}")
        elif partial_embeddings_path.exists() or progress_path.exists():
            raise FileExistsError("Incomplete dense artifacts exist; rerun with --resume-dense")
        text_batch: list[str] = []
        id_batch: list[str] = []
        started = time.perf_counter()
        checkpoint_stride = max(config.models.dense.batch_size, 1024)
        next_checkpoint = ((offset // checkpoint_stride) + 1) * checkpoint_stride

        def encode_batch() -> None:
            nonlocal memory_map, offset, next_checkpoint
            if not text_batch:
                return
            values = encoder.encode_documents(text_batch).astype(np.float32)
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            if memory_map is None:
                memory_map = np.lib.format.open_memmap(
                    partial_embeddings_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(chunk_count, values.shape[1]),
                )
            elif memory_map.shape[1] != values.shape[1]:
                raise ValueError("Embedding dimension changed while resuming dense build")
            memory_map[offset : offset + len(values)] = values
            offset += len(values)
            text_batch.clear()
            id_batch.clear()
            if offset >= next_checkpoint or offset == chunk_count:
                memory_map.flush()
                write_json(progress_path, {"signature": signature, "offset": offset})
                elapsed = max(time.perf_counter() - started, 1e-9)
                processed_this_run = max(1, offset - initial_offset)
                rate = processed_this_run / elapsed
                eta_hours = (chunk_count - offset) / max(rate, 1e-9) / 3600.0
                notify(
                    f"Dense encoded {offset}/{chunk_count} chunks "
                    f"({rate:.2f} chunks/s, ETA {eta_hours:.2f}h)"
                )
                next_checkpoint = ((offset // checkpoint_stride) + 1) * checkpoint_stride

        # Exact retrieval is invariant to vector row order. Grouping similarly
        # sized texts reduces Transformer padding substantially on CPU while
        # rowid keeps the order deterministic for checkpoint/resume.
        dense_store = CorpusStore(store_path)
        try:
            rows = dense_store.connection.execute(
                "SELECT id, retrieval_text FROM chunks ORDER BY length(retrieval_text), rowid"
            )
            for index, value in enumerate(rows):
                chunk_id = str(value["id"])
                ids.append(chunk_id)
                if index < offset:
                    continue
                text_batch.append(str(value["retrieval_text"]))
                id_batch.append(chunk_id)
                if len(text_batch) >= config.models.dense.batch_size:
                    encode_batch()
        finally:
            dense_store.close()
        encode_batch()
        if memory_map is None or offset != chunk_count or len(ids) != chunk_count:
            raise ValueError(
                f"Dense build incomplete: offset={offset}, ids={len(ids)}, expected={chunk_count}"
            )
        memory_map.flush()
        dense_shape = list(memory_map.shape)
        del memory_map
        partial_embeddings_path.replace(final_embeddings_path)
        report["dense_shape"] = dense_shape
        write_json(output / "dense_chunk_ids.json", ids)
        if faiss is not None:
            embeddings = np.load(final_embeddings_path, mmap_mode="r")
            faiss_index = faiss.IndexFlatIP(int(embeddings.shape[1]))
            for start in range(0, chunk_count, 50_000):
                faiss_index.add(np.asarray(embeddings[start : start + 50_000], dtype=np.float32))
            faiss.write_index(faiss_index, str(output / "dense.faiss"))
            report["dense_index"] = "faiss.IndexFlatIP"
        else:
            report["dense_index"] = "numpy_exact_fallback"
        report["dense_model"] = config.models.dense.name_or_path
        report["dense_seconds"] = round(time.perf_counter() - started, 3)
        if progress_path.exists():
            progress_path.unlink()
    write_json(output / "retrieval_index_report.json", report)
    return report


class HybridRetriever:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        output = artifact_dir(config)
        from .store import CorpusStore

        store_path = output / "corpus.sqlite"
        if not store_path.exists():
            raise FileNotFoundError(f"Missing {store_path}; run build-index first")
        self.store = CorpusStore(store_path)
        self.dense_encoder = None
        self.dense_index = None
        embeddings_path = output / "dense_embeddings.npy"
        if config.retrieval.dense_enabled and config.models.dense.enabled and embeddings_path.exists():
            self.dense_encoder = DenseEncoder(config.models.dense)
            chunk_ids = __import__("json").loads((output / "dense_chunk_ids.json").read_text(encoding="utf-8"))
            faiss_path = output / "dense.faiss"
            self.dense_index = (
                FaissDenseIndex(faiss_path, chunk_ids)
                if faiss_path.exists()
                else DenseIndex(np.load(embeddings_path, mmap_mode="r"), chunk_ids)
            )
        if config.retrieval.rerank_enabled:
            self.reranker: Reranker = (
                CrossEncoderReranker(config.models.reranker)
                if config.models.reranker.enabled and config.models.reranker.backend == "cross_encoder"
                else LexicalReranker()
            )
        else:
            self.reranker = LexicalReranker()

    def retrieve(self, question: str) -> list[Candidate]:
        rankings: dict[str, list[tuple[str, float]]] = {}
        if self.config.retrieval.bm25_enabled:
            rankings["bm25"] = self.store.search_bm25(question, self.config.retrieval.bm25_top_k)
        if self.dense_encoder is not None and self.dense_index is not None:
            query_embedding = self.dense_encoder.encode_query(question)
            rankings["dense"] = self.dense_index.search_vector(query_embedding, self.config.retrieval.dense_top_k)
        fused = reciprocal_rank_fusion(
            rankings,
            weights={"bm25": self.config.retrieval.bm25_weight, "dense": self.config.retrieval.dense_weight},
            constant=self.config.retrieval.rrf_constant,
            top_k=self.config.retrieval.rrf_top_k,
        )
        chunks = self.store.get_chunks(candidate.chunk_id for candidate in fused)
        boosted = apply_legal_metadata_boost(question, fused, chunks, self.config.retrieval.legal_metadata_boost)
        rerank_set = boosted[: self.config.retrieval.rerank_top_k]
        if self.config.retrieval.rerank_enabled and rerank_set:
            rerank_chunks = [chunks[item.chunk_id] for item in rerank_set]
            scores = self.reranker.score(question, rerank_chunks)
            for candidate, score in zip(rerank_set, scores):
                candidate.channel_scores["reranker"] = score
                candidate.score = score
            rerank_set.sort(key=lambda item: (-item.score, item.chunk_id))
        return rerank_set

    def get_chunk(self, chunk_id: str) -> LegalChunk:
        return self.store.get_chunk(chunk_id)
