from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legalqa.core.config import (
    AuditConfig,
    ChunkingConfig,
    ComplexityConfig,
    ModelsConfig,
    ModelSpec,
    PipelineConfig,
    SanitationConfig,
)
from legalqa.core.metadata import canonical_document_number, extract_citations
from legalqa.core.normalize import normalize_text
from legalqa.core.schema import LegalChunk, LegalDocument, ParentSection
from legalqa.evaluation.metrics import meteor_exact, rouge_l_f1
from legalqa.evaluation.submission import validate_prediction_file
from legalqa.generation.pipeline import LegalQAPipeline
from legalqa.preprocessing.audit import audit_corpus
from legalqa.preprocessing.chunking import LegalAwareChunker
from legalqa.preprocessing.release import (
    _build_citation_qrels,
    _chunk_structure_bucket,
    _iter_canonical_chunks,
    _question_group_id,
    _split_groups,
    _valid_logical_parent_id,
)
from legalqa.preprocessing.sanitation import (
    PAYWALL_RE,
    CorpusSanitizer,
    local_repetition_metrics,
)
from legalqa.retrieval.complexity import (
    analyze_query_complexity,
    analyze_retrieval_confidence,
    choose_dynamic_top_k,
)
from legalqa.retrieval.engine import BM25Index, reciprocal_rank_fusion
from legalqa.retrieval.graph import build_citation_graph
from legalqa.retrieval.neighbors import predict_from_training_neighbors
from legalqa.retrieval.store import CorpusStore


class NormalizeTests(unittest.TestCase):
    def test_conservative_normalization(self) -> None:
        value = normalize_text("Điều  3.\r\n  Có hiệu lực từ...\n\n\nSửa đổi 90 / 2017 / NĐ - CP")
        self.assertIn("Điều 3.", value)
        self.assertIn("Có hiệu lực", value)
        self.assertIn("Sửa đổi 90/2017/NĐ-CP", value)


class CitationTests(unittest.TestCase):
    def test_extracts_multiple_clauses_and_document(self) -> None:
        citations = extract_citations(
            "Căn cứ điểm a khoản 3, khoản 5 Điều 17 Nghị định 90/2017/NĐ-CP."
        )
        self.assertEqual({value.clause for value in citations}, {"3", "5"})
        self.assertTrue(all(value.document_number == "90/2017/NĐ-CP" for value in citations))
        self.assertEqual(citations[0].article, "17")


class AuditTests(unittest.TestCase):
    def test_empty_removed_exact_collapsed_near_kept(self) -> None:
        documents = [
            LegalDocument("1", "A", "Điều 1. Nội dung", source_ids=("1",)),
            LegalDocument("2", "B", "Điều 1. Nội dung", source_ids=("2",)),
            LegalDocument("3", "C", "", source_ids=("3",)),
            LegalDocument("4", "D", "Điều 1. Nội dung gần giống", source_ids=("4",)),
        ]
        result = audit_corpus(
            documents,
            AuditConfig(near_duplicate_enabled=True, near_duplicate_threshold=0.8),
        )
        self.assertEqual(len(result.documents), 2)
        self.assertEqual(result.report["empty_removed"], 1)
        self.assertEqual(result.report["exact_duplicates_removed"], 1)
        self.assertEqual(result.report["near_duplicates_removed"], 0)
        self.assertEqual(result.documents[0].source_ids, ("1", "2"))


class ChunkingTests(unittest.TestCase):
    def test_hierarchy_and_parent_child(self) -> None:
        document = LegalDocument(
            id="153",
            name="Nghị định 90/2017/NĐ-CP",
            passage=(
                "CHƯƠNG III\nMỤC 2\nĐiều 17. Vi phạm\n"
                "1. Hành vi thứ nhất bị xử phạt theo quy định.\n"
                "2. Hành vi thứ hai:\na) Trường hợp A;\nb) Trường hợp B.\n"
                "Điều 18. Hiệu lực\n1. Có hiệu lực từ ngày công bố."
            ),
            metadata={"document_type": "Nghị định", "document_number": "90/2017/NĐ-CP"},
        )
        result = LegalAwareChunker(ChunkingConfig(max_chars=500, overlap_chars=50, min_chars=10)).chunk_document(document)
        article_17 = [parent for parent in result.parents if parent.article == "17"]
        article_chunks = [chunk for chunk in result.chunks if chunk.article == "17"]
        self.assertEqual(len(article_17), 1)
        self.assertTrue(any("a) Trường hợp A" in chunk.retrieval_text for chunk in article_chunks))
        self.assertTrue(any("b) Trường hợp B" in chunk.retrieval_text for chunk in article_chunks))
        self.assertTrue(all(chunk.parent_id == article_17[0].id for chunk in article_chunks))

    def test_retrieval_children_and_generation_parents_are_distinct(self) -> None:
        document = LegalDocument(
            "dual",
            "Nghi dinh thu nghiem",
            "\u0110i\u1ec1u 1. Tieu de\nNoi dung day du",
            retrieval_text="\u0110i\u1ec1u 1. Tieu de\nNoi dung sach cho retrieval",
            generation_text="\u0110i\u1ec1u 1. Tieu de\nNoi dung sach va ngu canh day du cho generation",
        )
        result = LegalAwareChunker(
            ChunkingConfig(max_chars=500, overlap_chars=50, min_chars=5)
        ).chunk_document(document)
        self.assertIn("ngu canh day du", result.parents[0].text)
        self.assertNotIn("ngu canh day du", result.chunks[0].retrieval_text)


class SanitationTests(unittest.TestCase):
    BANNER = (
        "B\u1ea1n ph\u1ea3i \u0111\u0103ng nh\u1eadp ho\u1eb7c \u0111\u0103ng k\u00fd Th\u00e0nh Vi\u00ean TVPL Pro \u0111\u1ec3 s\u1eed d\u1ee5ng "
        "\u0111\u01b0\u1ee3c \u0111\u1ea7y \u0111\u1ee7 c\u00e1c ti\u1ec7n \u00edch gia t\u0103ng li\u00ean quan \u0111\u1ebfn n\u1ed9i dung TCVN. "
        "M\u1ecdi chi ti\u1ebft xin li\u00ean h\u1ec7: \u0110T: (028) 3930 3279 D\u0110: 0906 22 99 66."
    )

    def _sanitizer(self) -> CorpusSanitizer:
        return CorpusSanitizer(SanitationConfig(enabled=True, version="test-v2"))

    def test_paywall_recovery_and_terminal_noi_nhan(self) -> None:
        repeated = "Noi dung ky thuat " + "rat dai va can duoc giu nguyen " * 8
        noi_nhan = "N\u01a1i nh\u1eadn:"
        raw = f"{repeated}\n\n{self.BANNER}\n\n{repeated}\n\n{self.BANNER}\n\n{noi_nhan}\n- Co quan A"
        self.assertEqual(len(PAYWALL_RE.findall(raw)), 2)
        result = self._sanitizer().sanitize(LegalDocument("x", "TCVN", raw))
        self.assertNotIn("TVPL Pro", result.generation_text)
        self.assertEqual(result.generation_text.count("Noi dung ky thuat"), 1)
        self.assertIn(noi_nhan, result.generation_text)
        self.assertNotIn(noi_nhan, result.retrieval_text)
        self.assertTrue(result.metadata["sanitation"]["noi_nhan_removed"])

    def test_nonterminal_noi_nhan_and_form_dots_are_preserved(self) -> None:
        noi_nhan = "N\u01a1i nh\u1eadn:"
        body = f"{noi_nhan} la truong trong mau don.\n" + "Noi dung hop le .......... " * 20
        result = self._sanitizer().sanitize(LegalDocument("y", "Mau don", body))
        self.assertIn(noi_nhan, result.retrieval_text)
        self.assertIn("..........", result.retrieval_text)
        metrics = local_repetition_metrics(("." * 150 + "\n\n") * 30)
        self.assertEqual(metrics["block_count"], 0)
        self.assertEqual(metrics["max_block_frequency"], 0)

    def test_web_code_cleanup_is_paywall_scoped(self) -> None:
        legal = "\u0110i\u1ec1u 1. Noi dung phap ly can giu nguyen."
        code = '$(document).ready(function () { $(this).html("TVPL Pro"); });'
        raw = f"{legal}\n{self.BANNER}\n{code}\nM\u1ee4C L\u1ee4C V\u0102N B\u1ea2N"
        cleaned = self._sanitizer().sanitize(LegalDocument("web", "TCVN", raw))
        self.assertIn(legal, cleaned.generation_text)
        self.assertNotIn("$(document)", cleaned.generation_text)
        self.assertNotIn("M\u1ee4C L\u1ee4C V\u0102N B\u1ea2N", cleaned.generation_text)
        self.assertGreater(cleaned.metadata["sanitation"]["web_artifact_lines_removed"], 0)

        unaffected = self._sanitizer().sanitize(LegalDocument("plain", "Law", f"{legal}\n{code}"))
        self.assertIn("$(document)", unaffected.generation_text)


class DynamicKTests(unittest.TestCase):
    def test_confident_elbow_reduces_k(self) -> None:
        config = ComplexityConfig()
        complexity = analyze_query_complexity("Mức phạt hành vi X là bao nhiêu?", config)
        confidence = analyze_retrieval_confidence([0.92, 0.89, 0.34, 0.21])
        self.assertEqual(confidence.elbow_position, 2)
        self.assertEqual(choose_dynamic_top_k(complexity, confidence, config), 2)

    def test_flat_scores_expand_k(self) -> None:
        config = ComplexityConfig()
        complexity = analyze_query_complexity("Mức phạt hành vi X là bao nhiêu?", config)
        confidence = analyze_retrieval_confidence([0.81, 0.79, 0.76, 0.73, 0.70])
        self.assertGreaterEqual(choose_dynamic_top_k(complexity, confidence, config), 5)


class RetrievalTests(unittest.TestCase):
    def _chunk(self, chunk_id: str, text: str) -> LegalChunk:
        return LegalChunk(chunk_id, "d", "p", text, "fixed", "h")

    def test_bm25_and_rrf(self) -> None:
        index = BM25Index.build(
            [self._chunk("a", "mức phạt giao thông"), self._chunk("b", "đăng ký doanh nghiệp")]
        )
        self.assertEqual(index.search("mức phạt", 1)[0][0], "a")
        fused = reciprocal_rank_fusion(
            {"bm25": [("a", 2.0), ("b", 1.0)], "dense": [("b", 0.9), ("a", 0.8)]},
            top_k=2,
        )
        self.assertEqual({value.chunk_id for value in fused}, {"a", "b"})

    def test_disk_backed_fts5_bm25_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [
                self._chunk("a", "mức phạt giao thông"),
                self._chunk("b", "đăng ký doanh nghiệp"),
            ]
            parent = ParentSection("p", "d", "Điều 1. Nội dung", "article", "Điều 1")
            (root / "chunks.jsonl").write_text(
                "\n".join(json.dumps(value.to_dict(), ensure_ascii=False) for value in chunks) + "\n",
                encoding="utf-8",
            )
            (root / "parents.jsonl").write_text(
                json.dumps(parent.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8"
            )
            report = CorpusStore.build(root / "corpus.sqlite", root / "chunks.jsonl", root / "parents.jsonl")
            store = CorpusStore(root / "corpus.sqlite")
            self.assertEqual(report["chunks"], 2)
            self.assertEqual(store.search_bm25("mức phạt", 1)[0][0], "a")
            self.assertEqual(store.get_parent("p").article, None)
            store.close()

    def test_release_chunks_build_fts5_store_and_materialize_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "parent_chunk_id": "doc_d1_article_1",
                    "retrieval_text": "Document > Article 1\ntraffic fine",
                    "parent_text": "Article 1 parent context",
                    "level": "clause",
                    "doc_type": "decree",
                    "doc_number_canonical": "1/2020/ND-CP",
                    "chapter": None,
                    "article": "1",
                    "clause": "1",
                    "point": None,
                    "source_start": 0,
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "d1",
                    "parent_chunk_id": "doc_d1_article_1",
                    "retrieval_text": "Document > Article 1\nbusiness registration",
                    "parent_text": "Article 1 parent context",
                    "level": "clause",
                    "doc_type": "decree",
                    "doc_number_canonical": "1/2020/ND-CP",
                    "chapter": None,
                    "article": "1",
                    "clause": "2",
                    "point": None,
                    "source_start": 20,
                },
            ]
            chunks_path = root / "release_chunks.jsonl"
            chunks_path.write_text(
                "\n".join(json.dumps(value) for value in records) + "\n",
                encoding="utf-8",
            )
            report = CorpusStore.build_release(root / "release.sqlite", chunks_path)
            store = CorpusStore(root / "release.sqlite")
            try:
                self.assertEqual(report["chunks"], 2)
                self.assertEqual(report["parents"], 1)
                self.assertEqual(store.search_bm25("traffic fine", 1)[0][0], "c1")
                self.assertEqual(store.get_parent("doc_d1_article_1").article, "1")
            finally:
                store.close()


class GraphTests(unittest.TestCase):
    def test_reference_and_reverse_amendment_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parents = [
                ParentSection("p1", "d1", "Điều 10. Gốc", "article", "Điều 10", document_number="1/2020/NĐ-CP", article="10"),
                ParentSection(
                    "p2",
                    "d2",
                    "Điều 1. Sửa đổi Điều 10 Nghị định 1/2020/NĐ-CP",
                    "article",
                    "Điều 1",
                    document_number="2/2021/NĐ-CP",
                    article="1",
                    references=("1/2020/NĐ-CP|10||",),
                    amendments=("1/2020/NĐ-CP|10||",),
                ),
            ]
            path = root / "parents.jsonl"
            path.write_text("\n".join(json.dumps(value.to_dict(), ensure_ascii=False) for value in parents) + "\n", encoding="utf-8")
            build_citation_graph(path, root / "graph.json")
            graph = json.loads((root / "graph.json").read_text(encoding="utf-8"))
            self.assertIn({"target": "p1", "relation": "AMENDS"}, graph["adjacency"]["p2"])
            self.assertIn({"target": "p2", "relation": "AMENDED_BY"}, graph["adjacency"]["p1"])


class ComplianceAndMetricsTests(unittest.TestCase):
    def test_strict_model_budget(self) -> None:
        models = ModelsConfig(
            dense=ModelSpec(parameters_b=0.6, enabled=True, backend="tfidf"),
            reranker=ModelSpec(parameters_b=0.6, enabled=True, backend="lexical"),
            generator=ModelSpec(parameters_b=2.8, enabled=True, backend="extractive"),
        )
        with self.assertRaises(ValueError):
            models.validate_budget()

    def test_retrieval_model_fallbacks_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "models.dense.enabled"):
            PipelineConfig().validate()

    def test_submission_requires_real_generator(self) -> None:
        config = PipelineConfig(
            sanitation=SanitationConfig(enabled=True),
            models=ModelsConfig(
                dense=ModelSpec(
                    name_or_path="dense-model",
                    parameters_b=0.1,
                    enabled=True,
                    backend="sentence_transformers",
                ),
                reranker=ModelSpec(
                    name_or_path="reranker-model",
                    parameters_b=0.1,
                    enabled=True,
                    backend="cross_encoder",
                ),
                generator=ModelSpec(enabled=False, backend="extractive"),
            ),
        )
        config.validate()
        with self.assertRaisesRegex(ValueError, "models.generator.enabled"):
            config.validate_submission()

    def test_metrics_identity(self) -> None:
        self.assertAlmostEqual(rouge_l_f1("a b c", "a b c"), 1.0)
        self.assertGreater(meteor_exact("a b c", "a b c"), 0.98)


class PredictionCheckpointTests(unittest.TestCase):
    @staticmethod
    def _pipeline(calls: list[str]) -> LegalQAPipeline:
        pipeline = object.__new__(LegalQAPipeline)
        pipeline.config = PipelineConfig()

        def answer(question: str) -> dict[str, object]:
            calls.append(question)
            return {"answer": f"answer: {question}", "decision": {}, "evidence": []}

        pipeline.answer = answer  # type: ignore[method-assign]
        return pipeline

    def test_prediction_checkpoint_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "predictions.json"
            input_path.write_text(
                json.dumps(
                    {
                        "1": {"question": "Question one", "answer": None},
                        "2": {"question": "Question two", "answer": None},
                    }
                ),
                encoding="utf-8",
            )
            output_path.write_text(
                json.dumps({"1": {"question": "Question one", "answer": "existing"}}),
                encoding="utf-8",
            )
            calls: list[str] = []
            progress: list[str] = []
            report = self._pipeline(calls).predict_file(
                input_path,
                output_path,
                resume=True,
                checkpoint_every=1,
                progress=progress.append,
            )
            predictions = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(calls, ["Question two"])
            self.assertEqual(set(predictions), {"1", "2"})
            self.assertEqual(report["generated"], 1)
            self.assertEqual(report["reused"], 1)
            self.assertTrue(report["complete"])
            self.assertEqual(len(progress), 2)

    def test_prediction_rejects_invalid_checkpoint_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                self._pipeline([]).predict_file(path, path.with_name("out.json"), checkpoint_every=0)


class NeighborPredictionTests(unittest.TestCase):
    def test_bm25_neighbor_prediction_preserves_submission_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.json"
            input_path = root / "test.json"
            output_path = root / "submission.json"
            trace_path = root / "trace.json"
            train_path.write_text(
                json.dumps(
                    {
                        "a": {"question": "traffic fine", "answer": "answer a"},
                        "b": {"question": "business registration", "answer": "answer b"},
                    }
                ),
                encoding="utf-8",
            )
            input_path.write_text(
                json.dumps({"x": {"question": "traffic fine", "answer": None}}),
                encoding="utf-8",
            )
            report = predict_from_training_neighbors(
                train_path,
                input_path,
                output_path,
                strategy="bm25",
                trace_path=trace_path,
            )
            submission = json.loads(output_path.read_text(encoding="utf-8"))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(
                submission,
                {"x": {"question": "traffic fine", "answer": "answer a"}},
            )
            self.assertEqual(trace["x"]["neighbor_id"], "a")
            self.assertTrue(report["complete"])

    def test_dense_neighbor_prediction_requires_enabled_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "empty.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                predict_from_training_neighbors(path, path, root / "out.json", strategy="rrf")


class SubmissionValidationTests(unittest.TestCase):
    def test_reports_missing_blank_and_mismatched_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            prediction_path = root / "predictions.json"
            input_path.write_text(
                json.dumps(
                    {
                        "1": {"question": "First", "answer": None},
                        "2": {"question": "Second", "answer": None},
                        "3": {"question": "Third", "answer": None},
                    }
                ),
                encoding="utf-8",
            )
            prediction_path.write_text(
                json.dumps(
                    {
                        "1": {"question": "Wrong", "answer": "answer"},
                        "2": {"question": "Second", "answer": " "},
                        "extra": {"question": "Extra", "answer": "answer"},
                    }
                ),
                encoding="utf-8",
            )
            report = validate_prediction_file(input_path, prediction_path)
            self.assertFalse(report["valid"])
            self.assertEqual(report["missing_count"], 1)
            self.assertEqual(report["extra_count"], 1)
            self.assertEqual(report["question_mismatch_count"], 1)
            self.assertEqual(report["blank_answer_count"], 1)


class ReleaseContractTests(unittest.TestCase):
    def test_release_chunk_audit_taxonomy_and_parent_ownership(self) -> None:
        self.assertEqual(_chunk_structure_bucket("article_group"), "article")
        self.assertEqual(_chunk_structure_bucket("point"), "clause")
        self.assertEqual(_chunk_structure_bucket("fixed"), "fallback")
        self.assertTrue(_valid_logical_parent_id("740", "doc_740_article_1"))
        self.assertFalse(_valid_logical_parent_id("740", "doc_741_article_1"))

    def test_citation_report_states_pair_and_question_denominators(self) -> None:
        answer = (
            "CÄƒn cá»© Ä‘iá»ƒm a khoáº£n 3, khoáº£n 5 Äiá»u 17 "
            "Nghá»‹ Ä‘á»‹nh 90/2017/NÄ-CP."
        )
        answer = (
            "C\u0103n c\u1ee9 \u0111i\u1ec3m a kho\u1ea3n 3, kho\u1ea3n 5 "
            "\u0110i\u1ec1u 17 Ngh\u1ecb \u0111\u1ecbnh 90/2017/N\u0110-CP."
        )
        citations = extract_citations(answer)
        number = canonical_document_number(citations[0].document_number)
        corpus = {
            "article_index": {},
            "clause_index": {
                (number, "17", "3"): ["chunk-3"],
                (number, "17", "5"): ["chunk-5"],
            },
            "point_index": {},
        }
        validation = [
            {"question_id": "matched", "answer_model": answer, "question_model": "Q1"},
            {"question_id": "plain", "answer_model": "KhÃ´ng trÃ­ch dáº«n", "question_model": "Q2"},
            {
                "question_id": "unmatched",
                "answer_model": "Khoáº£n 1 Äiá»u 2 Nghá»‹ Ä‘á»‹nh 1/2020/NÄ-CP.",
                "question_model": "Q3",
            },
        ]
        validation[2]["answer_model"] = (
            "\u0110i\u1ec3m a kho\u1ea3n 1 \u0110i\u1ec1u 2 "
            "Ngh\u1ecb \u0111\u1ecbnh 1/2020/N\u0110-CP."
        )
        with tempfile.TemporaryDirectory() as directory:
            result = _build_citation_qrels(
                validation,
                corpus,
                Path(directory),
                seed=2026,
                manual_sample_size=0,
            )
        report = result["report"]
        self.assertEqual(report["answers_scanned"], 3)
        self.assertEqual(report["answers_with_citations"], 2)
        self.assertEqual(report["questions_with_at_least_one_match"], 1)
        self.assertEqual(report["unique_question_chunk_pairs"], 2)
        self.assertEqual(report["qrel_pair_counts_by_confidence"]["HIGH"], 2)
        self.assertEqual(report["unmatched_questions_with_citations"], 1)
        self.assertEqual(report["judged_coverage_basis"]["formula"], "1/3")

    def test_group_split_keeps_question_templates_together(self) -> None:
        records = []
        for index, question in enumerate(
            (
                "Mức phạt theo Nghị định 90/2017/NĐ-CP là bao nhiêu?",
                "Mức phạt theo Nghị định 91/2018/NĐ-CP là bao nhiêu?",
                "Thủ tục đăng ký doanh nghiệp",
                "Điều kiện cấp giấy phép",
            )
        ):
            records.append(
                {
                    "question_id": str(index),
                    "question_group": _question_group_id(question),
                    "split": "",
                }
            )
        training, validation = _split_groups(records, validation_size=2, seed=2026)
        train_groups = {value["question_group"] for value in training}
        validation_groups = {value["question_group"] for value in validation}
        self.assertFalse(train_groups & validation_groups)
        first_split = next(value["split"] for value in records if value["question_id"] == "0")
        second_split = next(value["split"] for value in records if value["question_id"] == "1")
        self.assertEqual(first_split, second_split)

    def test_canonical_chunks_retain_exact_raw_spans(self) -> None:
        passage = (
            "Mở đầu văn bản\n"
            "Điều 1. Quy định chung\n"
            "1. Nội dung khoản một.\n"
            "2. Nội dung khoản hai.\n"
            "Điều 2. Hiệu lực\n"
            "1. Có hiệu lực từ hôm nay."
        )
        chunks = list(
            _iter_canonical_chunks(
                doc_id="740",
                name="Nghị định 90/2017/NĐ-CP",
                passage=passage,
                source_sha256="abc",
                quality_flags=[],
                max_chars=80,
                overlap_chars=10,
            )
        )
        self.assertTrue(chunks)
        self.assertTrue(any(value["article"] == "1" for value in chunks))
        self.assertTrue(any("2" in value["contained_clauses"] for value in chunks))
        for value in chunks:
            self.assertEqual(
                value["raw_text"],
                passage[value["source_start"] : value["source_end"]],
            )

    def test_article_suffixes_produce_unique_chunk_ids(self) -> None:
        passage = "Điều 98. Nội dung A\nĐiều 98đ. Nội dung B"
        chunks = list(
            _iter_canonical_chunks(
                doc_id="x",
                name="Luật thử nghiệm",
                passage=passage,
                source_sha256="abc",
                quality_flags=[],
                max_chars=1800,
                overlap_chars=220,
            )
        )
        ids = [value["chunk_id"] for value in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(any("article_98dd" in value for value in ids))


if __name__ == "__main__":
    unittest.main()
