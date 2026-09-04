from __future__ import annotations

import argparse
import json

from .core.config import artifact_dir, data_path, load_config
from .evaluation import (
    build_diagnostic_candidate_cache,
    build_diagnostic_labels,
    evaluate_cached_qa,
    evaluate_diagnostic_evidence,
    evaluate_diagnostic_retrieval,
    evaluate_qa_files,
    evaluate_retrieval_complementarity,
    tune_dynamic_k_rules,
    validate_prediction_file,
)
from .generation import LegalQAPipeline
from .preprocessing import (
    analyze_dataset,
    audit_sanitation,
    build_corpus,
    build_data_release,
    build_data_release_index,
    freeze_corpus,
    refresh_data_release_reports,
    release_contract,
    review_boilerplate_candidates,
    tokenize_data_release,
    validate_data_release,
)
from .retrieval import (
    HybridRetriever,
    benchmark_dense_throughput,
    build_citation_graph,
    build_retrieval_indexes,
    predict_from_training_neighbors,
)


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UIT DSC 2026 LegalQA pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-config")
    subparsers.add_parser("release-contract")
    release = subparsers.add_parser("build-release")
    release.add_argument("--release-root", required=True)
    release.add_argument("--validation-size", type=int, default=700)
    release.add_argument(
        "--tokenizer-model", default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    refresh_release = subparsers.add_parser("refresh-release-reports")
    refresh_release.add_argument("--release-root", required=True)
    tokenize_release = subparsers.add_parser("tokenize-release")
    tokenize_release.add_argument("--release-root", required=True)
    tokenize_release.add_argument(
        "--tokenizer-model", default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    release_index = subparsers.add_parser("build-release-index")
    release_index.add_argument("--release-root", required=True)
    preflight = subparsers.add_parser("preflight-release")
    preflight.add_argument("--release-root", required=True)
    preflight.add_argument("--profile", choices=("e0-direct", "e1-bm25"), required=True)
    preflight.add_argument(
        "--stage", choices=("training", "evaluation", "public", "promotion"), required=True
    )
    preflight.add_argument("--report")
    subparsers.add_parser("eda")
    subparsers.add_parser("audit-sanitation")
    subparsers.add_parser("review-boilerplate")
    subparsers.add_parser("freeze-corpus")
    corpus = subparsers.add_parser("build-corpus")
    corpus.add_argument("--reuse-audit", action="store_true")
    subparsers.add_parser("build-diagnostics")
    index = subparsers.add_parser("build-index")
    index.add_argument("--skip-dense", action="store_true")
    index.add_argument("--reuse-bm25", action="store_true")
    index.add_argument("--resume-dense", action="store_true")
    dense_benchmark = subparsers.add_parser("benchmark-dense")
    dense_benchmark.add_argument("--max-chunks", type=int, default=64)
    subparsers.add_parser("build-graph")
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("question")
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input")
    predict.add_argument("--output", required=True)
    predict.add_argument("--trace")
    predict.add_argument("--limit", type=int)
    predict.add_argument("--resume", action="store_true")
    predict.add_argument("--checkpoint-every", type=int, default=1)
    neighbor = subparsers.add_parser("predict-neighbor")
    neighbor.add_argument("--input")
    neighbor.add_argument("--output", required=True)
    neighbor.add_argument("--trace")
    neighbor.add_argument("--strategy", choices=("bm25", "dense", "rrf"), default="rrf")
    neighbor.add_argument("--top-k", type=int, default=20)
    neighbor.add_argument("--dense-weight", type=float, default=4.0)
    neighbor.add_argument("--limit", type=int)
    evaluate_retrieval = subparsers.add_parser("evaluate-retrieval")
    evaluate_retrieval.add_argument("--max-examples", type=int)
    evaluate_retrieval.add_argument("--output")
    complementarity = subparsers.add_parser("evaluate-complementarity")
    complementarity.add_argument("--max-examples", type=int)
    complementarity.add_argument("--limit", type=int, default=10)
    complementarity.add_argument("--output")
    cache = subparsers.add_parser("build-diagnostic-cache")
    cache.add_argument("--max-examples", type=int)
    cache.add_argument("--output")
    evidence = subparsers.add_parser("evaluate-evidence")
    evidence.add_argument("--cache", required=True)
    evidence.add_argument("--output")
    evidence.add_argument("--dynamic-k", action="store_true")
    evidence.add_argument("--parent", action="store_true")
    evidence.add_argument("--graph", action="store_true")
    tune_dynamic = subparsers.add_parser("tune-dynamic-k")
    tune_dynamic.add_argument("--cache", required=True)
    tune_dynamic.add_argument("--output")
    cached_qa = subparsers.add_parser("evaluate-cached-qa")
    cached_qa.add_argument("--cache", required=True)
    cached_qa.add_argument("--output", required=True)
    cached_qa.add_argument("--predictions")
    cached_qa.add_argument("--max-examples", type=int)
    cached_qa.add_argument("--dynamic-k", action="store_true")
    cached_qa.add_argument("--parent", action="store_true")
    cached_qa.add_argument("--graph", action="store_true")
    evaluate_qa = subparsers.add_parser("evaluate-qa")
    evaluate_qa.add_argument("--predictions", required=True)
    evaluate_qa.add_argument("--references", required=True)
    validate_predictions = subparsers.add_parser("validate-predictions")
    validate_predictions.add_argument("--input")
    validate_predictions.add_argument("--predictions", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output = artifact_dir(config)
    if args.command == "validate-config":
        _print({"valid": True, "total_parameters_b": config.models.total_parameters_b})
    elif args.command == "release-contract":
        _print(release_contract())
    elif args.command == "build-release":
        _print(
            build_data_release(
                config,
                args.release_root,
                validation_size=args.validation_size,
                tokenizer_model_id=args.tokenizer_model,
                progress=print,
            )
        )
    elif args.command == "refresh-release-reports":
        _print(refresh_data_release_reports(args.release_root, progress=print))
    elif args.command == "tokenize-release":
        _print(
            tokenize_data_release(
                args.release_root,
                model_id=args.tokenizer_model,
                progress=print,
            )
        )
    elif args.command == "build-release-index":
        _print(build_data_release_index(args.release_root, progress=print))
    elif args.command == "preflight-release":
        report = validate_data_release(
            args.release_root,
            profile=args.profile,
            stage=args.stage,
            report_path=args.report,
        )
        _print(report)
        return 0 if report["status"] == "PASS" else 2
    elif args.command == "eda":
        report = analyze_dataset(config.data.data_dir, output / "eda_report.json")
        _print(report)
    elif args.command == "audit-sanitation":
        _print(audit_sanitation(config, progress=print))
    elif args.command == "review-boilerplate":
        _print(review_boilerplate_candidates(config))
    elif args.command == "freeze-corpus":
        _print(freeze_corpus(config))
    elif args.command == "build-corpus":
        _print(build_corpus(config, progress=print, reuse_audit=args.reuse_audit))
    elif args.command == "build-diagnostics":
        _print(build_diagnostic_labels(config))
    elif args.command == "build-index":
        _print(
            build_retrieval_indexes(
                config,
                build_dense=not args.skip_dense,
                rebuild_store=not args.reuse_bm25,
                resume_dense=args.resume_dense,
                progress=print,
            )
        )
    elif args.command == "benchmark-dense":
        _print(benchmark_dense_throughput(config, max_chunks=args.max_chunks))
    elif args.command == "build-graph":
        _print(build_citation_graph(output / "parents.jsonl", output / "citation_graph.json"))
    elif args.command == "retrieve":
        retriever = HybridRetriever(config)
        values = retriever.retrieve(args.question)
        _print(
            [
                {
                    "chunk_id": value.chunk_id,
                    "score": value.score,
                    "channels": value.channel_scores,
                    "text": retriever.get_chunk(value.chunk_id).retrieval_text,
                }
                for value in values
            ]
        )
    elif args.command == "predict":
        config.validate_submission()
        input_path = args.input or data_path(config, config.data.test_file)
        _print(
            LegalQAPipeline(config).predict_file(
                input_path,
                args.output,
                trace_path=args.trace,
                limit=args.limit,
                resume=args.resume,
                checkpoint_every=args.checkpoint_every,
                progress=print,
            )
        )
    elif args.command == "predict-neighbor":
        input_path = args.input or data_path(config, config.data.test_file)
        _print(
            predict_from_training_neighbors(
                data_path(config, config.data.train_file),
                input_path,
                args.output,
                dense_spec=config.models.dense,
                strategy=args.strategy,
                retrieval_top_k=args.top_k,
                dense_weight=args.dense_weight,
                trace_path=args.trace,
                limit=args.limit,
                progress=print,
            )
        )
    elif args.command == "evaluate-retrieval":
        _print(
            evaluate_diagnostic_retrieval(
                config,
                max_examples=args.max_examples,
                output_path=args.output,
            )
        )
    elif args.command == "evaluate-complementarity":
        _print(
            evaluate_retrieval_complementarity(
                config,
                limit=args.limit,
                max_examples=args.max_examples,
                output_path=args.output,
            )
        )
    elif args.command == "build-diagnostic-cache":
        _print(
            build_diagnostic_candidate_cache(
                config,
                max_examples=args.max_examples,
                output_path=args.output,
                progress=print,
            )
        )
    elif args.command == "evaluate-evidence":
        config.complexity.enabled = args.dynamic_k
        config.complexity.confidence_enabled = args.dynamic_k
        config.expansion.parent_enabled = args.parent
        config.expansion.graph_enabled = args.graph
        _print(
            evaluate_diagnostic_evidence(
                config,
                args.cache,
                output_path=args.output,
            )
        )
    elif args.command == "tune-dynamic-k":
        _print(tune_dynamic_k_rules(config, args.cache, output_path=args.output))
    elif args.command == "evaluate-cached-qa":
        config.complexity.enabled = args.dynamic_k
        config.complexity.confidence_enabled = args.dynamic_k
        config.expansion.parent_enabled = args.parent
        config.expansion.graph_enabled = args.graph
        _print(
            evaluate_cached_qa(
                config,
                args.cache,
                output_path=args.output,
                predictions_path=args.predictions,
                max_examples=args.max_examples,
                progress=print,
            )
        )
    elif args.command == "evaluate-qa":
        _print(evaluate_qa_files(args.predictions, args.references))
    elif args.command == "validate-predictions":
        input_path = args.input or data_path(config, config.data.test_file)
        report = validate_prediction_file(input_path, args.predictions)
        _print(report)
        return 0 if report["valid"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
