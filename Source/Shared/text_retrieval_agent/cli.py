"""Expose search, answer, and evaluation commands."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from text_retrieval_agent.agent import EvidenceAgent
from text_retrieval_agent.contracts import RetrievalQuery
from text_retrieval_agent.evaluation import evaluate
from text_retrieval_agent.pipeline import RetrievalPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egta")
    parser.add_argument("--corpus", type=Path, default=Path("Data/Shared/fixtures/documents.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/Shared/smoke.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "ask"):
        command = commands.add_parser(name)
        command.add_argument("--query-id", default="cli-query")
        command.add_argument("--text", required=True)
        command.add_argument("--scope", action="append", default=["public"])
        command.add_argument("--top-k", type=int, default=5)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument(
        "--queries", type=Path, default=Path("Data/Shared/fixtures/queries.jsonl")
    )
    evaluation.add_argument("--qrels", type=Path, default=Path("Data/Shared/fixtures/qrels.jsonl"))
    evaluation.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and emit machine-readable JSON."""
    arguments = _parser().parse_args(argv)
    pipeline = RetrievalPipeline(arguments.corpus, arguments.config)
    if arguments.command == "evaluate":
        payload = evaluate(
            pipeline,
            arguments.queries,
            arguments.qrels,
            top_k=arguments.top_k,
        )
    else:
        query = RetrievalQuery(
            query_id=arguments.query_id,
            text=arguments.text,
            allowed_scopes=tuple(arguments.scope),
            top_k=arguments.top_k,
        )
        payload = (
            asdict(pipeline.search(query))
            if arguments.command == "search"
            else asdict(EvidenceAgent(pipeline).ask(query))
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
