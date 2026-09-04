from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.io import read_jsonl, write_json
from ..core.metadata import canonical_document_number
from ..core.normalize import canonical_text


def _relation(text: str, has_amendment: bool) -> str:
    value = canonical_text(text)
    if "bãi bỏ" in value or "hết hiệu lực" in value:
        return "REPEALS"
    if "thay thế" in value:
        return "REPLACES"
    if has_amendment:
        return "AMENDS"
    return "REFERENCES"


def build_citation_graph(parents_path: str | Path, output_path: str | Path) -> dict[str, object]:
    by_doc_article: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_document_id_article: dict[tuple[str, str], list[str]] = defaultdict(list)
    document_parents: dict[str, list[str]] = defaultdict(list)
    for value in read_jsonl(parents_path):
        parent_id = str(value["id"])
        document_id = str(value["document_id"])
        article = value.get("article")
        document_parents[document_id].append(parent_id)
        if article:
            by_document_id_article[(document_id, str(article).casefold())].append(parent_id)
            number = canonical_document_number(value.get("document_number"))
            if number:
                by_doc_article[(number, str(article).casefold())].append(parent_id)

    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_keys: set[tuple[str, str, str]] = set()

    def add_edge(source: str, target: str, relation: str) -> None:
        key = (source, target, relation)
        if source == target or key in edge_keys:
            return
        edge_keys.add(key)
        adjacency[source].append({"target": target, "relation": relation})

    for document_id, parent_ids in document_parents.items():
        document_node = f"doc:{document_id}"
        for parent_id in parent_ids:
            add_edge(document_node, parent_id, "CONTAINS")
            add_edge(parent_id, document_node, "CONTAINED_BY")

    reverse_relations = {"AMENDS": "AMENDED_BY", "REPEALS": "REPEALED_BY", "REPLACES": "REPLACED_BY"}
    for value in read_jsonl(parents_path):
        parent_id = str(value["id"])
        document_id = str(value["document_id"])
        amendment_keys = set(value.get("amendments") or [])
        for reference in value.get("references") or []:
            number, article, _clause, _point = (reference.split("|") + ["", "", "", ""])[:4]
            if not article:
                continue
            targets = (
                by_doc_article.get((canonical_document_number(number) or "", article), [])
                if number
                else by_document_id_article.get((document_id, article), [])
            )
            relation = _relation(str(value.get("text") or ""), reference in amendment_keys)
            for target in targets:
                add_edge(parent_id, target, relation)
                if relation in reverse_relations:
                    add_edge(target, parent_id, reverse_relations[relation])

    graph = {
        "adjacency": dict(adjacency),
        "node_count": len(set(adjacency) | {edge["target"] for edges in adjacency.values() for edge in edges}),
        "edge_count": len(edge_keys),
        "relations": dict(
            sorted(
                {
                    relation: sum(1 for _source, _target, value in edge_keys if value == relation)
                    for relation in {value for _source, _target, value in edge_keys}
                }.items()
            )
        ),
        "source": "BTC corpus only",
    }
    write_json(output_path, graph)
    return {key: value for key, value in graph.items() if key != "adjacency"}


def graph_neighbors(
    graph: dict[str, Any],
    node_id: str,
    *,
    allowed_relations: set[str] | None = None,
    limit: int = 2,
) -> list[tuple[str, str]]:
    allowed = allowed_relations or {
        "REFERENCES",
        "AMENDS",
        "AMENDED_BY",
        "REPEALS",
        "REPEALED_BY",
        "REPLACES",
        "REPLACED_BY",
    }
    result = []
    for edge in graph.get("adjacency", {}).get(node_id, []):
        if edge["relation"] in allowed:
            result.append((edge["target"], edge["relation"]))
        if len(result) >= limit:
            break
    return result
