from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..core.config import PipelineConfig, artifact_dir
from ..core.schema import Candidate, Evidence, LegalChunk, ParentSection
from .complexity import (
    analyze_query_complexity,
    analyze_retrieval_confidence,
    choose_dynamic_top_k,
)
from .graph import graph_neighbors
from .store import CorpusStore


def _anchored_parent_text(parent: ParentSection, child: LegalChunk, max_chars: int) -> str:
    if len(parent.text) <= max_chars:
        return parent.text
    child_body = child.retrieval_text.split("\n", 1)[-1]
    anchor = child_body[: min(300, len(child_body))]
    position = parent.text.find(anchor)
    if position < 0:
        return parent.text[:max_chars]
    start = max(0, position - max_chars // 4)
    end = min(len(parent.text), start + max_chars)
    start = max(0, end - max_chars)
    excerpt = parent.text[start:end]
    prefix = f"{parent.heading}\n" if start > 0 else ""
    return f"{prefix}{excerpt}"[:max_chars]


class EvidenceAssembler:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        output = artifact_dir(config)
        self.store = CorpusStore(output / "corpus.sqlite")
        graph_path = output / "citation_graph.json"
        self.graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {"adjacency": {}}

    def assemble(self, question: str, candidates: list[Candidate]) -> tuple[list[Evidence], dict[str, Any]]:
        complexity = analyze_query_complexity(question, self.config.complexity)
        confidence = analyze_retrieval_confidence([candidate.score for candidate in candidates])
        top_k = choose_dynamic_top_k(complexity, confidence, self.config.complexity)
        selected = candidates[:top_k]
        evidence_pool: list[Evidence] = []
        seen_parents: set[str] = set()
        for candidate in selected:
            chunk = self.store.get_chunk(candidate.chunk_id)
            if self.config.expansion.parent_enabled and self.store.has_parent(chunk.parent_id):
                parent = self.store.get_parent(chunk.parent_id)
                if parent.id not in seen_parents:
                    seen_parents.add(parent.id)
                    evidence_pool.append(
                        Evidence(
                            id=parent.id,
                            text=_anchored_parent_text(parent, chunk, self.config.evidence.parent_max_chars),
                            score=candidate.score,
                            document_id=parent.document_id,
                            source_name=parent.source_name,
                            provenance=("retrieval", "parent_expansion"),
                            chunk_ids=(chunk.id,),
                        )
                    )
            else:
                evidence_pool.append(
                    Evidence(
                        id=chunk.id,
                        text=chunk.retrieval_text,
                        score=candidate.score,
                        document_id=chunk.document_id,
                        source_name=chunk.source_name,
                        provenance=("retrieval",),
                        chunk_ids=(chunk.id,),
                    )
                )

        if self.config.expansion.graph_enabled:
            seeds = list(evidence_pool)
            for seed in seeds:
                parent_id = seed.id if self.store.has_parent(seed.id) else self.store.get_chunk(seed.id).parent_id
                for target, relation in graph_neighbors(
                    self.graph,
                    parent_id,
                    limit=self.config.expansion.graph_max_per_seed,
                ):
                    if target in seen_parents or not self.store.has_parent(target):
                        continue
                    parent = self.store.get_parent(target)
                    seen_parents.add(target)
                    evidence_pool.append(
                        Evidence(
                            id=target,
                            text=parent.text[: self.config.evidence.parent_max_chars],
                            score=seed.score * self.config.expansion.graph_decay,
                            document_id=parent.document_id,
                            source_name=parent.source_name,
                            provenance=("graph_expansion", relation),
                        )
                    )

        packed = self._pack(evidence_pool)
        decision = {
            "top_k": top_k,
            "query_complexity": {"score": complexity.score, "features": complexity.features, "base_top_k": complexity.top_k},
            "retrieval_confidence": {
                "concentration": confidence.concentration,
                "ambiguity": confidence.ambiguity,
                "top_gap": confidence.top_gap,
                "tail_mass": confidence.tail_mass,
                "elbow_position": confidence.elbow_position,
            },
            "retrieved": len(candidates),
            "selected_before_expansion": len(selected),
            "packed_evidence": len(packed),
        }
        return packed, decision

    def _pack(self, pool: list[Evidence]) -> list[Evidence]:
        if not pool:
            return []
        pool.sort(key=lambda value: (-value.score, value.id))
        best_score = pool[0].score
        counts: Counter[str] = Counter()
        used = 0
        selected: list[Evidence] = []
        for evidence in pool:
            if best_score > 0 and evidence.score < best_score * self.config.evidence.min_score_ratio:
                continue
            if counts[evidence.document_id] >= self.config.evidence.max_per_document:
                continue
            remaining = self.config.evidence.max_chars - used
            if remaining <= 0:
                break
            text = evidence.text[:remaining]
            if not text.strip():
                continue
            evidence.text = text
            selected.append(evidence)
            counts[evidence.document_id] += 1
            used += len(text)
        return selected
