"""Define evidence-bound Task 2 prediction records."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LegalQAPrediction:
    """Carry an answer and the ordered retrieved chunk IDs supporting it."""

    query_id: str
    answer: str
    evidence_chunk_ids: tuple[str, ...]
    abstained: bool = False

    def __post_init__(self) -> None:
        """Require evidence for answers and no evidence for abstentions."""
        if not self.query_id.strip() or any(char.isspace() for char in self.query_id):
            raise ValueError("query_id must be a non-empty identifier without whitespace.")
        if self.abstained:
            if self.evidence_chunk_ids:
                raise ValueError("An abstention cannot cite evidence chunks.")
            return
        if not self.answer.strip():
            raise ValueError("A non-abstaining prediction needs an answer.")
        if not self.evidence_chunk_ids or len(self.evidence_chunk_ids) > 5:
            raise ValueError("Answers need between 1 and 5 evidence chunks.")
        if len(set(self.evidence_chunk_ids)) != len(self.evidence_chunk_ids):
            raise ValueError("evidence_chunk_ids must be unique and rank ordered.")
