"""Define bounded Task 1 prediction records."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LegalIRPrediction:
    """Preserve a query ID and an ordered, unique list of document IDs."""

    query_id: str
    document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject blank identifiers, duplicates, and unbounded submissions."""
        if not self.query_id.strip() or any(char.isspace() for char in self.query_id):
            raise ValueError("query_id must be a non-empty identifier without whitespace.")
        if not self.document_ids or len(self.document_ids) > 100:
            raise ValueError("document_ids must contain between 1 and 100 items.")
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("document_ids must be unique and rank ordered.")
        if any(
            not item.strip() or any(char.isspace() for char in item) for item in self.document_ids
        ):
            raise ValueError("Each document ID must be non-empty and contain no whitespace.")
