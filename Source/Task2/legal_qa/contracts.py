"""Define independent Task 2 prediction records."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LegalQAPrediction:
    """Carry one non-empty answer for the Codabench submission."""

    query_id: str
    answer: str

    def __post_init__(self) -> None:
        """Require a valid identifier and answer."""
        if not self.query_id.strip() or any(char.isspace() for char in self.query_id):
            raise ValueError("query_id must be a non-empty identifier without whitespace.")
        if not self.answer.strip():
            raise ValueError("Task 2 prediction needs a non-empty answer.")

    def as_submission_item(self) -> dict[str, str]:
        """Serialize the answer without Task 1 evidence fields."""
        return {"answer": self.answer}
