"""Answer generators and the end-to-end prediction pipeline."""

from .generators import create_generator
from .pipeline import LegalQAPipeline

__all__ = ["LegalQAPipeline", "create_generator"]
