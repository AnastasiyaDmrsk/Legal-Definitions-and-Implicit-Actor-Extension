import dataclasses
from typing import List

from spacy.tokens import Span, Token


@dataclasses.dataclass
class FilterContext:
    """Additional context passed to the Candidate Filters"""
    context: Span
    initial_candidates: List[Token]
