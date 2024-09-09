import dataclasses
from typing import List

from spacy.tokens import Span, Token

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor


@dataclasses.dataclass
class FilterContext:
    """Additional context passed to the Candidate Filters"""
    context: Span
    initial_candidates: List[CandidateActor]
