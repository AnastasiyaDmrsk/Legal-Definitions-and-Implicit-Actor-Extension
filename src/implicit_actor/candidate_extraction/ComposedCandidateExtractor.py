from typing import List

from spacy.tokens import Doc, Token

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_extraction.CandidateExtractor import CandidateExtractor


class ComposedCandidateExtractor(CandidateExtractor):
    def __init__(self, candidate_extractors: List[CandidateExtractor]):
        if not candidate_extractors:
            raise ValueError("At least one candidate extractor is required")

        self._candidate_extractors: List[CandidateExtractor] = candidate_extractors

    def extract(self, context: Doc) -> List[CandidateActor]:
        return list(
            c for ce in self._candidate_extractors for c in ce.extract(context=context)
        )
