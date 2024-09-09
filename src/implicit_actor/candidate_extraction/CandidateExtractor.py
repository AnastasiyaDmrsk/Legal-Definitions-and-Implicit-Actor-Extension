from abc import ABC, abstractmethod
from typing import List

from spacy.tokens import Doc, Token

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor


class CandidateExtractor(ABC):
    """
    Extracts all possible candidates from the context
    """

    @abstractmethod
    def extract(self, context: Doc) -> List[CandidateActor]:
        """
        Takes in a context and extracts every possible subject candidate from it.
        """
        raise NotImplementedError()
