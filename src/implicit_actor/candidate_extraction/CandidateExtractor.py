from abc import ABC, abstractmethod
from typing import List

from spacy.tokens import Doc, Token


class CandidateExtractor(ABC):
    """
    Extracts all possible candidates from the context
    """

    @abstractmethod
    def extract(self, context: Doc) -> List[Token]:
        """
        Takes in a context and extracts every possible subject candidate from it.
        """
        raise NotImplementedError()
