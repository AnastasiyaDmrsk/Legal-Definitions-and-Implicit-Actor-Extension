from abc import ABC, abstractmethod
from typing import List

from spacy.tokens import Token, Span

from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class CandidateFilter(ABC):
    """
    Filters the documents. If no meaningful filtering can be achieved, return all the candidates.
    """

    @abstractmethod
    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], context: Span) -> List[Token]:
        """
        Filters the documents. If no meaningful filtering can be achieved, return all the candidates.

        :param target: The predicate or other token for which we are finding the subject.
        :param candidates: The potential candidates.
        :param context: The context available for the target.
        """

        raise NotImplementedError()
