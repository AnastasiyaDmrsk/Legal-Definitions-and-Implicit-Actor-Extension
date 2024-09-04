from abc import ABC, abstractmethod
from typing import List

from spacy.tokens import Token

from implicit_actor.candidate_filtering.FilterContext import FilterContext
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class CandidateFilter(ABC):
    """
    Filters the documents. If no meaningful filtering can be achieved, return all the candidates.
    """

    @abstractmethod
    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], context: FilterContext) -> List[Token]:
        """
        Filters the documents. If no meaningful filtering can be achieved, return all the candidates.

        :param target: The predicate or other token for which we are finding the subject.
        :param candidates: The potential candidates.
        :param context: The context available for the target.
        :param initial_candidates: The initial set of extracted candidates.
        """

        raise NotImplementedError()
