from abc import ABC, abstractmethod
from typing import List

from spacy.tokens import Token

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class CandidateFilter(ABC):
    """
    Filters the documents. If no meaningful filtering can be achieved, return all the candidates.
    """

    @abstractmethod
    def filter(self, target: ImplicitSubjectDetection, candidates: List[CandidateActor], context: FilterContext) -> \
            List[CandidateActor]:
        """
        Filters the documents. If no meaningful filtering can be achieved, return all the candidates.

        :param target: The predicate or other token for which we are finding the subject.
        :param candidates: The potential candidates.
        :param context: The context available for the target.
        """

        raise NotImplementedError()
