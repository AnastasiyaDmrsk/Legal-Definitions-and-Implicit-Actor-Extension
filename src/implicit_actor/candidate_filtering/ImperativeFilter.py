from typing import List

from spacy.lang.en import English

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectType, \
    ImplicitSubjectDetection


class ImperativeFilter(CandidateFilter):
    """
    Resolves the filtering if the target is an imperative by selecting `you`.
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[CandidateActor], _: FilterContext) -> \
            List[CandidateActor]:
        """
        Selects you if the target is an imperative.
        """
        if target.type == ImplicitSubjectType.IMPERATIVE:
            return [c for c in candidates if c.token.text.lower() == "you"][:1] or ImperativeFilter._you_token()

        return candidates

    @staticmethod
    def _you_token():
        nlp = English()
        return nlp("you")[:]
