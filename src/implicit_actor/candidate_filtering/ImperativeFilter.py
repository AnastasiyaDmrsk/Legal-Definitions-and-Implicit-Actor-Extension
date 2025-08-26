from typing import List

from spacy.lang.en import English

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor, CandidateSource
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectType, \
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
            return [c for c in candidates if c.token.text.lower() == "you"][:1] or [ImperativeFilter._you_token()]

        return candidates

    @staticmethod
    def _you_token() -> CandidateActor:
        nlp = English()
        return CandidateActor(token=nlp("you")[0], source=CandidateSource.ARTIFICIAL)
