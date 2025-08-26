from typing import List

from spacy.tokens import Token, Span

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class PartOfSpeechFilter(CandidateFilter):
    """
    Filters the candidates based on their POS.
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[CandidateActor], _: FilterContext) -> \
            List[CandidateActor]:
        """
        Filters everything but nouns, numbers and 'you'.
        """
        return [c for c in candidates if
                c.token.pos_ in {"PROPN", "NOUN"} or c.token.text.lower() == "you"] or candidates
