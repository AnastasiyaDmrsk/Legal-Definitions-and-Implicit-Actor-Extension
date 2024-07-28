from typing import List

from spacy.tokens import Token, Span

from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class PartOfSpeechFilter(CandidateFilter):
    """
    Filters the candidates based on their POS.
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], context: Span) -> List[Token]:
        """
        Filters everything but nouns, numbers and 'you'.
        Numbers are never the correct target in the gold standard but theoretically, they should be able to be.
        """
        return [c for c in candidates if c.pos_ in {"PROPN", "NOUN", "NUM"} or c.text.lower() == "you"] or candidates
