from typing import List, Optional

from nltk.corpus import wordnet as wn
from spacy.tokens import Token, Span

from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class SynsetFilter(CandidateFilter):
    """
    Tries to judge the quality of candidates by comparing the similarity of the target with that of the
    base target
    """

    def __init__(self, whitelist: Optional[List[str]] = None):
        self._whitelist = whitelist or [
            "person", "group", "entity"
        ]

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], _: Span) -> List[Token]:
        """
        The idea is to detect changes in the meaning of the verb between implicit and explicit subject of the target.
        We do not want to change the meaning of the target by inserting the target.
        """

        filtered = list(filter(self._is_valid, candidates))
        return filtered or candidates

    def _is_valid(self, token: Token) -> bool:
        lexnames = [x.lexname() for x in wn.synsets(token.text, pos=wn.NOUN)]
        # print(token.text, lexnames)

        return any(
            f"noun.{w}" in lexnames for w in self._whitelist
        ) or (
                "noun.Tops" in lexnames and token.text.lower() in self._whitelist
        )
