from typing import List, Optional

from nltk.corpus import wordnet as wn
from spacy.tokens import Token

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from implicit_actor.util import OBJ_DEPS
from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


def _get_definition_objects_from_head(token: Token) -> List[Token]:
    h = token.head
    if h is None or h.lemma_ != "mean":
        return []

    return [c for c in h.children if c.dep_ in OBJ_DEPS]


class SynsetFilter(CandidateFilter):
    """
    We check if the candidate (or its explanation) is a person or entity.
    """

    def __init__(self, whitelist: Optional[List[str]] = None):
        self._whitelist = whitelist or [
            "person", "entity"
        ]

    def filter(self, target: ImplicitSubjectDetection, candidates: List[CandidateActor], _: FilterContext) -> \
            List[CandidateActor]:
        """
        Filter candidates based on their noun group
        """

        filtered = list(
            filter(lambda x: self._is_valid(x.token) or any(self._is_valid(y) for y in _get_definition_objects_from_head(x.token)),
                   candidates))
        return filtered or candidates

    def _is_valid(self, token: Token) -> bool:
        lexnames = [x.lexname() for x in wn.synsets(token.lemma_, pos=wn.NOUN)]

        return any(
            f"noun.{w}" in lexnames for w in self._whitelist
        ) or (
                "noun.Tops" in lexnames and token.text.lower() in self._whitelist
        )
