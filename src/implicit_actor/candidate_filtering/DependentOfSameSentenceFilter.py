from typing import List

from spacy.tokens import Token, Span

from implicit_actor.candidate_filtering.FilterContext import FilterContext
from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from src.implicit_actor.util import search_for_head_block_nouns, OBJ_DEPS


class DependentOfSameSentenceFilter(CandidateFilter):
    """
    Filters candidates that are already a dependency of the target,
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], _: FilterContext) -> List[Token]:
        """
        Filters candidates that are part of the same sentence in an incompatible way, i.e., dependent of the
        same predicate or straight up a dependent.
        """
        dependent_lemma = {tok.lemma_ for tok in candidates if search_for_head_block_nouns(tok) == target.token}
        direct_obj_dependents = {t.lemma_ for t in target.token.children if t.dep_ in OBJ_DEPS}
        return [c for c in candidates if
                c.lemma_ not in dependent_lemma and
                c.lemma_ not in direct_obj_dependents] or candidates
