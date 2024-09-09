from typing import List

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from src.implicit_actor.util import search_for_head_block_nouns, OBJ_DEPS, SUBJ_DEPS


class DependentOfSameSentenceFilter(CandidateFilter):
    """
    Filters candidates that are already a dependency of the target,
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[CandidateActor], _: FilterContext) -> \
            List[CandidateActor]:
        """
        We ignore candidates that already seem to be a dependent of the target
        to avoid for example results like "The tree should be planted [by the tree]."
        """
        dependent_lemma = {candidate.token.lemma_ for candidate in candidates if
                           search_for_head_block_nouns(candidate.token) == target.token}
        # direct_obj_dependents = {t.lemma_ for t in target.token.children if t.dep_ in OBJ_DEPS}
        direct_dependents = {t.lemma_.lower() for t in target.token.children if t.dep_ in OBJ_DEPS | SUBJ_DEPS}

        return [c for c in candidates if
                c.token.lemma_.lower() not in dependent_lemma and
                c.token.lemma_ not in direct_dependents] or candidates
