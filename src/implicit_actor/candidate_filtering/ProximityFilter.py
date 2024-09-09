from typing import List

from spacy.tokens import Token, Span

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ProximityFilter(CandidateFilter):
    """
    Ranks according to how close the candidate is to the target.
    """

    CATAPHORIC_PENALTY = 20

    DEPENDANT_PENALTY = 999

    def filter(self, target: ImplicitSubjectDetection, candidates: List[CandidateActor], _: FilterContext) -> \
            List[CandidateActor]:
        """
        Returns the physically closest candidate. At most one candidate is selected so this can be used
        at the end of a pipeline.
        """

        if not candidates:
            return []

        if len(candidates) == 1:
            # short circuit to avoid problems with the imperative filter inserting new
            # 'you' tokens (i.e., different docs.)
            return candidates

        assert target.token.doc == candidates[
            0].token.doc, "The ProximityRanker requires targets and candidates to be from the same doc."

        target_children = set(target.token.children) | {tok for c in target.token.children for tok in c.children
                                                        if
                                                        c.dep_ == "auxpass"}
        return [min(candidates,
                    key=lambda c:
                    abs(c.token.i - target.token.i) +
                    (self.CATAPHORIC_PENALTY if c.token.i > target.token.i else 0) +
                    (self.DEPENDANT_PENALTY if c.token in target_children else 0)
                    )]
