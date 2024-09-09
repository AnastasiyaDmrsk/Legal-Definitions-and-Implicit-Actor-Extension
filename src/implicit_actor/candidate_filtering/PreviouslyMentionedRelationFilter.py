from typing import List

from nltk.stem import PorterStemmer
from spacy.tokens import Token, Span

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, \
    ImplicitSubjectType
from src.implicit_actor.util import ACTIVE_VOICE_SUBJ_DEPS, search_for_head


class PreviouslyMentionedRelationFilter(CandidateFilter):
    """
    Searches for occurrences of candidates as the subject of a verb with the same stem.

    Example:
    inspected_text = "After being eaten, the cat is sad."
    context = "The dog eats the cat."
    ->
    (ImplicitSubjectDetection(token=being, type=<ImplicitSubjectType.GERUND: 2>), cat)
    (ImplicitSubjectDetection(token=eaten, type=<ImplicitSubjectType.PASSIVE: 1>), dog)

    # The following example should also pass when using this filter (+PartOfSpeechFilter):
    pipeline.apply("After being eaten, the cat is sad.", "The dog eats the cat.")
    pipeline.apply("After being eaten, the cat is sad.", "The cat is eaten by the dog.")
    pipeline.apply("After the cat is eaten, the cat is sad.", "The cat is eaten by the dog.")
    pipeline.apply("The eating makes the cat sad.", "The dog eats the cat.")
    """

    def __init__(self):
        self._stemmer = PorterStemmer()

    def filter(self, target: ImplicitSubjectDetection, candidates: List[CandidateActor], _: FilterContext) -> \
            List[CandidateActor]:
        """
        Retain only candidates that are the subject of predicates with the same stem as the step of the target token.
        """
        if target.type == ImplicitSubjectType.IMPERATIVE:
            return candidates

        if target.token.dep_ == "auxpass":
            check_dep = PreviouslyMentionedRelationFilter._check_patient_dep
            target_token = target.token.head
        else:
            check_dep = PreviouslyMentionedRelationFilter._check_actor_dep
            target_token = target.token

        target_stem = self._stemmer.stem(target_token.lemma_)
        refined_candidates = {c.token.lemma_.lower() for c in candidates if
                              c.token.head.dep_ != "auxpass" and
                              self._stemmer.stem(search_for_head(c.token).lemma_) == target_stem and
                              check_dep(c.token)}

        return [c for c in candidates if c.token.lemma_.lower() in refined_candidates] or candidates

    @staticmethod
    def _check_patient_dep(tok: Token):
        return tok.dep_ in {"attr", "dobj", "dative", "oprd", "nsubjpass"}

    @staticmethod
    def _check_actor_dep(tok: Token):
        return (tok.dep_ in ACTIVE_VOICE_SUBJ_DEPS) or (tok.dep_ == "pobj" and tok.head.dep_ == "agent")
