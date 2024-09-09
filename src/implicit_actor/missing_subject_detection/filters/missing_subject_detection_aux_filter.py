from typing import List

from spacy.tokens import Token

from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from implicit_actor.missing_subject_detection.filters.missing_subject_detection_filter import \
    MissingSubjectDetectionFilter
from nltk.corpus import wordnet as wn


class MissingSubjectDetectionAuxFilter(MissingSubjectDetectionFilter):
    def __init__(self):
        self._auxiliary_whitelist = {"shall", "should", "must", "may"}

    def filter(self, detections: List[ImplicitSubjectDetection]) -> List[ImplicitSubjectDetection]:
        return [
            d for d in detections if not self._pred(d)
        ]

    def _pred(self, detection: ImplicitSubjectDetection) -> bool:
        # If the sentence already contains a basic action with subject, we can ignore the sentence
        return any(
            t.dep_ == "aux" and t.lemma_ in self._auxiliary_whitelist and self._has_subj_noun(t.head) for t in
            detection.token.sent
        )

    def _has_subj_noun(self, token: Token):
        return any(
            c.dep_ == "nsubj" and self._is_entity_or_person(c) for c in token.children
        )

    def _is_entity_or_person(self, token: Token) -> bool:
        lexnames = [x.lexname() for x in wn.synsets(token.lemma_, pos=wn.NOUN)]
        whitelist = [
            "person", "entity"
        ]
        return any(
            f"noun.{w}" in lexnames for w in whitelist
        ) or (
                "noun.Tops" in lexnames and token.text.lower() in whitelist
        )
