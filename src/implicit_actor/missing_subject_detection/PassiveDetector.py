from typing import List, Set

from spacy.tokens import Span

from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector
from src.implicit_actor.util import AUX_DEPS, has_explicit_subject, find_conj_head


class PassiveDetector(ImplicitSubjectDetector):
    """
    Detects passive verbs.
    """

    def __init__(self, blacklist: Set[str] = None):
        """
        :param blacklist: A list of predicates that should simply be ignored
        """
        if blacklist is None:
            blacklist = {"based", "referred"}
        self._blacklist = blacklist

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        return [
            ImplicitSubjectDetection(token=tok, type=ImplicitSubjectType.PASSIVE) for tok in span if
            tok.tag_ == "VBN"
            and not has_explicit_subject(find_conj_head(tok))
            and find_conj_head(tok).dep_ not in AUX_DEPS
            and not find_conj_head(tok).dep_ == "amod"
            and tok.text.lower() not in self._blacklist
        ]
