from typing import List

from spacy.tokens import Span

from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, \
    ImplicitSubjectType
from implicit_actor.missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector
from implicit_actor.util import has_explicit_subject


class GerundDetector(ImplicitSubjectDetector):
    """
    Detects gerunds which are missing a 'subject', i.e., a by phrase.
    """

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        return [
            ImplicitSubjectDetection(token=tok, type=ImplicitSubjectType.GERUND) for tok in span if
            tok.tag_ == "VBG"
            and not has_explicit_subject(tok)
            and tok.dep_ not in {"amod", "acl", "pcomp", "prep"}
        ]
