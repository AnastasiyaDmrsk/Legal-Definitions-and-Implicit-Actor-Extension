from typing import List

from spacy.tokens import Span

from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector


class NominalizedGerundWordlistDetector(ImplicitSubjectDetector):
    """
    Detects nouns that accept a subject, more specifically cases of gerundive nominalization.
    """

    def __init__(self):
        # wordlist ripped from patterns library
        with open("./data/external/en-verbs.txt", 'r', encoding="utf-8") as wf:
            self._verbs = {x.split(",")[0] for x in wf.readlines() if not x.startswith(";")}

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        """
        Detects nouns that look like gerunds, for example processing, packing, trading...
        Does not detect all nouns referring to an action.
        """
        return [
            ImplicitSubjectDetection(token=tok, type=ImplicitSubjectType.NOMINALIZED_VERB) for tok in span if
            tok.pos_ == "NOUN"
            and tok.text.endswith("ing")
            and tok.text.lower().removesuffix("ing") in self._verbs
            and not NominalizedGerundWordlistDetector._has_explicit_subject(tok)
        ]

    @staticmethod
    def _has_explicit_subject(tok):
        """
        Tests if the noun is already accompanied by a subject.
        """
        return any(True for t in tok.children if t.text.lower() == "by")
