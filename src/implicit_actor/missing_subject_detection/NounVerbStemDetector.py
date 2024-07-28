from typing import List

from nltk.stem.porter import PorterStemmer
from spacy.tokens import Span

from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, ImplicitSubjectType
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector


class NounVerbStemDetector(ImplicitSubjectDetector):
    """
    Detects nouns that have a verb as a stem.
    """

    def __init__(self):
        # wordlist ripped from patterns library
        with open("./data/external/en-verbs.txt", 'r', encoding="utf-8") as wf:
            self._verbs = {x.split(",")[0] for x in wf.readlines() if not x.startswith(";")}

        self._stemmer = PorterStemmer()

    def detect(self, span: Span) -> List[ImplicitSubjectDetection]:
        """
        Detects nouns that have a verb as stem
        """
        return [
            ImplicitSubjectDetection(token=tok, type=ImplicitSubjectType.NOMINALIZED_VERB) for tok in span if
            tok.pos_ == "NOUN" and self._stemmer.stem(tok.text) in self._verbs
            and not NounVerbStemDetector._has_explicit_subject(tok) and not tok.dep_ == "compound"
        ]

    @staticmethod
    def _has_explicit_subject(tok):
        """
        Tests if the noun is already accompanied by a subject.
        """
        return any(True for t in tok.children if t.text.lower() == "by")
