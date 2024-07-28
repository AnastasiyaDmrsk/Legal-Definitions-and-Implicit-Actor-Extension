from typing import List, Optional

from spacy.tokens import Span, Token

from src.implicit_actor.insertion.DefaultInserter import DefaultInserter
from src.implicit_actor.insertion.GerundInserter import GerundInserter
from src.implicit_actor.insertion.ImperativeInserter import ImperativeInserter
from src.implicit_actor.insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from src.implicit_actor.insertion.SpecializedInserter import SpecializedInserter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ImplicitSubjectInserterImpl(ImplicitSubjectInserter):
    """
    Basic implementation of the subject insertion step.
    """

    def __init__(self, inserters: Optional[List[SpecializedInserter]] = None):
        self._sub_inserters: List[SpecializedInserter] = inserters or [
            GerundInserter(),
            ImperativeInserter(),
            DefaultInserter()
        ]

    def insert(self, span: Span, targets: List[ImplicitSubjectDetection], subjects: List[Token]) -> str:
        """
        Creates a new string from the doc with the subjects inserted at the targets location.
        """

        if len(subjects) != len(targets):
            raise ValueError("subjects and targets must have the same length")

        list_tokens = list(token.text_with_ws for token in span)

        for target, subj in zip(targets, subjects):
            inserter = next((x for x in self._sub_inserters if x.accepts(target.type)), None)
            if inserter is None:
                raise Exception(f"Could not find subinserter capable of handling {target.type}")
            inserter.insert(subj, list_tokens, target, span)

        resolved_text = "".join(list_tokens)
        return resolved_text
