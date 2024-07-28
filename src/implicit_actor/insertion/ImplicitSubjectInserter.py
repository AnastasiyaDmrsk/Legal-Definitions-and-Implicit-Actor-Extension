from abc import ABC, abstractmethod
from typing import List

from spacy.tokens import Span, Token

from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ImplicitSubjectInserter(ABC):
    """
    Pipeline step of inserting the subject into the target sentence.
    """

    @abstractmethod
    def insert(self, span: Span, targets: List[ImplicitSubjectDetection], subjects: List[Token]) -> str:
        """
        Insert into the target sentence.

        :param span: The span from which the returned string is derived.
        :param targets: The targets for subject insertion (predicates/nominalized verbs).
        :param subjects: The subjects to be inserted.
        """
        raise NotImplementedError()
