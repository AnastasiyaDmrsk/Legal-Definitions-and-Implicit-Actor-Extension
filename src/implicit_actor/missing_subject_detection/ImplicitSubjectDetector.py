from abc import ABC, abstractmethod
from typing import List

from spacy.tokens import Doc, Span

from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ImplicitSubjectDetector(ABC):
    """
    Abstract base class for subject detection.
    """

    @abstractmethod
    def detect(self,  span: Span) -> List[ImplicitSubjectDetection]:
        """
        Detects verbs without an accompanying subject.
        """
        raise NotImplementedError()
