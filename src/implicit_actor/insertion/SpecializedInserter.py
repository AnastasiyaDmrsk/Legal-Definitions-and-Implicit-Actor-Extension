import dataclasses
from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from spacy.tokens import Token, Span

from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, \
    ImplicitSubjectType
from src.implicit_actor.util import get_noun_chunk


@dataclasses.dataclass()
class InsertionContext:
    """
    Represents passed conntext during insertion
    """
    insertion_id: str


class SpecializedInserter(ABC):
    """
    Inserts a subject into a sentence.
    """

    def __init__(self, subject_mapper: Optional[Callable[[str, InsertionContext], str]] = None,
                 target_mapper: Optional[Callable[[str, InsertionContext], str]] = None):
        self.subject_mapper = subject_mapper or (lambda x: x)
        self.target_mapper = target_mapper or (lambda x: x)

    @abstractmethod
    def accepts(self, subject_type: ImplicitSubjectType):
        """
        Checks if a subject type is accepted by this.
        """
        raise NotImplementedError()

    @abstractmethod
    def insert(self, subj: Token, list_tokens: List[str], target: ImplicitSubjectDetection, span: Span):
        """
        Inserts the subject into the list_tokens list.

        Note: This method is expected to modify list_tokens.
        """
        raise NotImplementedError()

    @staticmethod
    def _clean_subject(subj: Token) -> str:
        chunky_boi = get_noun_chunk(subj)
        return SpecializedInserter._lower_case_first(str(chunky_boi.text_with_ws)) \
            if not chunky_boi[0].pos_ == "PROPN" else str(chunky_boi)

    @staticmethod
    def _lower_case_first(s: str) -> str:
        return s[0].lower() + s[1:] if s else s

    @staticmethod
    def _upper_case_first(s: str) -> str:
        return s[0].upper() + s[1:] if s else s
