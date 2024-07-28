from dataclasses import dataclass
from enum import Enum

from spacy.tokens import Token


class ImplicitSubjectType(Enum):
    """
    Enum for holding different type of detected implicit subject types to be used for strategy selection in downstream
    tasks.
    """
    PASSIVE = 1
    GERUND = 2
    IMPERATIVE = 3
    NOMINALIZED_VERB = 4


@dataclass
class ImplicitSubjectDetection:
    """
    Class for holding detected implicit subjects (i.e., the lack of a subject, not the implicit subject itself).
    """

    token: Token
    type: ImplicitSubjectType
