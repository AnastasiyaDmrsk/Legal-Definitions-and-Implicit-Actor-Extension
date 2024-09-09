import dataclasses
from enum import Enum

from spacy.tokens import Token


class CandidateSource(Enum):
    DEFINITION = "DEFINITION"
    PREAMBLE = "PREAMBLE"
    BODY = "BODY"


@dataclasses.dataclass
class CandidateActor:
    token: Token
    source: CandidateSource
