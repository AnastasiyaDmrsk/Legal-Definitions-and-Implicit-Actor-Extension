from typing import List

from spacy.matcher import Matcher
from spacy.tokens import Doc, Token

from implicit_actor.candidate_extraction.CandidateExtractor import CandidateExtractor


class DefinitionCandidateExtractor(CandidateExtractor):
    def __init__(self):
        self._pattern = [
            {"TEXT": "‘"},
            {"IS_ALPHA": True, "OP": "+"},
            {"TEXT": "’"},
        ]

    def extract(self, context: Doc) -> List[Token]:
        matcher = Matcher(context.vocab)
        matcher.add("DEFINITION", [self._pattern], greedy="FIRST")

        def _matches():
            for match_id, start, end in matcher(context):
                span = context[start:end]
                yield from (t for t in span if t.dep_ == "nsubj" and t.pos_ == "NOUN")

        return list(_matches())
