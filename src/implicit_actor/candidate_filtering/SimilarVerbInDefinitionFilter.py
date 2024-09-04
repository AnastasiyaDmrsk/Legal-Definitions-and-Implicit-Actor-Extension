from typing import List

from spacy.tokens import Token, Span

from implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class SimilarVerbInDefinitionFilter(CandidateFilter):
    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], _: FilterContext) -> List[Token]:
        res = [
            c for c in candidates
            if c.lemma_ in self._lemmata_in_definition(c)
        ]

        return res or candidates

    def _lemmata_in_definition(self, c: Token) -> List[str]:
        return [tok.lemma_.removesuffix("ing") for tok in self._definition(c) if tok.pos_ == "VERB"]

    def _definition(self, token: Token):

        offset = token.i - token.sent.start

        l, r = offset, offset

        while l > 0 and token.sent[l].pos_ != "X":
            l -= 1

        while r < len(token.sent) and token.sent[r].pos_ != "X":
            r += 1

        return token.sent[l:r]
