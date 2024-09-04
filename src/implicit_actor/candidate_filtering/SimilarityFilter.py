from typing import List

import spacy
from spacy.tokens import Token, Span

from implicit_actor.candidate_filtering.FilterContext import FilterContext
from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from src.implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class SimilarityFilter(CandidateFilter):
    """
    Tries to judge the quality of candidates by comparing their similarity to that of the context.
    """

    def __init__(self, missing_subject_inserter: ImplicitSubjectInserter = None, top_k=10, use_context=False,
                 model="en_use_md"):
        self._missing_subject_inserter = missing_subject_inserter or ImplicitSubjectInserterImpl()
        self._nlp = spacy.load(model)
        self._top_k = top_k
        self._use_context = use_context

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], ctx: FilterContext) -> List[Token]:
        ctx_str = str(ctx.context) + " " if self._use_context else ""

        input_texts = [
            ctx_str + self._missing_subject_inserter.insert(target.token.sent, [target], [x]) for x in
            candidates
        ]

        docs = list(map(self._nlp, input_texts))
        benchmark = self._nlp(ctx_str + str(target.token.sent))

        sims = [benchmark.similarity(d) for d in docs]
        ret = list(zip(sims, candidates))
        ret.sort(key=lambda x: x[0], reverse=True)
        ret = list(map(lambda x: x[1], ret))
        return ret[:self._top_k] or candidates
