from typing import List

import spacy
from spacy.tokens import Token, Span

from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from src.implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


# TODO figure out how to make this work
class SingleTokenSimilarityFilter(CandidateFilter):
    """
    Tries to judge the quality of candidates by comparing the similarity of the target with that of the
    base target
    """

    def __init__(self, missing_subject_inserter: ImplicitSubjectInserter = None,
                 model="en_core_web_lg"):
        self._missing_subject_inserter = missing_subject_inserter or ImplicitSubjectInserterImpl()
        self._nlp = spacy.load(model)

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], _: Span) -> List[Token]:
        """
        The idea is to detect changes in the meaning of the verb between implicit and explicit subject of the target.
        We do not want to change the meaning of the target by inserting the target.
        """

        input_texts_with_insertion_spans = [
            (self._nlp(self._missing_subject_inserter.insert(target.token.sent, [target], [x])),
             self._missing_subject_inserter.last_insertion_spans()) for x in
            candidates
        ]

        baseline = self._nlp(
            target.token.sent.text
        )
        # TODO this can fail if the target is not found verbatim as a token in the baseline
        _, baseline_token = min(
            [(abs(i - target.token.i), x) for i, x in enumerate(baseline) if x.text == target.token.text],
            key=lambda x: x[0]
        )

        for itw, insp in input_texts_with_insertion_spans:
            s, e = insp[0]
            insertion_spans = itw.char_span(s, e, alignment_mode="contract")

            sims = (baseline_token.similarity(x) for x in insertion_spans)
            max_sim = max(
                s for s in sims if s < 1.0
            )

            print(baseline_token, insertion_spans, max_sim, sep="\t|\t")

        return candidates
