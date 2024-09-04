from collections import defaultdict
from itertools import chain
from typing import List

from spacy.tokens import Token, Span

from implicit_actor.candidate_filtering.FilterContext import FilterContext
from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class CandidateTextOccurrenceFilter(CandidateFilter):
    """
    Returns all sets of candidates that have maximum occurrence of their text, e.g.,

    [you, you, you, car, car, car, duck, penguin] -> [you, you, you, car, car, car]
    """

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], _: FilterContext) -> List[Token]:
        """
        Filter the candidates based purely on how often they occur.
        """

        if not candidates:
            return candidates

        grouping = defaultdict(list)

        for c in candidates:
            grouping[c.text.lower()].append(c)

        max_length = max(len(g) for g in grouping.values())

        return list(chain.from_iterable(x for x in grouping.values() if len(x) == max_length)) or candidates
