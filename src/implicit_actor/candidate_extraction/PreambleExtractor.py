import re
from typing import List

from spacy.tokens import Doc

from implicit_actor.candidate_extraction.CandidateActor import CandidateSource, CandidateActor
from implicit_actor.candidate_extraction.CandidateExtractor import CandidateExtractor
from implicit_actor.util import SUBJ_DEPS, OBJ_DEPS


class PreambleExtractor(CandidateExtractor):
    def __init__(self):
        self._divide = r"HAVE ADOPTED THIS REGULATION:"

    def extract(self, context: Doc) -> List[CandidateActor]:
        m = re.search(self._divide, context.text)

        if not m:
            return []

        start, _ = m.span()
        inspected_span = context.char_span(0, start, alignment_mode="expand")
        # return [CandidateActor(token=tok, source=CandidateSource.PREAMBLE) for tok in inspected_span if
        #         ((tok.dep_ in SUBJ_DEPS or tok.dep_ in OBJ_DEPS) and tok.pos_ in {"NOUN", "PROPN"})]
        return [CandidateActor(token=tok, source=CandidateSource.PREAMBLE) for tok in inspected_span if
                ((tok.dep_ in SUBJ_DEPS) and tok.pos_ in {"NOUN", "PROPN"})]
