from typing import List

from spacy.tokens import Doc, Token

from src.implicit_actor.candidate_extraction.CandidateExtractor import CandidateExtractor
from src.implicit_actor.util import OBJ_DEPS, SUBJ_DEPS


class CandidateExtractorImpl(CandidateExtractor):

    def extract(self, context: Doc) -> List[Token]:
        return [tok for tok in context if (tok.dep_ in SUBJ_DEPS or tok.dep_ in OBJ_DEPS)]
