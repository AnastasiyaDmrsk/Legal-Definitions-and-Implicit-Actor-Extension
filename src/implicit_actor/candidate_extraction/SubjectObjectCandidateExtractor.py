from typing import List

from spacy.tokens import Doc

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor, CandidateSource
from implicit_actor.candidate_extraction.CandidateExtractor import CandidateExtractor
from implicit_actor.util import OBJ_DEPS, SUBJ_DEPS


class SubjectObjectCandidateExtractor(CandidateExtractor):

    def extract(self, context: Doc) -> List[CandidateActor]:
        return [CandidateActor(token=tok, source=CandidateSource.BODY) for tok in context if
                (tok.dep_ in SUBJ_DEPS or tok.dep_ in OBJ_DEPS)]
