from collections import defaultdict
from typing import List, Dict, Optional, Set, Iterable

from nltk import PorterStemmer
from nltk.corpus import wordnet as wn
from spacy.tokens import Token, Span

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor, CandidateSource
from implicit_actor.candidate_extraction.PreambleExtractor import PreambleExtractor
from implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from implicit_actor.missing_subject_detection.NounVerbStemDetector import NounVerbStemDetector


class VerbLinkFilter(CandidateFilter):

    def __init__(self, add_synonyms=False, add_preamble_verbs=True):
        """
        TODO
        Test TODOs: Preambel mit mehr filtern
        When extracting verbs from the preamble, just stemming for definitions and only verbs for rest
        """

        self._stem_detector = NounVerbStemDetector()
        self._stemmer = PorterStemmer()
        self._add_preamble_verbs = add_preamble_verbs
        self._add_synonyms = add_synonyms
        self._candidate_definitions_with_actions: Optional[Dict[str, Set[str]]] = None
        self._preamble_detector = PreambleExtractor()

    def filter(self, target: ImplicitSubjectDetection, candidates: List[CandidateActor], ctx: FilterContext) -> \
            List[CandidateActor]:
        # if self._candidate_definitions_with_actions is None:
        self._candidate_definitions_with_actions = self._get_candidate_definitions_with_actions(candidates,
                                                                                                ctx)

        filtered = list(
            filter(lambda x: self._pred(target.token, x), candidates)
        )

        return filtered or candidates

    def _pred(self, target: Token, c: CandidateActor):
        x = self._candidate_definitions_with_actions.get(c.token.lemma_, set())
        return target.lemma_ in x or self._stemmer.stem(
            target.lemma_) in x

    def _extract_additional_stems(self, candidate: CandidateActor) -> Iterable[Token]:
        if candidate.source != CandidateSource.PREAMBLE:
            return (x.token for x in self._stem_detector.detect(self._definition(candidate)))

        return (x for x in candidate.token.sent if x.pos_ == "VERB")

    def _get_candidate_definitions_with_actions(self, candidates: List[CandidateActor], ctx: FilterContext):
        # Yes, we are doing a lot of duplicate and unnecessary calculation here :D
        # This holds data of the form: verb -> verbs inside of verb definition
        raw_definition_stems = [
            (self._stemmer.stem(v.token.text), [
                self._stemmer.stem(x.lemma_) for x in self._extract_additional_stems(v)
            ]) for v in ctx.initial_candidates
        ]

        definition_stems = defaultdict(list)
        for k, v in raw_definition_stems:
            definition_stems[k].extend(v)

        raw_verb_definitions_with_actions = [
            (candidate, [v.lemma_ for v in self._definition(candidate) if v.dep_ in {"relcl", "acl"}]) for candidate
            in candidates
        ]

        if self._add_preamble_verbs:
            preamble_candidates = self._preamble_detector.extract(ctx.context.doc)

            for candidate, verb_list in raw_verb_definitions_with_actions:
                for pc in preamble_candidates:
                    if pc.token.text.lower() == candidate.token.text.lower():
                        verbs_from_preamble = list(v.lemma_ for v in self._definition(pc) if v.dep_ in {"relcl", "acl"})
                        verb_list.extend(verbs_from_preamble)

        # maps from candidates to all the associated verbs
        candidate_definitions_with_actions = defaultdict(set)
        for candidate, candidate_verbs in raw_verb_definitions_with_actions:
            candidate_definitions_with_actions[candidate.token.lemma_].update(candidate_verbs)

        for v in candidate_definitions_with_actions.values():
            additional_verbs = {
                y for x in v for y in definition_stems[x]
            }
            v.update(additional_verbs)

        if self._add_synonyms:
            for c, verbs in candidate_definitions_with_actions.items():
                additional_verbs = set()
                for v in verbs:
                    # not using wn.synonyms directly, as we only want verbs
                    additional_verbs |= {y.name() for x in wn.synsets(v, pos=wn.VERB) for y in x.lemmas()}
                verbs.update(additional_verbs)

        return candidate_definitions_with_actions

    def _definition(self, token: CandidateActor) -> Span:
        offset = token.token.i - token.token.sent.start

        l, r = offset, offset

        while l > 0 and token.token.sent[l].pos_ != "X":
            l -= 1

        while r < len(token.token.sent) and token.token.sent[r].pos_ != "X":
            r += 1

        return token.token.sent[l:r]
