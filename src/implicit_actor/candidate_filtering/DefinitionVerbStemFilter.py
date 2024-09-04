from collections import defaultdict
from typing import List, Dict, Optional, Set

from nltk import PorterStemmer
from nltk.corpus import wordnet as wn
from spacy.tokens import Token, Span

from implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from implicit_actor.missing_subject_detection.NounVerbStemDetector import NounVerbStemDetector


class DefinitionVerbStemFilter(CandidateFilter):

    def __init__(self, add_synonyms=False):
        self._stem_detector = NounVerbStemDetector()
        self._stemmer = PorterStemmer()
        self._add_synonyms = add_synonyms
        self._candidate_definitions_with_actions: Optional[Dict[str, Set[str]]] = None

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], ctx: FilterContext) -> List[Token]:
        # if self._candidate_definitions_with_actions is None:
        self._candidate_definitions_with_actions = self._get_candidate_definitions_with_actions(candidates,
                                                                                                ctx.initial_candidates)

        filtered = list(
            filter(lambda x: self._pred(target.token, x), candidates)
        )

        return filtered or candidates

    def _pred(self, target: Token, c: Token):
        x = self._candidate_definitions_with_actions.get(c.lemma_, set())
        return target.lemma_ in x or self._stemmer.stem(
            target.lemma_) in x

    def _get_candidate_definitions_with_actions(self, candidates: List[Token], initial_candidates: List[Token]):
        # Yes, we are doing a lot of duplicate and unnecessary calculation here :D
        # This holds data of the form: verb -> verbs inside of verb definition
        raw_definition_stems = [
            (self._stemmer.stem(v.text), [
                self._stemmer.stem(x.token.lemma_) for x in self._stem_detector.detect(self._definition(v))
            ]) for v in initial_candidates
        ]

        definition_stems = defaultdict(list)
        for k, v in raw_definition_stems:
            definition_stems[k].extend(v)

        raw_verb_definitions_with_actions = [
            (candidate, [v.lemma_ for v in self._definition(candidate) if v.dep_ in {"relcl", "acl"}]) for candidate
            in candidates
        ]

        # maps from candidates to all the associated verbs
        candidate_definitions_with_actions = defaultdict(set)
        for candidate, candidate_verbs in raw_verb_definitions_with_actions:
            candidate_definitions_with_actions[candidate.lemma_].update(candidate_verbs)

        for k, v in candidate_definitions_with_actions.items():
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

    def _definition(self, token: Token) -> Span:
        offset = token.i - token.sent.start

        l, r = offset, offset

        while l > 0 and token.sent[l].pos_ != "X":
            l -= 1

        while r < len(token.sent) and token.sent[r].pos_ != "X":
            r += 1

        return token.sent[l:r]
