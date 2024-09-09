import uuid

from spacy.tokens import Token, Span, MorphAnalysis

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.insertion.TokenList import TokenList
from src.implicit_actor.insertion.SpecializedInserter import SpecializedInserter, InsertionContext
from src.implicit_actor.insertion.pattern.inflect import conjugate
from src.implicit_actor.insertion.pattern.inflect_global import PRESENT, SINGULAR, PLURAL
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, \
    ImplicitSubjectType


class GerundInserter(SpecializedInserter):
    """
    Functional decomposition is a bane on my existence.
    """

    # TODO one should probably also look for temporal signal words like "when" and "during"
    PREPOSITIONS_TAKING_GERUND = {"of", "with", "for", "at", "about", "against", "up", "to"}

    def accepts(self, subject_type: ImplicitSubjectType):
        """
        Accepts gerunds.
        """
        return subject_type == ImplicitSubjectType.GERUND

    @staticmethod
    def _conjugate(verb: Token, morph: MorphAnalysis) -> str:
        map_num = {
            "Sing": SINGULAR,
            "Plur": PLURAL
        }

        pers = (morph.get("Person", ["3"]))[0]
        num = (morph.get("Number", ["Sing"]))[0]
        c = conjugate(verb.lemma_, PRESENT, int(pers), map_num[num])
        return c

    def insert(self, subj: CandidateActor, list_tokens: TokenList[str], target: ImplicitSubjectDetection, span: Span):
        """
        Feel free to guess.
        """

        insertion_point = target.token
        while pots := [x for x in insertion_point.lefts if
                       x.dep_ == "advmod" and x.pos_ == "ADV" and all(y.dep_ != "advcl" for y in x.children)]:
            # I am just assuming that the advmod tree is in-order
            insertion_point = min(pots, key=lambda x: x.i)

        cleaned_subj = SpecializedInserter._clean_subject(subj)

        if target.token.head.text in GerundInserter.PREPOSITIONS_TAKING_GERUND:
            target_replacement = target.token.text
        else:
            target_replacement = GerundInserter._conjugate(target.token, subj.token.morph)

        ctx = InsertionContext(
            insertion_id=str(uuid.uuid4())
        )

        if insertion_point == target.token:
            if insertion_point.is_sent_start:
                list_tokens[
                    insertion_point.i - span.start] = self.subject_mapper(
                    SpecializedInserter._upper_case_first(cleaned_subj), ctx) + self.target_mapper(
                    SpecializedInserter._lower_case_first(target_replacement), ctx) + insertion_point.whitespace_
            else:
                list_tokens[insertion_point.i - span.start] = self.subject_mapper(SpecializedInserter._lower_case_first(
                    cleaned_subj).rstrip(), ctx) + " " + self.target_mapper(target_replacement,
                                                                            ctx) + insertion_point.whitespace_
        else:
            # TODO double check this logic
            if insertion_point.is_sent_start:
                list_tokens[
                    insertion_point.i - span.start] = self.subject_mapper(
                    SpecializedInserter._upper_case_first(cleaned_subj),
                    ctx) + " " + SpecializedInserter._lower_case_first(
                    insertion_point.text) + insertion_point.whitespace_
            else:
                list_tokens[
                    insertion_point.i - span.start] = self.target_mapper(SpecializedInserter._lower_case_first(
                    target_replacement), ctx) + " " + insertion_point.text + insertion_point.whitespace_
            list_tokens[target.token.i - span.start] = self.target_mapper(target_replacement,
                                                                          ctx) + target.token.whitespace_
