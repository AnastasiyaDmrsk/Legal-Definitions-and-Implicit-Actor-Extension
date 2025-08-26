import uuid

from spacy.tokens import Span

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.insertion.TokenList import TokenList
from implicit_actor.insertion.SpecializedInserter import SpecializedInserter, InsertionContext
from implicit_actor.insertion.pattern.inflect import conjugate
from implicit_actor.insertion.pattern.inflect_global import PRESENT, SINGULAR
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, \
    ImplicitSubjectType


class ImperativeInserter(SpecializedInserter):
    """
    Inserts imperatives.
    """

    def accepts(self, subject_type: ImplicitSubjectType):
        """
        Accepts imperatives.
        """
        return subject_type == ImplicitSubjectType.IMPERATIVE

    def insert(self, subj: CandidateActor, list_tokens: TokenList[str], target: ImplicitSubjectDetection, span: Span):
        """
        Do the insert.
        """
        # To the left of advmod if part of same phrase, e.g.,
        # "[You] [a]lways use the cheapest parts"
        # But ignore if the advmod if it is its own clause, e.g.,
        # As soon as you have an account, [you] log into it.
        insertion_point = target.token
        while pots := [x for x in insertion_point.lefts if
                       x.dep_ == "advmod" and x.pos_ == "ADV" and all(y.dep_ != "advcl" for y in x.children)]:
            # I am just assuming that the advmod tree is in-order
            insertion_point = min(pots, key=lambda x: x.i)

        # Conjugating the verb is only necessary if the verb is highly irregular, e.g., Be better -> You are better.
        conjugated_verb = SpecializedInserter._lower_case_first(conjugate(target.token.lemma_, PRESENT, 2, SINGULAR))

        ctx = InsertionContext(
            insertion_id=str(uuid.uuid4())
        )

        if insertion_point == target.token:
            if insertion_point.is_sent_start:
                list_tokens[
                    insertion_point.i - span.start] = self.subject_mapper("You", ctx) + " " + self.target_mapper(
                    SpecializedInserter._lower_case_first(
                        conjugated_verb), ctx) + insertion_point.whitespace_
            else:
                list_tokens[insertion_point.i - span.start] = self.subject_mapper("you",
                                                                                  ctx) + " " + self.target_mapper(
                    conjugated_verb, ctx) + insertion_point.whitespace_
        else:
            if insertion_point.is_sent_start:
                list_tokens[
                    insertion_point.i - span.start] = self.subject_mapper(
                    "You", ctx) + " " + SpecializedInserter._lower_case_first(
                    insertion_point.text) + insertion_point.whitespace_
            else:
                list_tokens[
                    insertion_point.i - span.start] = self.subject_mapper(
                    "you", ctx) + " " + insertion_point.text + insertion_point.whitespace_
            list_tokens[target.token.i - span.start] = self.target_mapper(conjugated_verb,
                                                                          ctx) + target.token.whitespace_
