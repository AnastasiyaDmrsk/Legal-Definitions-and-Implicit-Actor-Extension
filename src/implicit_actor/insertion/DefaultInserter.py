import itertools
import uuid
from typing import List

from spacy import displacy
from spacy.tokens import Token, Span

from src.implicit_actor.insertion.SpecializedInserter import SpecializedInserter, InsertionContext
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, \
    ImplicitSubjectType


class DefaultInserter(SpecializedInserter):
    """
    Default insertion strategy but is specialized for passive and nouns.
    """

    def accepts(self, subject_type: ImplicitSubjectType):
        """
        Accept all.
        """
        return True

    def insert(self, subj: Token, list_tokens: List[str], target: ImplicitSubjectDetection, span: Span):
        """
        Do the insert,
        """
        # Skip subtrees that have an ADP as head and are linked by prep.
        potential_insertion_points = list(itertools.chain.from_iterable(
            x.subtree for x in target.token.rights if
            (x.dep_ == "prep" and x.pos_ == "ADP") or x.dep_ in {"dative", "advmod", "nummod"})) or [
                                         target.token]
        insertion_point = max(potential_insertion_points, key=lambda tok: tok.i) or target.token

        # I think I remember this from an English course but cannot find a source for this anymore
        if abs(insertion_point.i - target.token.i) > 9:
            insertion_point = target.token

        ctx = InsertionContext(
            insertion_id=str(uuid.uuid4())
        )

        cleaned_subj = SpecializedInserter._clean_subject(subj)

        if insertion_point.i == target.token.i:
            list_tokens[
                insertion_point.i - span.start] = self.target_mapper(insertion_point.text,
                                                                     ctx) + " by " + self.subject_mapper(
                cleaned_subj.strip(), ctx) + insertion_point.whitespace_
        else:
            list_tokens[
                insertion_point.i - span.start] = insertion_point.text + " by " + self.subject_mapper(
                cleaned_subj.strip(), ctx) + insertion_point.whitespace_
            list_tokens[target.token.i - span.start] = self.target_mapper(target.token, ctx) + target.token.whitespace_
