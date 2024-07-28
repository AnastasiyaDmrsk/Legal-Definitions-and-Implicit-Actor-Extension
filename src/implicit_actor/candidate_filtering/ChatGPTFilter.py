import warnings
from collections import defaultdict
from typing import List

import openai
from openai import OpenAI
from spacy.tokens import Token, Span

from src.implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from src.implicit_actor.insertion import ImplicitSubjectInserter
from src.implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from src.implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class ChatGPTFilter(CandidateFilter):

    def __init__(self, subject_inserter: ImplicitSubjectInserter = None, model="gpt-3.5-turbo"):
        self._subject_inserter = subject_inserter or ImplicitSubjectInserterImpl()
        self._client = OpenAI()
        self._model = model

    def filter(self, target: ImplicitSubjectDetection, candidates: List[Token], context: Span) -> List[Token]:
        """
        Asks ChatGPT to pick the best candidate.
        """

        sentence_to_candidate_mapping = defaultdict(list)

        for c in candidates:
            sentence_to_candidate_mapping[
                self._subject_inserter.insert(target.token.sent, [target], [c]).strip()].append(c)

        sentences = sentence_to_candidate_mapping.keys()
        sentences_str = "\n".join(f"{s}" for i, s in enumerate(sentences, 1))

        prompt = f"""
Given the following context:

{str(context)}

Which of the following sentences is most fitting? Provide only the sentence.

{sentences_str}"""

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=0,  # better replicability
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            res = completion.choices[0].message.content.strip()
            if res not in sentence_to_candidate_mapping:
                warnings.warn(
                    f"ChatGPT produced a sentence that is not part of of the provided sentences: '{res}'. "
                    f"You are a bad prompt engineer!")

            return sentence_to_candidate_mapping[res] or candidates
        except openai.BadRequestError as e:
            warnings.warn(f"Failed to generate issue request to OpenAI with error {e}.")
            return candidates
