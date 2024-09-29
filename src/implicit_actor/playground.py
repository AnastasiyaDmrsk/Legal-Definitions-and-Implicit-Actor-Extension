import csv
import math
from cProfile import Profile
from collections import defaultdict
from pstats import Stats, SortKey
from typing import List

import spacy
from nltk import PorterStemmer
from spacy import displacy

from implicit_actor.ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
from implicit_actor.candidate_filtering.CandidateTextOccurrenceFilter import CandidateTextOccurrenceFilter
from implicit_actor.candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from implicit_actor.candidate_filtering.ImperativeFilter import ImperativeFilter
from implicit_actor.candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from implicit_actor.candidate_filtering.PerplexityFilter import PerplexityFilter
from implicit_actor.candidate_filtering.PreviouslyMentionedRelationFilter import PreviouslyMentionedRelationFilter
from implicit_actor.candidate_filtering.SynsetFilter import SynsetFilter
from implicit_actor.evaluation.ClassificationStatisticsAccumulator import ClassificationStatisticsAccumulator
from implicit_actor.evaluation.FitlerFailAccumulator import FilterFailAccumulator
from implicit_actor.evaluation.util import run_evaluation, eval_insertion
from implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
from implicit_actor.missing_subject_detection.ImperativeDetector import ImperativeDetector
from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from implicit_actor.missing_subject_detection.NounVerbStemDetector import NounVerbStemDetector
from implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector

# from spacy import displacy

# from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
# from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
# from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
# from implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector


# from dotenv import load_dotenv
# from nltk.stem import PorterStemmer
# from spacy import displacy

# from src.implicit_actor.insertion.pattern.inflect import lexeme
# from src.implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector


from nltk.corpus import wordnet as wn

from implicit_actor.util import get_noun_chunk, has_explicit_subject


# load_dotenv()


def ctx_name_from_num(s: str) -> str:
    if s.strip() in ["32017R1563", "32021R0444", "32019R0517"]:
        return s
    return "article5"


def main():
    """
    This is just a file for messing around mostly with the dependency parser.
    Nothing of value can be found here.
    """

    # nlp = spacy.load('en_core_web_trf')
    # with open("./data/gold_standard/implicit/article4.txt", encoding="utf-8") as file:
    #     gdpr = file.read()
    #
    # doc = nlp(gdpr)
    #
    # for t in doc:
    #     print(t)


    #
    # for token in doc:
    #     if token.tag_ == "VBG" and (token.dep_ == "pcomp" or token.dep_ == "prep") and not has_explicit_subject(token):
    #         print(token.text, token.dep_, token.sent[:20])


    # lexnames = [x.lexname() for x in wn.synsets("entity", pos=wn.NOUN)]

    # print(lexnames)

    # print(DefinitionCandidateExtractor().extract(
    #     doc
    # ))

    # with Profile() as profile:

    # lexnames = [x.lexname() for x in wn.synsets("data", pos=wn.NOUN)]
    # print(lexnames)
    #

    # nlp = spacy.load('en_core_web_trf')
    # displacy.serve(nlp("""
    # The processing of personal data by an authorised entity carried out within the framework of this Regulation by an authorised entity shall be carried out in compliance with Directives 95/46/EC and 2002/58/EC.
    # """))

    # with open(f"./data/gold_standard/implicit/article4.txt", "r",
    #           encoding="utf-8") as additional_ctx_file:
    #     art4 = additional_ctx_file.read()
    # doc = nlp(art4)
    #
    # print(
    #     [get_noun_chunk(t) for t in doc if t.dep_ in {"relcl", "acl"} and t.pos_ == "VERB"]
    # )

    # displacy.serve(
    #     doc, style="dep"
    # )

    # doc = nlp(
    #     "‘processing’ means any operation or set of operations which is performed on personal data or on sets of personal data, whether or not by automated means, such as collection, recording, organisation, structuring, storage, adaptation or alteration, retrieval, consultation, use, disclosure by transmission, dissemination or otherwise making available, alignment or combination, restriction, erasure or destruction;")
    #
    # print(
    #     NounVerbStemDetector().detect(doc[:])
    # )

    """
    precision 0.23448275862068965, recall 0.7692307692307693
    correct actors 53 / 144 (0.3680555555555556)
    correct top 5 actors 70 / 144 (0.4861111111111111)
    """

    """
    Outgoing relcl/acl
    precision 0.228, recall 0.7737556561085973
    """

    # nlp = spacy.load("en_core_web_sm")
    #
    # doc = nlp("The member states shall establish something")
    # displacy.serve(
    #     doc, style="dep"
    # )

    """
    With partition of lambda stuff
    precision 0.27682403433476394, recall 0.583710407239819
    correct actors 20 / 144 (0.1388888888888889)
    correct top 5 actors 63 / 144 (0.4375)
    correct initial candidates 81 / 144 (0.5625)
    ---
    """

    # def _t(token):
    #     lexnames = [x.lexname() for x in wn.synsets(token.lemma_, pos=wn.NOUN)]
    #     # print(token.lemma_, lexnames)
    #     return "noun.act" in lexnames

    #
    # stemmer = PorterStemmer()
    #
    # with open("./data/external/en-verbs.txt", 'r', encoding="utf-8") as wf:
    #     verbs = {y for x in wf.readlines() if not x.startswith(";") for y in (x.split(",")[0], x.split(",")[5])}

    # eval_insertion()
    run_evaluation()

    # Stats(profile).sort_stats(SortKey.CUMULATIVE).print_stats()

    # nlp = spacy.load(
    #     "en_core_web_trf"
    # )
    #
    # doc = nlp(
    #     """
    #    (2) ‘processing’ means any operation or set of operations which is performed on personal data or on sets of personal data, whether or not by automated means, such as collection, recording, organisation, structuring, storage, adaptation or alteration, retrieval, consultation, use, disclosure by transmission, dissemination or otherwise making available, alignment or combination, restriction, erasure or destruction;
    #    """)
    #
    # for t in doc:
    #     print(t.text, _t(t), t.pos_, t.lemma_ in verbs)

    """
    With preamble for linker
    precision 0.23666210670314639, recall 0.7828054298642534
    correct actors 22 / 144 (0.1527777777777778)
    correct top 5 actors 71 / 144 (0.4930555555555556)
    """

    # nlp = spacy.load("en_core_web_lg")
    # with open("./data/gold_standard/implicit/gold_standard.csv", 'r', encoding="utf-8") as file:
    #     reader = csv.reader(file, delimiter=";")
    #     next(reader, None)
    #
    #     for _, sent, *_ in reader:
    #         doc = nlp(sent)
    #         print(PassiveDetector().detect(doc[:]))

    # text = unicodedata.normalize("NFKD", text)
    # if "’" in text:
    #     doc = nlp(
    #         text.split(";")[0]
    #     )  # this way only the most important part of the definition will be examined
    #     definition_set = set()
    #     explanation_set = set()
    #     # search for the first verb after the definition
    #     first_verb = None
    #     for token in doc:
    #         if "mean" in token.text:
    #             first_verb = token
    #             break
    #         if token.dep_ == "ROOT" and (token.pos_ == "VERB"
    #                                      or token.pos_ == "AUX"):
    #             first_verb = token
    #             break
    #     if first_verb is not None and is_synonym(first_verb.lemma_):
    #         definition = text[:first_verb.idx].strip()
    #         explanation = text[first_verb.idx:].strip()
    #         d = [s for s in definition.split("\n") if s != ""]
    #         e = [s for s in explanation.split("\n") if s != ""]
    #         save_in_annotations("".join(d), "".join(e))
    #         for element in d:
    #             if element.__contains__("‘"):
    #                 if element.__contains__(" and ‘") or element.__contains__(
    #                         " or ‘") or element.__contains__(", ‘"):
    #                     definition_set = split_multiples(element)
    #                 else:
    #                     definition_set.add(element)
    #         for element_e in e:
    #             if element_e != "" and element_e[0] != "(":
    #                 # multiple explanations
    #                 if len(e) > 1 and element_e.__contains__(":"):
    #                     base = element_e
    #                     while len(e) > e.index(element_e) + 1:
    #                         next_element = e[e.index(element_e) + 1]
    #                         if next_element[0] == "(" and len(
    #                                 e) > e.index(element_e) + 2:
    #                             new_element = base + " " + e[e.index(element_e)
    #                                                          + 2]
    #                             element_e = e[e.index(element_e) + 2]
    #                         else:
    #                             new_element = base + " " + next_element
    #                             element_e = e[e.index(element_e) + 1]
    #                         explanation_set.add(new_element)
    #                     break
    #                 # single explanation
    #                 else:
    #                     explanation_set.add(element_e)
    #         global definitions
    #         save_in_list(definition_set, explanation_set)
    #         d_set = tuple(definition_set)
    #         definitions[d_set] = explanation_set





if __name__ == "__main__":
    main()
