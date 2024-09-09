import csv
import math
from cProfile import Profile
from collections import defaultdict
from pstats import Stats, SortKey
from typing import List

import spacy
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
from implicit_actor.evaluation.util import run_evaluation_2
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

from implicit_actor.util import get_noun_chunk


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

    # lexnames = [x.lexname() for x in wn.synsets("entity", pos=wn.NOUN)]

    # print(lexnames)

    # with Profile() as profile:

    # lexnames = [x.lexname() for x in wn.synsets("data", pos=wn.NOUN)]
    # print(lexnames)
    #

    # nlp = spacy.load('en_core_web_trf')
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

    run_evaluation_2()
    # Stats(profile).sort_stats(SortKey.CUMULATIVE).print_stats()

    # nlp = spacy.load(
    #     "en_core_web_trf"
    # )
    # #
    # doc = nlp(
    #     """
    #     For the purposes of this Regulation: (1) ‘personal data’ means any information relating to an identified or identifiable natural person (‘data subject’); an identifiable natural person is one who can be identified, directly or indirectly, in particular by reference to an identifier such as a name, an identification number, location data, an online identifier or to one or more factors specific to the physical, physiological, genetic, mental, economic, cultural or social identity of that natural person; (2) ‘processing’ means any operation or set of operations which is performed on personal data or on sets of personal data, whether or not by automated means, such as collection, recording, organisation, structuring, storage, adaptation or alteration, retrieval, consultation, use, disclosure by transmission, dissemination or otherwise making available, alignment or combination, restriction, erasure or destruction; (3) ‘restriction of processing’ means the marking of stored personal data with the aim of limiting their processing in the future; (4) ‘profiling’ means any form of automated processing of personal data consisting of the use of personal data to evaluate certain personal aspects relating to a natural person, in particular to analyse or predict aspects concerning that natural person's performance at work, economic situation, health, personal preferences, interests, reliability, behaviour, location or movements; (5) ‘pseudonymisation’ means the processing of personal data in such a manner that the personal data can no longer be attributed to a specific data subject without the use of additional information, provided that such additional information is kept separately and is subject to technical and organisational measures to ensure that the personal data are not attributed to an identified or identifiable natural person; (6) ‘filing system’ means any structured set of personal data which are accessible according to specific criteria, whether centralised, decentralised or dispersed on a functional or geographical basis; (7) ‘controller’ means the natural or legal person, public authority, agency or other body which, alone or jointly with others, determines the purposes and means of the processing of personal data; where the purposes and means of such processing are determined by Union or Member State law, the controller or the specific criteria for its nomination may be provided for by Union or Member State law; (8) ‘processor’ means a natural or legal person, public authority, agency or other body which processes personal data on behalf of the controller;
    #     """)

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


if __name__ == "__main__":
    main()
