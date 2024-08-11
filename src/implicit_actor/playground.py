import csv
import math
from collections import defaultdict
from typing import List

from implicit_actor.ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
from implicit_actor.candidate_filtering.CandidateTextOccurrenceFilter import CandidateTextOccurrenceFilter
from implicit_actor.candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from implicit_actor.candidate_filtering.ImperativeFilter import ImperativeFilter
from implicit_actor.candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from implicit_actor.candidate_filtering.PerplexityFilter import PerplexityFilter
from implicit_actor.candidate_filtering.PreviouslyMentionedRelationFilter import PreviouslyMentionedRelationFilter
from implicit_actor.evaluation.ClassificationStatisticsAccumulator import ClassificationStatisticsAccumulator
from implicit_actor.evaluation.FitlerFailAccumulator import FilterFailAccumulator
from implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
from implicit_actor.missing_subject_detection.ImperativeDetector import ImperativeDetector
from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
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

    #
    # nlp = spacy.load("en_core_web_trf")
    # with open(f"./data/gold_standard/implicit/32019R0517.txt", "r",
    #           encoding="utf-8") as art_4_file:
    #     art_4 = art_4_file.read()
    #
    # pipeline = ImplicitSubjectPipeline(
    #     missing_subject_detectors=[
    #         PassiveDetector(),
    #         ImperativeDetector(),
    #         GerundDetector(),
    #         NominalizedGerundWordlistDetector(),
    #         # NounVerbStemDetector(),
    #     ],
    #     candidate_filters=[
    #         SynsetFilter(),
    #     ],
    #     missing_subject_inserter=definition_candidate_inserter,
    #     candidate_extractor=DefinitionCandidateExtractor(),
    #     verbose=True
    # )
    #
    # pipeline.apply(art_4)
    #
    # return

    definition_candidate_inserter = ImplicitSubjectInserterImpl.for_definition_candidates()

    # TODO show second most likely candidate on hover over candidate
    pipeline = ImplicitSubjectPipeline(
        missing_subject_detectors=[
            PassiveDetector(),
            ImperativeDetector(),
            GerundDetector(),
            NominalizedGerundWordlistDetector(),
            # NounVerbStemDetector(),
        ],
        candidate_filters=[
            ImperativeFilter(),
            PartOfSpeechFilter(),
            DependentOfSameSentenceFilter(),
            # ChatGPTFilter(),
            # TODO check if we can only compare target verb
            # SimilarityFilter(use_context=False, model="en_use_lg"),
            # TODO better tuning for the perplexity buffer (rho) value
            PerplexityFilter(max_returned=100000, missing_subject_inserter=definition_candidate_inserter,
                             perplexity_buffer=1.1),
            # ProximityFilter(),
            # TODO check if this is broken with new candidate extractor
            # TODO make this focus on key verbs (whatever that means) and give wordnet a try
            PreviouslyMentionedRelationFilter(),
            # TODO This is broken
            CandidateTextOccurrenceFilter(),
        ],
        missing_subject_inserter=definition_candidate_inserter,
        # TODO filter definitions on only relevant verb in definition
        candidate_extractor=DefinitionCandidateExtractor(),  # SubjectObjectCandidateExtractor(),
        verbose=False
    )

    n_correct_actor = 0
    n_inspected_actor = 0

    detection_accumulator = ClassificationStatisticsAccumulator()
    filter_stats_accumulator = FilterFailAccumulator()

    with open("./data/gold_standard/implicit/gold_standard.csv", 'r', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        next(reader, None)

        for _ in range(6):
            next(reader, None)

        # Now, this could be done more efficiently, but that is not how we roll :)
        grouped_by_sentence = defaultdict(list)
        for line in reader:
            grouped_by_sentence[line[1]].append(line)

        for sentence, lines in grouped_by_sentence.items():
            # We assume that if the sentences are the same, the source is also the same
            source = lines[0][0]
            with open(f"./data/gold_standard/implicit/{source}.txt", "r", encoding="utf-8") as ctx_file:
                ctx = ctx_file.read()
            additional_ctx = ""
            if source not in ["32017R1563", "32021R0444", "32019R0517"]:
                with open(f"./data/gold_standard/implicit/article4.txt", "r",
                          encoding="utf-8") as additional_ctx_file:
                    additional_ctx = additional_ctx_file.read()
            act_enhanced = pipeline.apply(sentence, ctx + "\n\n\n" + additional_ctx)

            # target, actor, type (y/n/i)
            expected_detections = list(map(lambda x: x[3].split(" ")[0], filter(lambda x: x[2] != "n", lines)))
            provided_detections = list(map(lambda x: x.token.text, pipeline.last_detections()))

            detection_accumulator.tp += len(intersect_lists(expected_detections, provided_detections))
            detection_accumulator.fp += len(subtract_lists(provided_detections, expected_detections))
            detection_accumulator.fn += len(subtract_lists(expected_detections, provided_detections))

            num_y = sum(map(lambda l: 1 if l[2] == "y" else 0, lines))

            # Note, we know that every target always has the same actor per sentence in the GS thus some simplifications
            for provided_target, provided_actor in pipeline.last_selected_candidate_with_target():
                for (_, _, _, actual_target, _, actual_actor, *_) in lines:
                    if provided_target.token.text in actual_target and provided_actor.text in actual_actor:
                        n_correct_actor += 1
                        print("Found")
                        break

                else:
                    print("None found")

            n_inspected_actor += num_y


            print(pipeline.last_selected_candidate_with_target())
            print(list(map(lambda x: (x[3], x[5]), lines)))

            # print(pipeline.last_selected_candidates())
            print(f"precision {detection_accumulator.precision()}, recall {detection_accumulator.recall()}")
            # Note, this is a "conditional probability" stat
            print(
                f"correct actors {n_correct_actor} / {n_inspected_actor} ({n_correct_actor / n_inspected_actor if n_inspected_actor > 0 else math.nan})")
            print("---")

            # for (source, original_sentence, _, gs_verb, _, gs_subj, gs_enhanced, *_) in lines:
            #     ctx = ctx_file.read()
            #
            #     # act_verbs = pipeline.last_detections()
            #     # act_subjs = pipeline.last_selected_candidates()
            #
            #     print(pipeline.last_filter_log())


def subtract_lists(list1: List[str], list2: List[str]):
    """
    Checks the difference between two lists
    """
    result = list1.copy()
    for element in list2:
        if element in result:
            result.remove(element)
    return result


def intersect_lists(list1, list2):
    """
    Checks the intersection between two lists
    """
    result = []
    list2_copy = list2.copy()
    for element in list1:
        if element in list2_copy:
            result.append(element)
            list2_copy.remove(element)
    return result


if __name__ == "__main__":
    main()
