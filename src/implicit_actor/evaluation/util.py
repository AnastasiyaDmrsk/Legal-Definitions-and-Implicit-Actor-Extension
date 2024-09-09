import csv
import math
from collections import defaultdict
from typing import List

import spacy

from implicit_actor.ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.candidate_extraction.ComposedCandidateExtractor import ComposedCandidateExtractor
from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
from implicit_actor.candidate_extraction.PreambleExtractor import PreambleExtractor
from implicit_actor.candidate_filtering.CandidateTextOccurrenceFilter import CandidateTextOccurrenceFilter
from implicit_actor.candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from implicit_actor.candidate_filtering.ImperativeFilter import ImperativeFilter
from implicit_actor.candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from implicit_actor.candidate_filtering.PerplexityFilter import PerplexityFilter
from implicit_actor.candidate_filtering.PreviouslyMentionedRelationFilter import PreviouslyMentionedRelationFilter
from implicit_actor.candidate_filtering.ProximityFilter import ProximityFilter
from implicit_actor.candidate_filtering.SimilarVerbInDefinitionFilter import SimilarVerbInDefinitionFilter
from implicit_actor.candidate_filtering.SimilarityFilter import SimilarityFilter
from implicit_actor.candidate_filtering.SynsetFilter import SynsetFilter
from implicit_actor.candidate_filtering.VerbLinkFilter import VerbLinkFilter
from implicit_actor.evaluation.ClassificationStatisticsAccumulator import ClassificationStatisticsAccumulator
from implicit_actor.evaluation.FitlerFailAccumulator import FilterFailAccumulator
from implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
from implicit_actor.missing_subject_detection.ImperativeDetector import ImperativeDetector
from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from implicit_actor.missing_subject_detection.NounVerbStemDetector import NounVerbStemDetector
from implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector
from src.implicit_actor.util import load_gold_standard, dependency_trees_equal


def evaluate_detection(expected: List[str], actual: List[str]) -> ClassificationStatisticsAccumulator:
    """
    Evaluates a list of detected subjects against a gold standard
    """

    acc = ClassificationStatisticsAccumulator()

    exp_grouping = defaultdict(int)
    act_grouping = defaultdict(int)

    for e in expected:
        exp_grouping[e] += 1

    for a in actual:
        act_grouping[a] += 1

    for k in exp_grouping.keys() | act_grouping.keys():
        exp = exp_grouping[k]
        act = act_grouping[k]
        acc.tp += min(exp, act)
        acc.fp += max(0, act - exp)
        acc.fn += max(0, exp - act)

    return acc


def run_gs_eval(pipeline: ImplicitSubjectPipeline, start=None, end=None):
    """
    Runs the provided pipeline against the gold standard and prints
    debug and evaluation data.

    :param pipeline: The pipeline to be used for evaluation
    :param start:    The index of the entry of the gold standard to start from.
    :param end:      The index of the last entry of the gold standard to inspect.
    """

    similarity_nlp = spacy.load("en_core_web_lg")
    n_inspected = 0
    n_correct = 0

    detection_accumulator = ClassificationStatisticsAccumulator()
    filter_stats_accumulator = FilterFailAccumulator()

    mask = ""
    for i, (source, inp, gs, impl_subjects, targets) in enumerate(list(load_gold_standard())[start:end]):

        print(f"Enter {i}")
        print("Context:")
        print(source)
        print("-" * 5)
        print("Inspected text:")
        print(inp)
        print("-" * 5)

        generated = pipeline.apply(
            inspected_text=inp,
            context=source
        )

        current_stats = evaluate_detection(targets, [x.token.text for x in pipeline.last_detections()])

        detection_accumulator.apply(current_stats)
        filter_stats_accumulator.apply(pipeline.last_filter_log(), targets, impl_subjects)

        print(
            f"Detection stats: Precision {current_stats.precision() * 100 :.2f}%, Recall {current_stats.recall() * 100 :.2f}%")

        gs_doc = similarity_nlp(gs)
        generated_doc = similarity_nlp(generated)
        similarity = gs_doc.similarity(generated_doc)
        n_inspected += 1

        if gs.strip() == generated.strip() or dependency_trees_equal(gs_doc, generated_doc):
            n_correct += 1
            mask += "x"
        elif similarity > 0.995:
            mask += "-"
        else:
            mask += "_"

        print("-" * 4)
        print("Expected:", gs)
        print("Actual:  ", generated)
        print("Similarity: ", similarity)
        print(f"Dependency equality: {dependency_trees_equal(gs_doc, generated_doc)}")

        print("Filter failures by filter:", filter_stats_accumulator.counts())
        print(f"Did not filter correct candidate: {filter_stats_accumulator.performance_str()}")
        print(f"Num filtered by filter: {filter_stats_accumulator.num_filtered()}")

        print("-" * 9)

    print(mask)
    result_txt = f"Correct: {n_correct}/{n_inspected} ({n_correct / n_inspected * 100 :.2f}%). Detection stats: Precision {detection_accumulator.precision() * 100 :.2f}%, Recall {detection_accumulator.recall() * 100 :.2f}%"
    print(result_txt)

    with open("./log/res", "a") as f:
        f.write(
            f"{mask} {result_txt} | {[x.__class__.__name__ for x in pipeline._missing_subject_detectors]} | {[x.__class__.__name__ for x in pipeline._candidate_filters]}\n")


def run_evaluation_2():
    # TODO clean up

    definition_candidate_inserter = ImplicitSubjectInserterImpl()

    # TODO show second most likely candidate on hover over candidate
    pipeline = ImplicitSubjectPipeline(
        missing_subject_detectors=[
            PassiveDetector(),
            # ImperativeDetector(),  # remove me
            GerundDetector(),
            NominalizedGerundWordlistDetector(),
            # NounVerbStemDetector(),  # remove me
        ],
        # missing_subject_filters=[MissingSubjectDetectionAuxFilter()],
        candidate_filters=[
            SynsetFilter(),
            DependentOfSameSentenceFilter(),
            VerbLinkFilter(
                add_preamble_verbs=False,
            ),
            ImperativeFilter(),
            # SimilarVerbInDefinitionFilter(),
            # PartOfSpeechFilter(),
            # #     # # ChatGPTFilter(),
            # #     # # TODO check if we can only compare target verb
            # SimilarityFilter(use_context=False, model="en_use_lg"),
            # #     # # TODO better tuning for the perplexity buffer (rho) value
            # # ProximityFilter(),
            # #     # # TODO check if this is broken with new candidate extractor
            # #     # # TODO make this focus on key verbs (whatever that means) and give wordnet a try
            # PreviouslyMentionedRelationFilter(),
            # #     # # TODO This is broken
            # CandidateTextOccurrenceFilter(),
            # PerplexityFilter(max_returned=3, missing_subject_inserter=definition_candidate_inserter,
            #                  perplexity_buffer=1.3),
        ],
        missing_subject_inserter=definition_candidate_inserter,
        candidate_extractor=ComposedCandidateExtractor([
            DefinitionCandidateExtractor(),
            PreambleExtractor(),
            # SubjectObjectCandidateExtractor(),
        ]),
        verbose=False
    )

    n_correct_actor = 0
    n_inspected_actor = 0
    n_top_5_actor = 0
    n_initial_correct_candidate = 0

    detection_accumulator = ClassificationStatisticsAccumulator()
    filter_stats_accumulator = FilterFailAccumulator()

    with open(f"./data/gold_standard/implicit/article4.txt", "r",
              encoding="utf-8") as additional_ctx_file:
        art4 = additional_ctx_file.read()

    with open(f"./data/gold_standard/implicit/gdpr_preamble.txt", "r",
              encoding="utf-8") as gdpr_preamble_file:
        gdpr_preamble = gdpr_preamble_file.read()

    with open(f"./data/gold_standard/implicit/32017R1563.txt", "r", encoding="utf-8") as f:
        t_32017R1563 = f.read()

    with open(f"./data/gold_standard/implicit/32019R0517.txt", "r", encoding="utf-8") as f:
        t_32019R0517 = f.read()

    with open(f"./data/gold_standard/implicit/32021R0444.txt", "r", encoding="utf-8") as f:
        t_32021R0444 = f.read()

    passive_problems = ""

    with open("./data/gold_standard/implicit/gold_standard.csv", 'r', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        next(reader, None)

        # for _ in range(55):
        #     next(reader, None)

        # Now, this could be done more efficiently, but that is not how we roll :)
        grouped_by_sentence = defaultdict(list)
        for line in reader:
            grouped_by_sentence[line[1].strip()].append(line)

    for sentence, lines in grouped_by_sentence.items():
        # We assume that if the sentences are the same, the source is also the same
        source = lines[0][0]

        if source == "32017R1563":
            ctx = t_32017R1563
        elif source == "32021R0444":
            ctx = t_32021R0444
        elif source == "32019R0517":
            ctx = t_32019R0517
        else:
            with open(f"./data/gold_standard/implicit/{source}.txt", "r", encoding="utf-8") as ctx_file:
                ctx = art4 + "\n" * 5 + ctx_file.read()

        if source not in ["32017R1563", "32021R0444", "32019R0517"]:
            # ctx = art4 + "\n" * 5 + ctx  # gdpr_preamble + "\n" * 5 + art4 + "\n" * 5 + ctx
            ctx = gdpr_preamble + "\n" * 5 + art4 + "\n" * 5 + ctx
        print("---")
        print(pipeline.apply(sentence, ctx))

        # target, actor, type (y/n/i)
        expected_detections = list(map(lambda x: x[3].split(" ")[0].lower(), filter(lambda x: x[2] != "n", lines)))
        provided_detections = list(map(lambda x: x.token.text.lower(), pipeline.last_detections()))

        detection_accumulator.tp += len(intersect_lists(expected_detections, provided_detections))
        detection_accumulator.fp += len(subtract_lists(provided_detections, expected_detections))
        detection_accumulator.fn += len(subtract_lists(expected_detections, provided_detections))

        bad = subtract_lists(provided_detections, expected_detections)
        for b in bad:
            d = next(p for p in pipeline.last_detections() if p.token.text.lower() == b)
            if d:
                passive_problems += f'"{d.token.text}";"{d.type}";"{sentence}";"{source}";"{pipeline.last_initial_candidates()}";"{pipeline.last_selected_candidates_before_tie_break()}"\n'

        num_y = sum(map(lambda l: 1 if l[2] == "y" else 0, lines))

        initial_candidate_text = [candidate.token.text.lower() for candidate in pipeline.last_initial_candidates()]

        # Note, we know that every target always has the same actor per sentence in the GS thus some simplifications
        for provided_target, provided_actor in pipeline.last_selected_candidate_with_target():
            for (_, _, _, actual_target, _, actual_actor, *_) in lines:
                if provided_target.token.text not in actual_target:
                    break

                print("?")
                print(actual_actor, initial_candidate_text)
                print("?")
                if any(a.lower() in initial_candidate_text for a in actual_actor.split(" ")):
                    n_initial_correct_candidate += 1

                if provided_actor.token.text.lower() in actual_actor.lower():
                    n_correct_actor += 1
                    n_top_5_actor += 1
                    break

                if any(a.lower() in pipeline.last_selected_candidates_before_tie_break().get(provided_target.token.text, [])[:5]
                       for a in
                       actual_actor.split(" ")):
                    n_top_5_actor += 1
                    break

        n_inspected_actor += num_y

        print(sentence)
        # print(expected_detections)
        print(pipeline.last_initial_candidates())
        print(list(map(lambda x: (x[3], x[5]), lines)))
        print(pipeline.last_selected_candidates_before_tie_break())
        # print(pipeline.last_selected_candidate_with_target())
        # print(pipeline.last_active_filters())

        # print(pipeline.last_selected_candidates())
        print(f"precision {detection_accumulator.precision()}, recall {detection_accumulator.recall()}")
        # Note, this is a "conditional probability" stat -> !!! no i do not think it is???
        print(
            f"correct actors {n_correct_actor} / {n_inspected_actor} ({n_correct_actor / n_inspected_actor if n_inspected_actor > 0 else math.nan})")
        print(
            f"correct top 5 actors {n_top_5_actor} / {n_inspected_actor} ({n_top_5_actor / n_inspected_actor if n_inspected_actor > 0 else math.nan})")
        print(
            f"correct initial candidates {n_initial_correct_candidate} / {n_inspected_actor} ({n_initial_correct_candidate / n_inspected_actor if n_inspected_actor > 0 else math.nan})"
        )
        print("---")

    with open("./data/output/passive_problems.txt", "w+", encoding="utf-8") as out:
        out.write(passive_problems)

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
