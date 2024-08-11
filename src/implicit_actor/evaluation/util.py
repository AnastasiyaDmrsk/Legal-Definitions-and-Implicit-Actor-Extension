from collections import defaultdict
from typing import List

import spacy

from implicit_actor.ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.evaluation.ClassificationStatisticsAccumulator import ClassificationStatisticsAccumulator
from implicit_actor.evaluation.FitlerFailAccumulator import FilterFailAccumulator
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



