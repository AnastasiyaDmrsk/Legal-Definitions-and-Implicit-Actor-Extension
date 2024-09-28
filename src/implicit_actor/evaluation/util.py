import csv
import math
from collections import defaultdict
from typing import List

import spacy

from implicit_actor.ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.candidate_extraction.CandidateActor import CandidateActor, CandidateSource
from implicit_actor.candidate_extraction.ComposedCandidateExtractor import ComposedCandidateExtractor
from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
from implicit_actor.candidate_extraction.PreambleExtractor import PreambleExtractor
from implicit_actor.candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from implicit_actor.candidate_filtering.ImperativeFilter import ImperativeFilter
from implicit_actor.candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from implicit_actor.candidate_filtering.PerplexityFilter import PerplexityFilter
from implicit_actor.candidate_filtering.SynsetFilter import SynsetFilter
from implicit_actor.candidate_filtering.VerbLinkFilter import VerbLinkFilter
from implicit_actor.evaluation.ClassificationStatisticsAccumulator import ClassificationStatisticsAccumulator
from implicit_actor.evaluation.FitlerFailAccumulator import FilterFailAccumulator
from implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection, \
    ImplicitSubjectType
from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector


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
            ImperativeFilter(),
            DependentOfSameSentenceFilter(),
            PartOfSpeechFilter(),
            VerbLinkFilter(
                add_preamble_verbs=False,
            ),
            # # SimilarVerbInDefinitionFilter(),
            SynsetFilter(),
            # # #     # # ChatGPTFilter(),
            # # #     # # TODO check if we can only compare target verb
            # # SimilarityFilter(use_context=False, model="en_use_lg"),
            # # #     # # TODO better tuning for the perplexity buffer (rho) value
            # # # ProximityFilter(),
            # # #     # # TODO check if this is broken with new candidate extractor
            # # #     # # TODO make this focus on key verbs (whatever that means) and give wordnet a try
            # # PreviouslyMentionedRelationFilter(),
            # # #     # # TODO This is broken
            # # CandidateTextOccurrenceFilter(),
            PerplexityFilter(max_returned=3, missing_subject_inserter=definition_candidate_inserter,
                             perplexity_buffer=1.3),
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

    n_clauses_inspected = 0
    n_clauses_inspected_where_detection = 0
    n_clauses_inspected_where_detection_and_gs = 0
    n_total_extracted_candidates = 0

    detection_accumulator = ClassificationStatisticsAccumulator()
    detection_accumulator_y_only = ClassificationStatisticsAccumulator()
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
    sent_output = ""

    with open("./data/gold_standard/implicit/gold_standard_new.csv", 'r', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        grouped_by_sentence = defaultdict(list)
        for line in list(reader)[1:232]:
            print(line)
            grouped_by_sentence[(line[0].strip(), line[1].strip())].append(line)

    for (source, sentence), lines in grouped_by_sentence.items():
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
            pass
            # TODO put back in when needed
            # ctx = gdpr_preamble + "\n" * 5 + ctx
        print("---")
        produced_sent = pipeline.apply(sentence, ctx)
        n_clauses_inspected += 1
        n_total_extracted_candidates += len(pipeline.last_initial_candidates())
        print(pipeline.last_initial_candidates())
        if pipeline.last_detections():
            n_clauses_inspected_where_detection += 1

        # Note, that we are only comparing the root word for each actor
        # This is not
        # target, actor, type (y/n/i)
        expected_detections_y = list(map(lambda x: x[3].lower(), filter(lambda x: x[2] == "y", lines)))
        expected_detections = list(map(lambda x: x[3].lower(), filter(lambda x: x[2] != "n", lines)))
        provided_detections = list(map(lambda x: x.token.text.lower(), pipeline.last_detections()))

        num_y = sum(map(lambda l: 1 if l[2] == "y" else 0, lines))

        detection_accumulator.tp += len(intersect_lists(expected_detections, provided_detections))
        detection_accumulator.fp += len(subtract_lists(provided_detections, expected_detections))
        detection_accumulator.fn += len(subtract_lists(expected_detections, provided_detections))

        detection_accumulator_y_only.tp += len(intersect_lists(expected_detections_y, provided_detections))
        detection_accumulator_y_only.fp += len(subtract_lists(provided_detections, expected_detections_y))
        detection_accumulator_y_only.fn += len(subtract_lists(expected_detections_y, provided_detections))

        bad = subtract_lists(provided_detections, expected_detections)
        for expected_sent in bad:
            d = next(p for p in pipeline.last_detections() if p.token.text.lower() == expected_sent)
            if d:
                passive_problems += f'"{d.token.text}";"{d.type}";"{sentence}";"{source}";"{pipeline.last_initial_candidates()}";"{pipeline.last_selected_candidates_before_tie_break()}"\n'

        if pipeline.last_detections() and num_y > 0:
            n_clauses_inspected_where_detection_and_gs += 1

        initial_candidate_text = [candidate.token.text.lower().strip() for candidate in
                                  pipeline.last_initial_candidates()]
        for (_, _, type_flag, _, _, actual_actor, *_) in lines:
            if type_flag != "y":
                continue

            if actual_actor.lower().strip() in initial_candidate_text:
                n_initial_correct_candidate += 1

        # Note, we know that every target always has the same actor per sentence in the GS thus some simplifications
        for (_, _, _, actual_target, _, actual_actor, *_) in lines:
            for provided_target, provided_actor in pipeline.last_selected_candidate_with_target():
                if provided_target.token.text not in actual_target:
                    continue

                f = False
                # if any(a.lower() in initial_candidate_text for a in actual_actor.split(" ")):
                #     n_initial_correct_candidate += 1
                #     f = True

                if provided_actor.token.text.lower() in actual_actor.lower():
                    n_correct_actor += 1
                    f = True

                # print("!!!!")
                # print(pipeline.last_selected_candidates_before_tie_break_root_only())
                # print(provided_target.token.text.lower())
                # print(actual_actor.lower())

                for top_5 in pipeline.last_selected_candidates_before_tie_break_root_only().get(
                        provided_target.token.text.lower(), [])[
                             :5]:
                    if top_5.lower() in actual_actor.lower():
                        n_top_5_actor += 1
                        f = True
                        break

                if f:
                    break

        n_inspected_actor += num_y

        original_sent = str(sentence).replace('"', "'").replace(";", "&#59;")
        expected_sent = str(line[0][6]).replace('"', "'").replace(";", "&#59;")
        provided_sent = str(produced_sent).replace('"', "'").replace(";", "&#59;")
        sent_output += f'"{original_sent}";"{expected_sent}";"{provided_sent}"\n'

        print(sentence)
        # print(expected_detections)
        print(pipeline.last_initial_candidates())
        print(list(map(lambda x: (x[3], x[5]), lines)))
        # print(pipeline.last_selected_candidates_before_tie_break())
        print(pipeline.last_selected_candidate_with_target())
        # print(pipeline.last_active_filters())

        # print(pipeline.last_selected_candidates())
        print(f"precision {detection_accumulator.precision()}, recall {detection_accumulator.recall()}")
        print(f"{str(detection_accumulator)}")
        print(f"y only {str(detection_accumulator_y_only)}")
        # Note, this is a "conditional probability" stat -> !!! no i do not think it is???
        print(
            f"correct actors {n_correct_actor} / {n_inspected_actor} ({n_correct_actor / n_inspected_actor if n_inspected_actor > 0 else math.nan})")
        print(
            f"correct top 5 actors {n_top_5_actor} / {n_inspected_actor} ({n_top_5_actor / n_inspected_actor if n_inspected_actor > 0 else math.nan})")
        print(
            f"correct initial candidates {n_initial_correct_candidate} / {n_inspected_actor} ({n_initial_correct_candidate / n_inspected_actor if n_inspected_actor > 0 else math.nan})"
        )

        print(
            f"clauses inspected {n_clauses_inspected}, n_clauses_inspected_where_chance {n_clauses_inspected_where_detection}, n_total_extracted_candidates {n_total_extracted_candidates}, n_clauses_inspected_where_detection_and_gs {n_clauses_inspected_where_detection_and_gs}"
        )

        print("---")

    with open("./data/output/passive_problems.txt", "w+", encoding="utf-8") as out:
        out.write(passive_problems)

    with open("./data/output/sent_output.txt", "w+", encoding="utf-8") as out:
        out.write(sent_output)

        # for (source, original_sentence, _, gs_verb, _, gs_subj, gs_enhanced, *_) in lines:
        #     ctx = ctx_file.read()
        #
        #     # act_verbs = pipeline.last_detections()
        #     # act_subjs = pipeline.last_selected_candidates()
        #
        #     print(pipeline.last_filter_log())


def eval_insertion():
    with open("./data/gold_standard/implicit/gold_standard.csv", 'r', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        grouped_by_sentence = defaultdict(list)
        for line in list(reader)[233:]:
            grouped_by_sentence[(line[0].strip(), line[1].strip())].append(line)

    nlp = spacy.load("en_core_web_trf")

    inserter = ImplicitSubjectInserterImpl()

    out_sents = "source;expected;actual;equals\n"

    for (_, sentence), lines in grouped_by_sentence.items():
        if lines[0][2] != "y":
            continue

        doc = nlp(sentence)

        expected = lines[0][6]

        verbs = [x[3].split(" ")[0].lower() for x in lines if x[2] == "y"]
        types = [_detection_type_from_str(x[4]) for x in lines if x[2] == "y"]

        verb_tokens = []
        # Single pass through document to select different tokens if two occurrences have the same orthography
        i = 0
        if verbs:
            for t in doc:
                if verbs[i] == t.text.lower():
                    verb_tokens.append(t)
                    i += 1
                if i == len(verbs):
                    break

        if len(verb_tokens) != len(verbs):
            print(f"Not all tokens found for sentence {sentence}, {verb_tokens}, {verbs}")
            print(lines)
            exit()

        targets = [
            ImplicitSubjectDetection(
                type=t,
                token=v,
            ) for v, t in zip(verb_tokens, types)
        ]

        candidate_tokens = [nlp(x[5]) for x in lines if x[2] == "y"]

        candidates = [
            CandidateActor(
                # Please note, this is not a great general approach, but I know
                # from the implementation that an artificial token here should work fine
                token=x[-1],
                source=CandidateSource.DEFINITION if x[0].pos_ != "DET" else CandidateSource.ARTIFICIAL,
            ) for x in candidate_tokens
        ]

        if len(candidates) != len(verbs):
            print(f"Could not generate all necessary candidates for sentence {sentence}, {candidates}")

        actual = inserter.insert(doc[:], targets, candidates)

        print("-")
        print(expected)
        print(actual)
        print("-")

        expected_san = expected.replace("\n", " ").replace('"', "'")
        actual_san = actual.replace("\n", " ").replace('"', "'")
        out_sents += f'{lines[0][0]};"{expected_san}";"{actual_san}";{expected == actual}\n'

    with open("./data/output/step_4_isolated.csv", "w+", encoding="utf-8") as f:
        f.write(out_sents)

        # print(
        #     [list((t.text.lower().split(" ")[0], line[3]) for t in doc) for line in lines]
        # )
        #
        # detections = [
        #     ImplicitSubjectDetection(
        #         type=_detection_type_from_str(line[5]),
        #         token=next((t for t in doc if t.text.lower().split(" ")[0] == line[3]), None),
        #     ) for line in lines if line[2] == "y"
        # ]

        # print(sentence, detections)

        # print(detections)


def _detection_type_from_str(detection_type: str) -> ImplicitSubjectType:
    d = {
        "passive": ImplicitSubjectType.PASSIVE,
        "gerund": ImplicitSubjectType.GERUND,
        "action_noun": ImplicitSubjectType.NOMINALIZED_VERB,
        "imperative": ImplicitSubjectType.IMPERATIVE,
        "infinitive": ImplicitSubjectType.IMPERATIVE,  # ?
    }
    return d[detection_type.strip().lower()]


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
