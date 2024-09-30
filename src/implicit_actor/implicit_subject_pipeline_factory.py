from typing import List, Iterator

from implicit_actor.ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.candidate_extraction.ComposedCandidateExtractor import ComposedCandidateExtractor
from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
from implicit_actor.candidate_extraction.PreambleExtractor import PreambleExtractor
from implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from implicit_actor.candidate_filtering.CandidateTextOccurrenceFilter import CandidateTextOccurrenceFilter
from implicit_actor.candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from implicit_actor.candidate_filtering.ImperativeFilter import ImperativeFilter
from implicit_actor.candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from implicit_actor.candidate_filtering.PerplexityFilter import PerplexityFilter
from implicit_actor.candidate_filtering.PreviouslyMentionedRelationFilter import PreviouslyMentionedRelationFilter
from implicit_actor.candidate_filtering.ProximityFilter import ProximityFilter
from implicit_actor.candidate_filtering.SimilarityFilter import SimilarityFilter
from implicit_actor.candidate_filtering.SynsetFilter import SynsetFilter
from implicit_actor.candidate_filtering.VerbLinkFilter import VerbLinkFilter
from implicit_actor.insertion.DefaultInserter import DefaultInserter
from implicit_actor.insertion.GerundInserter import GerundInserter
from implicit_actor.insertion.ImperativeInserter import ImperativeInserter
from implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from implicit_actor.insertion.SpecializedInserter import InsertionContext
from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
from implicit_actor.missing_subject_detection.ImperativeDetector import ImperativeDetector
from implicit_actor.missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector
from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector

_filter_options = [
    ("Imperative Filter", ImperativeFilter.__name__, ImperativeFilter()),
    ("DependentOfSameSentenceFilter", DependentOfSameSentenceFilter.__name__,
     DependentOfSameSentenceFilter()),
    ("Part-Of-Speech Filter", PartOfSpeechFilter.__name__, PartOfSpeechFilter()),
    ("VerbLink Filter", VerbLinkFilter.__name__, VerbLinkFilter()),
    ("SynsetFilter", SynsetFilter.__name__, SynsetFilter()),
    # ("ChatGPT Filter", ChatGPTFilter.__name__, ChatGPTFilter()),
    ("Similarity Filter", SimilarityFilter.__name__, SimilarityFilter(use_context=True, model="en_use_lg")),
    ("Candidate Text Occurrence Filter", CandidateTextOccurrenceFilter.__name__,
     CandidateTextOccurrenceFilter()),
    ("Proximity Filter", ProximityFilter.__name__, ProximityFilter()),
    ("Previously MentionedRelation Filter", PreviouslyMentionedRelationFilter.__name__,
     PreviouslyMentionedRelationFilter()),
    (
        "Perplexity Filter (Warning, may lead to long execution times)",
        PerplexityFilter.__name__, PerplexityFilter(max_returned=10000)),
]

_default_filter_selection = [
    ImperativeFilter.__name__,
    DependentOfSameSentenceFilter.__name__,
    PartOfSpeechFilter.__name__,
    VerbLinkFilter.__name__,
    SynsetFilter.__name__,
    PerplexityFilter.__name__,
]

_detector_options = [
    ("Passive Detector", PassiveDetector.__name__, PassiveDetector()),
    ("Gerund Detector", GerundDetector.__name__, GerundDetector()),
    ("Gerundive Nominal Detector", NominalizedGerundWordlistDetector.__name__, NominalizedGerundWordlistDetector()),
    ("Imperative Detector", ImperativeDetector.__name__, ImperativeDetector()),
]

_default_detector_options = [
    PassiveDetector.__name__,
    NominalizedGerundWordlistDetector.__name__,
]


def filters_default_form_selection():
    """
    Provides a default selection for filters
    """
    return _default_filter_selection


def detectors_default_form_selection():
    """
    Provides a default selection for detectors
    """
    return _default_detector_options


def filters_form_options():
    """
    Returns the options to use for filterion
    """
    return map(lambda x: (x[1], x[0]), _filter_options)


def detectors_form_options():
    return map(lambda x: (x[1], x[0]), _detector_options)


def implicit_subject_pipeline_from_form(
        selected: List[str]
) -> Iterator[CandidateFilter]:
    for f in _filter_options:
        if f[1] in selected:
            yield f[2]


def implicit_subject_pipeline_detectors_from_form(
        selected: List[str]
) -> Iterator[ImplicitSubjectDetector]:
    for f in _detector_options:
        if f[1] in selected:
            yield f[2]


def create_implicit_subject_pipeline(
        selected_filters: List[str],
        add_preamble_extraction: bool,
        selected_detectors: List[str],
) -> ImplicitSubjectPipeline:
    return ImplicitSubjectPipeline(
        missing_subject_detectors=list(implicit_subject_pipeline_detectors_from_form(selected_detectors)),
        candidate_filters=list(implicit_subject_pipeline_from_form(selected_filters)),
        missing_subject_inserter=ImplicitSubjectInserterImpl(
            inserters=[
                GerundInserter(_subject_annotation, _target_annotation),
                ImperativeInserter(_subject_annotation, _target_annotation),
                DefaultInserter(_subject_annotation, _target_annotation)
            ]
        ),
        candidate_extractor=ComposedCandidateExtractor(
            [
                DefinitionCandidateExtractor(),
                PreambleExtractor(),
            ] if add_preamble_extraction else [
                DefinitionCandidateExtractor()
            ]),
        verbose=False
    )


def _subject_annotation(x: str, ctx: InsertionContext):
    return f'<span class="subject" data-highlight="0" data-insertion-id="{ctx.insertion_id}">{x}</span>'


def _target_annotation(x: str, ctx: InsertionContext):
    return f'<span class="target" data-highlight="0" data-insertion-id="{ctx.insertion_id}">{x}</span>'
