from typing import List, Iterator

from implicit_actor.ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
from implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from implicit_actor.candidate_filtering.CandidateTextOccurrenceFilter import CandidateTextOccurrenceFilter
from implicit_actor.candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from implicit_actor.candidate_filtering.ImperativeFilter import ImperativeFilter
from implicit_actor.candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from implicit_actor.candidate_filtering.PerplexityFilter import PerplexityFilter
from implicit_actor.candidate_filtering.PreviouslyMentionedRelationFilter import PreviouslyMentionedRelationFilter
from implicit_actor.candidate_filtering.ProximityFilter import ProximityFilter
from implicit_actor.candidate_filtering.SimilarityFilter import SimilarityFilter
from implicit_actor.insertion.DefaultInserter import DefaultInserter
from implicit_actor.insertion.GerundInserter import GerundInserter
from implicit_actor.insertion.ImperativeInserter import ImperativeInserter
from implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from implicit_actor.insertion.SpecializedInserter import InsertionContext
from implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
from implicit_actor.missing_subject_detection.ImperativeDetector import ImperativeDetector
from implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import NominalizedGerundWordlistDetector
from implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector

_filter_options = [
    ("Imperative Filter", ImperativeFilter.__name__, ImperativeFilter()),
    ("Part-Of-Speech Filter", PartOfSpeechFilter.__name__, PartOfSpeechFilter()),
    (
        "DependentOfSameSentenceFilter", DependentOfSameSentenceFilter.__name__,
        DependentOfSameSentenceFilter()),
    # ("ChatGPT Filter", ChatGPTFilter.__name__, ChatGPTFilter()),
    ("Similarity Filter", SimilarityFilter.__name__, SimilarityFilter(use_context=True, model="en_use_lg")),
    ("Perplexity Filter", PerplexityFilter.__name__, PerplexityFilter(max_returned=10000)),
    ("Candidate Text Occurrence Filter", CandidateTextOccurrenceFilter.__name__,
     CandidateTextOccurrenceFilter()),
    ("Proximity Filter", ProximityFilter.__name__, ProximityFilter()),
    ("Previously MentionedRelation Filter", PreviouslyMentionedRelationFilter.__name__,
     PreviouslyMentionedRelationFilter()),
]

_default_selection = [
    ImperativeFilter.__name__,
    PartOfSpeechFilter.__name__,
    DependentOfSameSentenceFilter.__name__,
    CandidateTextOccurrenceFilter.__name__,
    PreviouslyMentionedRelationFilter.__name__,
]


def filters_default_form_selection():
    """
    Provides a default selection for filters
    """
    return _default_selection


def filters_form_options():
    """
    Returns the options to use for filterion
    """
    return map(lambda x: (x[1], x[0]), _filter_options)


def implicit_subject_pipeline_from_form(
        selected: List[str]
) -> Iterator[CandidateFilter]:
    for candidate in selected:
        for f in _filter_options:
            if f[1] == candidate:
                yield f[2]


def create_implicit_subject_pipeline(
        selected_filters: List[str]
) -> ImplicitSubjectPipeline:
    return ImplicitSubjectPipeline(
        missing_subject_detectors=[
            PassiveDetector(),
            ImperativeDetector(),
            GerundDetector(),
            NominalizedGerundWordlistDetector(),
            # NounVerbStemDetector(),
        ],
        candidate_filters=list(implicit_subject_pipeline_from_form(selected_filters)),
        missing_subject_inserter=ImplicitSubjectInserterImpl(
            inserters=[
                GerundInserter(_subject_annotation, _target_annotation),
                ImperativeInserter(_subject_annotation, _target_annotation),
                DefaultInserter(_subject_annotation, _target_annotation)
            ]
        ),
        candidate_extractor=DefinitionCandidateExtractor(),
        verbose=False
    )


def _subject_annotation(x: str, ctx: InsertionContext):
    return f'<span class="subject" data-highlight="0" data-insertion-id="{ctx.insertion_id}">{x}</span>'


def _target_annotation(x: str, ctx: InsertionContext):
    return f'<span class="target" data-highlight="0" data-insertion-id="{ctx.insertion_id}">{x}</span>'
