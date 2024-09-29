from dotenv import load_dotenv

from ImplicitSubjectPipeline import ImplicitSubjectPipeline
from implicit_actor.candidate_extraction.ComposedCandidateExtractor import ComposedCandidateExtractor
from implicit_actor.candidate_extraction.DefinitionCandidateExtractor import DefinitionCandidateExtractor
from implicit_actor.candidate_filtering.SynsetFilter import SynsetFilter
from implicit_actor.candidate_filtering.VerbLinkFilter import VerbLinkFilter
from src.implicit_actor.candidate_filtering.DependentOfSameSentenceFilter import DependentOfSameSentenceFilter
from src.implicit_actor.candidate_filtering.ImperativeFilter import ImperativeFilter
from src.implicit_actor.candidate_filtering.PartOfSpeechFilter import PartOfSpeechFilter
from src.implicit_actor.candidate_filtering.PerplexityFilter import PerplexityFilter
from src.implicit_actor.missing_subject_detection.GerundDetector import GerundDetector
from src.implicit_actor.missing_subject_detection.NominalizedGerundWordlistDetector import \
    NominalizedGerundWordlistDetector
from src.implicit_actor.missing_subject_detection.PassiveDetector import PassiveDetector

load_dotenv()


def main():
    """
    Creates an ImplicitSubjectPipeline and runs it against the gold standard.
    """

    # setup the used pipeline here
    pipeline = ImplicitSubjectPipeline(
        missing_subject_detectors=[
            PassiveDetector(),
            GerundDetector(),
            NominalizedGerundWordlistDetector(),
        ],
        # missing_subject_filters=[MissingSubjectDetectionAuxFilter()],
        candidate_filters=[
            ImperativeFilter(),
            DependentOfSameSentenceFilter(),
            PartOfSpeechFilter(),
            VerbLinkFilter(
                add_preamble_verbs=False,
            ),
            SynsetFilter(),
            PerplexityFilter(max_returned=3, perplexity_buffer=1.3),
        ],
        candidate_extractor=ComposedCandidateExtractor([
            DefinitionCandidateExtractor(),
            # PreambleExtractor(),
        ]),
        verbose=True
    )

    # The pipeline can be used by simply providing a text to be inspected for implicit subjects and
    # a context from which candidates should be taken. This is obviously a dummy example and more context would be
    # provided in a real world use case.
    inspected = f"every reasonable step must be taken to ensure that personal data that are inaccurate, having regard to the purposes for which they are processed, are erased or rectified without delay (‘accuracy’).;"
    context = """
    [...]
    (2) ‘processing’ means any operation or set of operations which is performed on personal data or on sets of personal data, whether or not by automated means, such as collection, recording, organisation, structuring, storage, adaptation or alteration, retrieval, consultation, use, disclosure by transmission, dissemination or otherwise making available, alignment or combination, restriction, erasure or destruction;
    [...]
    (8) ‘processor’ means a natural or legal person, public authority, agency or other body which processes personal data on behalf of the controller;
    [...]
    {inspected}
    """
    result = pipeline.apply(inspected, context)
    print(result)


if __name__ == "__main__":
    main()
