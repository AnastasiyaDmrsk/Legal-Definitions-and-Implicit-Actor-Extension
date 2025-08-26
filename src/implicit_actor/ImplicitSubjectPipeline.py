import warnings
from functools import reduce
from typing import List, Optional, Tuple, Dict, Iterable

import spacy
from spacy.tokens import Token, Span

from implicit_actor.candidate_extraction.CandidateActor import CandidateActor
from implicit_actor.candidate_extraction.SubjectObjectCandidateExtractor import SubjectObjectCandidateExtractor
from implicit_actor.candidate_filtering.FilterContext import FilterContext
from implicit_actor.missing_subject_detection.filters.missing_subject_detection_filter import \
    MissingSubjectDetectionFilter
from implicit_actor.util import get_noun_chunk
from implicit_actor.candidate_extraction.CandidateExtractor import CandidateExtractor
from implicit_actor.candidate_filtering.CandidateFilter import CandidateFilter
from implicit_actor.insertion.ImplicitSubjectInserter import ImplicitSubjectInserter
from implicit_actor.insertion.ImplicitSubjectInserterImpl import ImplicitSubjectInserterImpl
from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection
from implicit_actor.missing_subject_detection.ImplicitSubjectDetector import ImplicitSubjectDetector


class ImplicitSubjectPipeline:
    """
    Manages the different components of the implicit subject process.
    """

    def __init__(self,
                 missing_subject_detectors: List[ImplicitSubjectDetector],
                 candidate_filters: List[CandidateFilter],
                 candidate_extractor: CandidateExtractor = None,
                 missing_subject_inserter: ImplicitSubjectInserter = None,
                 missing_subject_filters: List[MissingSubjectDetectionFilter] = None,
                 verbose: bool = False,
                 fast: bool = False):
        """
        Creates an ImplicitSubjectPipeline.

        :param missing_subject_detectors: A list of ImplicitSubjectDetectors to be used.
        :param candidate_extractor: The CandidateExtractor to be used.
        :param candidate_filters: A list of CandidateFilters to be used.
        :param missing_subject_inserter: The ImplicitSubjectInserter to be used.
        :param verbose: If debug infromation should be printed.
        :param fast: If True uses en_core_web_sm instead of en_core_web_trf for spaCy
        """
        self._candidate_filters = candidate_filters
        self._missing_subject_detectors = missing_subject_detectors
        self._missing_subject_filters: List[MissingSubjectDetectionFilter] = missing_subject_filters or []
        self._candidate_extractor = candidate_extractor or SubjectObjectCandidateExtractor()
        self._missing_subject_inserter = missing_subject_inserter or ImplicitSubjectInserterImpl()
        self._verbose = verbose
        self._nlp = spacy.load("en_core_web_sm" if fast else "en_core_web_trf")
        self._last_detections = None
        self._last_selected_candidates: Optional[List[CandidateActor]] = []
        self._last_selected_candidate_with_target: Optional[
            List[Tuple[ImplicitSubjectDetection, CandidateActor]]] = []
        self._last_log = None
        self._last_initial_candidates: List[CandidateActor] = []
        self._last_candidates_before_tie_break: Dict[str, List[str]] = dict()
        self._last_candidates_before_tie_break_root_only: Dict[str, List[str]] = dict()
        self._last_active_filters: List[str] = list()

    def last_active_filters(self) -> List[str]:
        return self._last_active_filters

    def last_initial_candidates(self) -> List[CandidateActor]:
        return self._last_initial_candidates

    def last_selected_candidates_before_tie_break(self) -> Dict[str, List[str]]:
        return self._last_candidates_before_tie_break

    def last_selected_candidates_before_tie_break_root_only(self) -> Dict[str, List[str]]:
        return self._last_candidates_before_tie_break_root_only

    def last_selected_candidates(self) -> Optional[List[CandidateActor]]:
        """
        Returns the last selected candidates. Useful for evaluation.
        """
        return self._last_selected_candidates

    def last_selected_candidate_with_target(self) -> Optional[List[Tuple[ImplicitSubjectDetection, CandidateActor]]]:
        return self._last_selected_candidate_with_target

    def last_detections(self) -> List[ImplicitSubjectDetection]:
        """
        Returns the last implicit subject detections. Useful for evaluation.
        """
        return self._last_detections

    def last_filter_log(self):
        """
        Returns a log of the candidate filter mechanism. Useful for evaluation.
        """
        return self._last_log

    def _debug(self, *msg, **kwargs):
        if self._verbose:
            print(*msg, **kwargs)

    def _apply_candidate_filters(self, targets: List[ImplicitSubjectDetection], candidates: List[CandidateActor],
                                 context: Span) -> Iterable[CandidateActor]:
        """
        Applies the candidate filters to the candidates in a chain operation.
        """

        def _apply_filter(acc: List[CandidateActor], f: CandidateFilter):
            prev_len = len(acc)
            if prev_len == 1:
                self._debug(f"Short circuiting  {f.__class__.__name__} with {acc}.")
                return acc

            intermediate_result = f.filter(target, acc, FilterContext(
                context=context,
                initial_candidates=self._last_initial_candidates
            ))
            if len(intermediate_result) != len(acc):
                self._last_active_filters.append(f.__class__.__name__)
            self._debug(
                f"Applied {f.__class__.__name__}, filtered {100 - 100 * len(intermediate_result) / prev_len :.2f}% of "
                f"candidates and returned {intermediate_result}")
            return intermediate_result

        def _logged_apply(
                acc: Tuple[List[Tuple[Optional[CandidateFilter], List[CandidateActor]]], List[CandidateActor]],
                f: CandidateFilter):
            _log, _acc = acc
            _res = _apply_filter(_acc, f)
            _log.append((f, _res))
            return _log, _res

        for target in targets:
            self._debug(f"Searching for candidate for target {target}")
            log, res = reduce(_logged_apply, self._candidate_filters, ([(None, candidates)], candidates))
            tok = res[0] if res else candidates[0]
            self._last_candidates_before_tie_break[target.token.text] = list(
                map(lambda x: get_noun_chunk(x.token).text, res)
            )
            self._last_candidates_before_tie_break_root_only[target.token.text.lower()] = list(
                map(lambda x: x.token.text, res)
            )
            yield tok, log

    def apply(self, inspected_text: str, context: Optional[str] = None) -> str:
        """
        Runs the pipeline using the components defined in the constructor.
        """
        if not context:
            context = inspected_text

        self._last_candidates_before_tie_break = dict()
        self._last_candidates_before_tie_break_root_only = dict()
        self._last_selected_candidates = []
        self._last_active_filters = []
        self._last_initial_candidates = []

        # Generally we assume the inspected text to be part of the context. If we can use a single doc
        # which is necessary for some filters (ProximityFilter) to work correctly. Otherwise, we create two docs.
        if inspected_text not in context:
            # warnings.warn(f"Could not find inspected text in context. Using separate docs. ('{inspected_text}')")
            inspected_text_span = self._nlp(inspected_text)[:]
            context_doc = self._nlp(context)
        else:
            context_doc = self._nlp(context)
            ind = context.index(inspected_text)
            inspected_text_span = context_doc.char_span(ind, ind + len(inspected_text))
            if str(inspected_text_span) != inspected_text:
                warnings.warn("Could not extract the inspected text from the context. Will skip inspection")
                self._last_detections = dict()
                self._last_log = []
                self._last_selected_candidates = []
                self._last_selected_candidate_with_target = []
                return inspected_text
            # assert inspected_text == str(inspected_text_span), "Could not extract the inspected text from the context."

        targets = dict()
        for detector in reversed(self._missing_subject_detectors):
            # (ImplicitSubjectDetection are not hashable, so we use the hashable predicate token as a unique key)
            # Detectors at the front of the list take precedence over those at the back.
            targets.update({
                x.token: x for x in detector.detect(inspected_text_span)
            })
        targets = list(targets.values())

        for detection_filter in self._missing_subject_filters:
            targets = detection_filter.filter(targets)

        self._last_detections = targets

        self._debug("Detected the following targets:\n", targets, sep="")
        self._debug("-----")

        candidates = self._candidate_extractor.extract(context_doc)
        self._last_initial_candidates = list(
            candidates
        )

        self._debug("Extracted the following candidates:\n", candidates, sep="")
        self._debug("-----")

        filter_res = self._apply_candidate_filters(targets, candidates, context_doc[:])
        log = []
        subjects_for_insertion = []
        for t, l in filter_res:
            log.append(l)
            subjects_for_insertion.append(t)

        self._last_log = list(zip(targets, log))

        self._debug("Picked the following subjects for insertion:\n", *zip(targets, subjects_for_insertion), sep="")
        self._last_selected_candidates = subjects_for_insertion
        self._last_selected_candidate_with_target = list(zip(targets, subjects_for_insertion))

        resolved = self._missing_subject_inserter.insert(inspected_text_span, targets, subjects_for_insertion)

        return resolved
