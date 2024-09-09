from abc import ABC, abstractmethod
from typing import List

from implicit_actor.missing_subject_detection.ImplicitSubjectDetection import ImplicitSubjectDetection


class MissingSubjectDetectionFilter(ABC):
    @abstractmethod
    def filter(self, detections: List[ImplicitSubjectDetection]) -> List[ImplicitSubjectDetection]:
        """
        Checks if a subject type is accepted by this.
        """
        raise NotImplementedError()
