from collections import defaultdict
from typing import List


class FilterFailAccumulator:
    """
    Accumulator for determining where in the filter chain the correct candidate went missing.
    """

    def __init__(self):
        # for each filter, count how often it is at fault
        self._counts = defaultdict(int)
        self._correct = 0
        self._inspected = 0
        self._num_out = defaultdict(int)

    def counts(self):
        """
        Returns a count how often a candidate is lost per filter.
        """
        return self._counts

    def inspected(self):
        """
        Returns the number of inspected detections. (A detection is only inspected if it is both
        listed in the log and part of the gold standard)
        """
        return self._inspected

    def num_filtered(self):
        """
        Returns the total number of candidates output per filter. Can be used to calculate the average
        filter rate of each filter.
        """
        return self._num_out

    def correct(self):
        """
        Returns the number of candidates correctly
        """
        return self._correct

    def apply(self, filter_log, correct_targets: List[str], correct_candidates: List[str]):
        """
        Add a log entry to the accumulator.
        """
        target_to_candidates = {x.lower(): y for x, y in zip(correct_targets, correct_candidates)}

        for detection, log in filter_log:
            det_text = detection.token.text.lower()
            if det_text not in target_to_candidates:
                continue

            correct_candidate = target_to_candidates[det_text].lower()
            self._inspected += 1

            for f, l in log:
                self._num_out[f] += len(l)
                if correct_candidate not in {x.text.lower() for x in l}:
                    self._counts[f.__class__.__name__] += 1
                    break
            else:
                self._correct += 1

    def performance_str(self):
        """
        Returns a formatted string of the performance.
        """
        return f"{self.correct()}/{self.inspected()} ({self.correct() / self.inspected() * 100 :.2f}%)"