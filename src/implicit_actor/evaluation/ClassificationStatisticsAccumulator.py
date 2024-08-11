import math


class ClassificationStatisticsAccumulator:
    """
    Accumulates stats
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def apply(self, acc: "ClassificationStatisticsAccumulator"):
        """
        Adds the stats from one accumulator to this one.
        """
        self.tp += acc.tp
        self.tn += acc.tn
        self.fp += acc.fp
        self.fn += acc.fn

    def precision(self) -> float:
        """
        Returns the current precision.
        """
        if self.tp + self.fp == 0:
            return math.nan
        return self.tp / (self.tp + self.fp)

    def recall(self) -> float:
        """
        Returns the current recall.
        """
        if self.tp + self.fn == 0:
            return math.nan
        return self.tp / (self.tp + self.fn)

    def __str__(self):
        return f"StatAcc(tp = {self.tp}, fp = {self.fp}, tn = {self.tn}, fn = {self.fn})"