"""
Distance Metrics
"""

class SpearmansRankCorrelation(object):

    def __init__(self, labels):
        self.k = len(labels)
        self.labels = labels

    def __d(self, ranking_a, ranking_b):
        return sum([(ranking_a.loc(l) - ranking_b.loc(l))**2 for l in self.labels])

    def compute(self, ranking_a, ranking_b):
        return 1 - (6 * self.__d(ranking_a, ranking_b) / (self.k * (self.k**2 - 1)))


# Todo
class KendallTau(object):
    pass
