from itertools import combinations
from collections import namedtuple
import numpy as np
import logging
from PyPRSVT.preprocessing.ranking import Ranking
from sklearn.base import BaseEstimator
import sklearn


rpc_logger = logging.getLogger('PyPRSVT.RPC')
rpc_logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
rpc_logger.addHandler(ch)

# GreaterOrEqualThan
Geq = namedtuple('Geq', 'a b')


class TrivialClassifier(object):

    def __init__(self, classes, prediction_index):
        self.classes = classes
        self.prediction_index = prediction_index

    def predict(self, X):
        return [self.classes[self.prediction_index] for _ in X]

    def predict_proba(self, X):
        return [[1.0 if i == self.prediction_index else 0.0 for i, _ in enumerate(self.classes)] for _ in X]

    def score(self, X, y):
        return np.mean([1.0 if i != self.classes[self.prediction_index] else 0.0 for i in y])


class RPC(BaseEstimator):

    def __init__(self, labels, distance_metric, base_learner):
        self.base_learner = base_learner
        self.fitted = False
        self.bin_clfs = {}
        self.labels = labels
        self.distance_metric = distance_metric

    def fit(self, X, y):
        """
        Todo
        :param X:
        :param y:
        :return:
        """
        # Initialize base learners
        for (a, b) in combinations(self.labels, 2):
            self.bin_clfs[Geq(a, b)] \
                = sklearn.base.clone(self.base_learner)

        rpc_logger.info('Decomposed label ranking problem into {} binary classification problems'
                        .format(len(self.bin_clfs.keys())))

        assert len(X) == len(y)

        # Create multiple binary classification problems from data
        for rel in self.bin_clfs.keys():

            rpc_logger.info('Solving binary classification problem for {} > {}'
                            .format(rel.a, rel.b))

            y_rel = []
            y_nan_indices = []
            for i, ranking in enumerate(y):
                if not ranking.part_of(rel.a, rel.b):
                    y_nan_indices.append(i)
                elif ranking.greater_or_equal_than(rel.a, rel.b):
                    y_rel.append(1)
                else:
                    y_rel.append(0)

            if self.base_learner.get_params()['kernel'] == 'precomputed':
                X_rel = np.delete(X, y_nan_indices, 1)
                X_rel = np.delete(X_rel, y_nan_indices, 0)
            else:
                X_rel = np.array([x for i, x in enumerate(X) if i not in y_nan_indices])

            assert len(X_rel) == len(y_rel)

            one_count = len([i for i in y_rel if i == 1])
            zero_count = len([i for i in y_rel if i == 0])
            # Todo How can we handle such problems?
            if one_count == 0 or zero_count == 0:
                rpc_logger.warning('''
                Only one class in binary classification problem detected. You need better data!
                Replacing base learner with trivial classifier...
                (Note that this might negatively influence generalization performance!)''')

            if one_count == 0:
                self.bin_clfs[rel] = TrivialClassifier([0, 1], 0)
            elif zero_count == 0:
                self.bin_clfs[rel] = TrivialClassifier([0, 1], 1)
            else:
                self.bin_clfs[rel].fit(X_rel, y_rel)
                scores = self.bin_clfs[rel].score(X_rel, y_rel)
                rpc_logger.info('Accuracy on training data: {}, class imbalance: {}'
                                .format(scores, one_count / zero_count))

        return self

    def __R(self, X, i, j):
        if Geq(i, j) in self.bin_clfs.keys():
            return np.array([x[1] for x in self.bin_clfs[Geq(i, j)].predict_proba(X)])
        else:
            return 1 - np.array([x[1] for x in self.bin_clfs[Geq(j, i)].predict_proba(X)])

    def predict(self, X):
        """
        Todo
        :param labels:
        :param X:
        :return:
        """
        # Compute scores
        scores = {}
        for l in self.labels:
            scores[l] = sum([self.__R(X, l, ll) for ll in self.labels if ll != l])

        # Build rankings from scores
        return [Ranking(sorted([l for l in self.labels], key=lambda l: scores[l][i])) for i, _ in enumerate(X)]

    def score(self, X, y):
        """
        Todo
        :param X_df:
        :param y:
        :return:
        """
        correlations = []
        for rs, rt in zip(self.predict(X), y):
            c = self.distance_metric.compute(rs, rt)
            print("RS: " + str(rs))
            print("RT: " + str(rt))
            correlations.append(c)
        return np.mean(correlations)


