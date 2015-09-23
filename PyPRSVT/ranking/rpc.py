import pandas as pd
from itertools import combinations
from collections import namedtuple
from ast import literal_eval
import numpy as np
import logging
from PyPRSVT.preprocessing.ranking import Ranking


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


class RPC(object):

    def __init__(self, labels, base_learner, **bl_options):
        self.base_learner = base_learner
        self.bl_options = bl_options
        self.fitted = False
        self.bin_clfs = {}
        self.labels = labels

    def fit(self, X_df, y_sr):
        """
        Todo
        :param X:
        :param y:
        :return:
        """
        # Initialize base learners
        for (a, b) in combinations(self.labels, 2):
            self.bin_clfs[Geq(a, b)] \
                = self.base_learner(**self.bl_options) if self.bl_options else self.base_learner()

        rpc_logger.info('Decomposed label ranking problem into {} binary classification problems'
                        .format(len(self.bin_clfs.keys())))

        # Create multiple binary classification problems from data
        for rel in self.bin_clfs.keys():

            rpc_logger.info('Solving binary classification problem for {} > {}'
                            .format(rel.a, rel.b))

            y_rel_df = pd.DataFrame(columns=['y'])
            for (p, r) in y_sr.iteritems():
                ranking = Ranking(literal_eval(r))
                if not ranking.part_of(rel.a, rel.b):
                    y_rel_df.loc[p] = np.NaN
                elif ranking.greater_or_equal_than(rel.a, rel.b):
                    y_rel_df.loc[p] = 1
                else:
                    y_rel_df.loc[p] = 0

            # Only use rows where the label is not NaN
            rel_df = pd.concat([X_df, y_rel_df], axis=1)
            rel_df.dropna(inplace=True)

            # Solve binary ML problem with base learner
            X = rel_df.drop('y', 1).values
            y = rel_df['y'].values

            one_count = len([i for i in y if i == 1])
            zero_count = len([i for i in y if i == 0])

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
                self.bin_clfs[rel].fit(X, y)
                scores = self.bin_clfs[rel].score(X, y)
                rpc_logger.info('Accuracy on training data: {}, class imbalance: {}'
                                .format(scores, one_count / zero_count))

        self.fitted = True

    def __R(self, X, i, j):
        if Geq(i, j) in self.bin_clfs.keys():
            return np.array([x[1] for x in self.bin_clfs[Geq(i, j)].predict_proba(X)])
        else:
            return 1 - np.array([x[1] for x in self.bin_clfs[Geq(j, i)].predict_proba(X)])

    def predict(self, X_df):
        """
        Todo
        :param labels:
        :param X:
        :return:
        """
        X = X_df.values
        # Compute scores
        scores = {}
        for l in self.labels:
            scores[l] = sum([self.__R(X, l, ll) for ll in self.labels if ll != l])

        # Build rankings from scores
        return [Ranking(sorted([l for l in self.labels], key=lambda l: scores[l][i])) for i, _ in enumerate(X)]

    def score(self, X_df, y_sr, distance_metric):
        """
        Todo
        :param X_df:
        :param y:
        :return:
        """
        y = [Ranking(literal_eval(y)) for y in y_sr.tolist()]
        correlations = []
        for rs, rt in zip(self.predict(X_df), y):
            c = distance_metric.compute(rs, rt)
            correlations.append(c)
        return np.mean(correlations)


