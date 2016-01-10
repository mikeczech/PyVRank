from itertools import combinations
import numpy as np
import logging
from PyPRSVT.preprocessing.ranking import Ranking
from sklearn import cross_validation, svm
from sklearn.grid_search import ParameterGrid
import random
import math
from tqdm import tqdm


rpc_logger = logging.getLogger('PyPRSVT.RPC_FEATURES')
rpc_logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
rpc_logger.addHandler(ch)


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

    def __init__(self, labels, distance_metric):
        self.bin_clfs = {}
        self.params = {}
        self.labels = labels
        self.distance_metric = distance_metric

    @staticmethod
    def _k_fold_cv(X, y, C, gamma, folds=10, shuffle=True):
        """
        K-fold cross-validation
        """
        scores = []
        loo = cross_validation.KFold(len(y), folds, shuffle=shuffle, random_state=random.randint(0, 100))
        for train_index, test_index in loo:
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            one_count = len([x for x in y_train if x == 1]);
            zero_count = len([x for x in y_train if x == 0]);
            balance = min(one_count, zero_count)/max(one_count, zero_count)
            if balance != 0:
                clf = svm.SVC(C=C, probability=True, kernel='rbf', gamma=gamma)
                clf.fit(X_train, y_train)
            else:
                print('Warning. Use of trivial classifier.')
                clf = TrivialClassifier([0, 1], 1)
            score = clf.score(X_test, y_test)
            scores.append(score)
        return np.mean(scores), np.std(scores)

    def fit(self, gamma_set, C_set, X_train, y_train):
        # Initialize base learners
        for (a, b) in tqdm(list(combinations(self.labels, 2)), nested=True):

            # Build binary labels
            y_bin = []
            for i, ranking in enumerate(y_train):
                assert ranking.part_of(a, b), 'Incomplete preference information detected'
                if ranking.greater_or_equal_than(a, b):
                    y_bin.append(1)
                else:
                    y_bin.append(0)
            y_bin = np.array(y_bin)

            one_count = len([x for x in y_bin if x == 1])
            zero_count = len([x for x in y_bin if x == 0])
            balance = min(one_count, zero_count)/max(one_count, zero_count)

            if balance != 0:
                # Perform grid search to find optimal parameters for each binary classification problem
                param_gid = {'gamma': gamma_set, 'C': C_set}
                max_mean = -math.inf
                for params in tqdm(list(ParameterGrid(param_gid)), nested=True):
                    mean, _ = self._k_fold_cv(X_train, y_bin, params['C'], params['gamma'])
                    if mean > max_mean:
                        max_mean = mean
                        self.params[a, b] = params

                # Use determined parameters to train base learner
                clf = svm.SVC(C=self.params[a, b]['C'], probability=True, kernel='rbf', gamma=self.params[a, b]['gamma'])
                clf.fit(X_train, y_bin)
                self.bin_clfs[a, b] = clf
            else:
                print('Warning. Use of trivial classifier.')
                self.bin_clfs[a, b] = TrivialClassifier([0, 1], 1)
                self.params[a, b] = {'C': C_set[0], 'gamma': gamma_set[0]}

        return self

    def __R_inner(self, X, i, j):
        return np.array([x[1] for x in self.bin_clfs[i, j].predict_proba(X)])

    def __R(self, X, i, j):
        if (i, j) in self.bin_clfs.keys():
            return self.__R_inner(X, i, j)
        else:
            return 1 - self.__R_inner(X, j, i)

    def predict(self, X, y):
        # Compute scores
        scores = {}
        for l in self.labels:
            scores[l] = sum([self.__R(X, l, ll) for ll in self.labels if ll != l])

        # Build rankings from scores
        ret = []
        for i in range(len(y)):
            r = sorted([l for l in self.labels], key=lambda l: scores[l][i])
            ret.append(Ranking(r))
        return ret

    def score(self, X, y):
        correlations = []
        for rs, rt in zip(self.predict(X, y), y):
            c = self.distance_metric.compute(rs, rt)
            correlations.append(c)
        return np.mean(correlations)


