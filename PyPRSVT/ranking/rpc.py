from itertools import combinations
from collections import namedtuple
import numpy as np
import logging
from PyPRSVT.preprocessing.ranking import Ranking
from sklearn import cross_validation, svm
from sklearn.grid_search import ParameterGrid
import random
import math


rpc_logger = logging.getLogger('PyPRSVT.RPC')
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
    def _k_fold_cv_gram(gram_matrix, y, C, folds=10, shuffle=True):
        """
        K-fold cross-validation
        """
        scores = []
        loo = cross_validation.KFold(len(y), folds, shuffle=shuffle, random_state=random.randint(0, 100))
        for train_index, test_index in loo:
            X_train, X_test = gram_matrix[train_index][:, train_index], gram_matrix[test_index][:, train_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = svm.SVC(C=C, probability=True, kernel='precomputed')
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
        return clf, np.mean(scores), np.std(scores)

    def gram_fit(self, h_set, D_set, C_set, gram_paths, train_index, y):
        # Initialize base learners
        for (a, b) in combinations(self.labels, 2):
            # Perform grid search to find optimal parameters for each binary classification problem
            param_gid = {'h': h_set, 'D': D_set, 'C': C_set}
            min_mean = math.inf
            for params in ParameterGrid(param_gid):
                gram_matrix_train = np.load(gram_paths[params['h'], params['D']])[train_index][:,train_index]
                y_bin = []
                for i, ranking in enumerate(y):
                    assert ranking.part_of(a, b), 'Incomplete preference information detected'
                    if ranking.greater_or_equal_than(a, b):
                        y_bin.append(1)
                    else:
                        y_bin.append(0)

                clf, mean, std = self._k_fold_cv_gram(gram_matrix_train, y_bin, params['C'])
                if mean < min_mean:
                    min_mean = mean
                    self.bin_clfs[a, b] = clf
                    self.params[a, b] = params

        rpc_logger.info('Decomposed label ranking problem into {} binary classification problems'
                        .format(len(self.bin_clfs.keys())))
        return self

    def __R(self, gram_paths, test_index, train_index, i, j):
        if (i, j) in self.bin_clfs.keys():
            K = np.load(gram_paths[self.params[i, j]['h'], self.params[i, j]['D']])
            K_test = K[test_index][:, train_index]
            return np.array([x[1] for x in self.bin_clfs[i, j].predict_proba(K_test)])
        else:
            K = np.load(gram_paths[self.params[j, i]['h'], self.params[j, i]['D']])
            K_test = K[test_index][:, train_index]
            return 1 - np.array([x[1] for x in self.bin_clfs[j, i].predict_proba(K_test)])

    def predict(self, gram_paths, test_index, train_index, y_test):
        # Compute scores
        scores = {}
        for l in self.labels:
            scores[l] = sum([self.__R(gram_paths, test_index, train_index, l, ll) for ll in self.labels if ll != l])

        # Build rankings from scores
        return [Ranking(sorted([l for l in self.labels], key=lambda l: scores[l][i])) for i in range(len(y_test))]

    def score(self, gram_paths, test_index, train_index, y_test):
        correlations = []
        for rs, rt in zip(self.predict(gram_paths, test_index, train_index, y_test), y_test):
            c = self.distance_metric.compute(rs, rt)
            print("RS: " + str(rs))
            print("RT: " + str(rt))
            correlations.append(c)
        return np.mean(correlations)


