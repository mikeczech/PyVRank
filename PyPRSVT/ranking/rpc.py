from itertools import combinations
import numpy as np
import logging
from PyPRSVT.preprocessing.ranking import Ranking
from sklearn import cross_validation, svm
from sklearn.grid_search import ParameterGrid
import random
import math
from tqdm import tqdm


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
        return np.mean(scores), np.std(scores)

    def gram_fit(self, h_set, D_set, C_set, gram_paths, train_index, y_train):
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

            # one_count = len([x for x in y_bin if x == 1]);
            # zero_count = len([x for x in y_bin if x == 0]);
            # print('Balance {}'.format(min(one_count, zero_count)/max(one_count, zero_count)))

            # Perform grid search to find optimal parameters for each binary classification problem
            param_gid = {'h': h_set, 'D': D_set, 'C': C_set}
            min_mean = math.inf
            for params in tqdm(list(ParameterGrid(param_gid)), nested=True):
                gram_matrix_train = np.load(gram_paths[params['h'], params['D']])[train_index][:, train_index]
                mean, _ = self._k_fold_cv_gram(gram_matrix_train, y_bin, params['C'])
                if mean < min_mean:
                    min_mean = mean
                    self.params[a, b] = params

            # Use determined parameters to train base learner
            gram_matrix_best = np.load(gram_paths[self.params[a, b]['h'], self.params[a, b]['D']])[train_index][:, train_index]
            clf = svm.SVC(C=self.params[a, b]['C'], probability=True, kernel='precomputed')
            clf.fit(gram_matrix_best, y_bin)
            self.bin_clfs[a, b] = clf

        return self

    def __R_inner(self, gram_paths, test_index, train_index, i, j):
        K = np.load(gram_paths[self.params[i, j]['h'], self.params[i, j]['D']])
        K_test = K[test_index][:, train_index]
        return np.array([x[1] for x in self.bin_clfs[i, j].predict_proba(K_test)])

    def __R(self, gram_paths, test_index, train_index, i, j):
        if (i, j) in self.bin_clfs.keys():
            return self.__R_inner(gram_paths, test_index, train_index, i, j)
        else:
            return 1 - self.__R_inner(gram_paths, test_index, train_index, j, i)

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
            correlations.append(c)
        return np.mean(correlations)


