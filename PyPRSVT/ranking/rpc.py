import pandas as pd
import ast
from itertools import combinations
import numpy as np
from PyPRSVT.preprocessing.ranking import Geq


class RPC(object):

    def __init__(self, base_learner, **bl_options):
        self.base_learner = base_learner
        self.bl_options = bl_options
        self.fitted = False
        self.bin_clfs = {}

    def __reverse_rel(self, rel):
        return Geq(rel.b, rel.a)

    def fit(self, labels, X_df, y_sr):
        """
        Todo
        :param X:
        :param y:
        :return:
        """
        # Initialize base learners
        for rel in combinations(labels, 2):
            self.bin_clfs[Geq(rel[0], rel[1])] \
                = self.base_learner(**self.bl_options) if self.bl_options else self.base_learner()

        # Create multiple binary classification problems from data
        for rel in self.bin_clfs.keys():
            y_rel_df = pd.DataFrame(columns=['y'])
            for (p, rel_set) in y_sr.iteritems():
                # Todo: Find better solution as eval
                prefs = eval(rel_set)
                if rel in prefs:
                    y_rel_df.loc[p] = 1
                elif self.__reverse_rel(rel):
                    y_rel_df.loc[p] = 0
                else:
                    y_rel_df.loc[p] = np.NaN

            # Only use rows where the label is not NaN
            rel_df = pd.concat([X_df, y_rel_df], axis=1)
            rel_df.dropna(inplace=True)

            # Solve binary ML problem with base learner
            X = rel_df.drop('y', 1).values
            y = rel_df['y'].values
            self.bin_clfs[rel].fit(X, y)

        self.fitted = True

    def __R(self, X, i, j):
        if Geq(i, j) in self.bin_clfs.keys():
            return np.array(self.bin_clfs[Geq(i, j)].predict_proba(X))
        else:
            return 1 - np.array(self.bin_clfs[Geq(j, i)].predict_proba(X))

    def predict(self, labels, X):
        """
        Todo
        :param labels:
        :param X:
        :return:
        """
        # Compute scores
        scores = {}
        for l in labels:
            scores[l] = sum([self.__R(X, l, ll) for ll in labels if ll != l])

        # Build rankings from scores
        return [sorted([l for l in labels], key=lambda l: scores[l][i]) for i, _ in enumerate(X)]
