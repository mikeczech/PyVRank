import pandas as pd
import ast
from itertools import combinations
import numpy as np


class RPC(object):

    def __init__(self, base_learner, **bl_options):
        self.base_learner = base_learner
        self.bl_options = bl_options
        self.fitted = False

    def __reverse_rel(rel):
        rel_parts = [r.strip() for r in rel.split('>=')]
        return '{0} >= {1}'.format(rel_parts[1], rel_parts[0])

    def fit(self, labels, X_df, y_sr):
        """
        Todo
        :param X:
        :param y:
        :return:
        """
        # Initialize base learners
        self.bin_clfs = {}
        for rel in combinations(labels, 2):
            self.bin_clfs['{0} >= {1}'.format(rel[0], rel[1])] \
                = self.base_learner(**self.bl_options) if self.bl_options else self.base_learner()

        # Create multiple binary classification problems from data
        for rel in self.bin_clfs.keys():
            y_rel_df = pd.DataFrame(columns=['y'])
            for (p, rel_set) in y_sr.iteritems():
                prefs = ast.literal_eval(rel_set)
                if rel in prefs:
                    y_rel_df.loc[p] = 1
                elif RPC.__reverse_rel(rel):
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


    def predict(self):
        pass
