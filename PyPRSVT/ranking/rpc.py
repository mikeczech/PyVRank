import pandas as pd
import ast
from itertools import combinations
import numpy as np

class RPC(object):

    def __init__(self, labels, base, base_options):
        bin_clfs = {}
        for rel in combinations(labels, 2):
            bin_clfs['{0} >= {1}'.format(rel[0], rel[1])] \
                = base(base_options)
        self.bin_clfs = bin_clfs
        self.base = base

    def __reverse_rel(rel):
        rel_parts = [r.strip() for r in rel.split('>=')]
        return '{0} >= {1}'.format(rel_parts[1], rel_parts[0])

    def fit(self, X_df, y_df):
        """
        Todo
        :param X:
        :param y:
        :return:
        """
        for rel in self.bin_clfs.keys():
            y_rel_df = pd.DataFrame(columns=['y'])
            for row in y_df.iterrows():
                p, rel_set = row
                rel_set = ast.literal_eval(rel_set)
                if rel in rel_set:
                    y_rel_df.loc[p] = 1
                elif RPC.__reverse_rel(rel):
                    y_rel_df.loc[p] = 0
                else:
                    y_rel_df.loc[p] = np.NaN
            rel_df = pd.concat([X_df, y_rel_df], axis=1)
            rel_df.dropna()
            # Todo solve ML problem with base learner



    def predict(self):
        pass
