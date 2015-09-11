import pandas as pd
import ast
from itertools import combinations

class RPC(object):

    def __init__(self, labels, base, base_options):
        bin_clfs = {}
        for rel in combinations(labels, 2):
            bin_clfs['{0} >= {1}'.format(rel[0], rel[1])] \
                = base(base_options)
        self.bin_clfs = bin_clfs
        self.base = base

    def __reverse_rel(self, rel):
        rel_parts = [r.strip() for r in rel.split('>=')]
        return '{0} >= {1}'.format(rel_parts[1], rel_parts[0])

    def fit(self, X, y):
        """
        Todo
        :param X:
        :param y:
        :return:
        """
        for rel in self.bin_clfs.keys():
            y_rel = []
            for rel_set in y:
                rel_set = ast.literal_eval(rel_set)
                if rel in rel_set:
                    y_rel.append(1)
                elif self.__reverse_rel(rel):
                    y_rel.append(0)
                else:
                    y_rel.append(None)


    def predict(self):
        pass
