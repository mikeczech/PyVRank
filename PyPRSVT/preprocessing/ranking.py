"""
Python module for preprocessing software verification competition results to solve ranking problems
"""
import pandas as pd
from itertools import combinations
from collections import namedtuple

Geq = namedtuple('GreaterOrEqualThan', 'a b')


def create_benchmark_ranking_df(results, compare_results):
    """
    Todo
    :param results:
    :param compare_results:
    :return:
    """
    benchmark_combinations = list(combinations(results.keys(), 2))
    df = pd.concat(results, axis=1)
    # rows with na values give us not information, so drop them.
    df.dropna(inplace=True)
    ret_df = pd.DataFrame(columns=['ranking'])
    ret_df.index.name = 'sourcefile'
    for row in df.iterrows():
        preferences = []
        sourcefile, results_df = row
        for pair in benchmark_combinations:
            tool_a, tool_b = pair
            c = compare_results(results_df[tool_a], results_df[tool_b])
            if c == 1 or c == 0:
                preferences.append(Geq(tool_a, tool_b))
        ret_df.set_value(sourcefile, 'ranking', preferences)
    return ret_df, results.keys()

