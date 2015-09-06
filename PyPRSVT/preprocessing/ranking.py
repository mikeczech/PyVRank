"""
Python module for preprocessing software verification competition results to solve ranking problems
"""
import pandas as pd
from itertools import permutations

def create_tool_ranking(results, compare_results):
    tools = list(results.keys())
    tool_permutations = list(permutations(tools, 2))
    ret = {}
    df = pd.concat(results, axis=1)
    # rows with na values give us not information
    nona_df = df.dropna()
    for row in nona_df.iterrows():
        preferences = []
        sourcefile, results_df = row
        for pair in tool_permutations:
            tool_a, tool_b = pair
            c = compare_results(results_df[tool_a], results_df[tool_b])
            if c == 1 or c == 0:
                preferences.append('{0} >= {1}'.format(tool_a, tool_b))
        ret[sourcefile] = preferences
    return ret
