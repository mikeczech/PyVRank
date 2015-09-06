"""
Python module for preprocessing software verification competition results to solve ranking problems
"""

def create_tool_ranking_df(results, ranking_function):
    tools = list(results.keys())
