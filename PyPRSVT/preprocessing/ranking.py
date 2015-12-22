"""
Python module for preprocessing software verification competition results to solve ranking problems
"""
import pandas as pd


class Ranking(object):

    def __init__(self, ranking):
        self.ranking = ranking

    def part_of(self, *tools):
        return all([t in self.ranking for t in tools])

    def loc(self, tool):
        return self.ranking.index(tool)

    def greater_or_equal_than(self, tool_a, tool_b):
        return self.loc(tool_a) > self.loc(tool_b)

    def __str__(self):
        return self.ranking.__str__()


def create_ranking_df(results, compare_results):
    """
    Todo
    :param results:
    :param compare_results:
    :return:
    """
    # Initialize new data frame
    df = pd.concat(results, axis=1)
    # rows with na values give us not information, so drop them.
    df.dropna(inplace=True)
    ranking_df = pd.DataFrame(columns=['ranking', 'property_type'])
    ranking_df.index.name = 'sourcefile'
    tools = results.keys()

    # Compute rankings from results
    for (sourcefile, results_df) in df.iterrows():
        geq_count = {t: 0 for t in tools}
        for t in tools:
            for tt in tools:
                if t != tt:
                    c = compare_results(results_df[t], results_df[tt])
                    if c >= 0:
                        geq_count[t] += 1
        ranking = sorted([t for t in tools], key=lambda t: geq_count[t])
        ranking_df.set_value(sourcefile, 'ranking', ranking)
        ranking_df.set_value(sourcefile, 'property_type', results_df[t]['property_type'])

    return ranking_df, tools
