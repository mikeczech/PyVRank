import pandas as pd
from PyPRSVT.preprocessing import utils, ranking

def create_benchmark_score_df(results, score):
    """
    Todo
    :param results:
    :param score:
    :return:
    """
    label_val = lambda df: score(df['status'], df['expected_status'])
    return utils.create_benchmark_labeling_df(results,
                                              label_val,
                                              label_title='score')


def create_benchmark_best_tool_df(results, compare_results):
    pass