import pandas as pd
from PyPRSVT.preprocessing import utils

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


def create_benchmark_ranking_df(results, compare_results):
    """
    Todo
    :param results:
    :param compare_results:
    :return:
    """
    df = pd.concat(results, axis=1)
    # rows with na values give us not information, so drop them.
    df.dropna(inplace=True)
    ret_df = pd.DataFrame(columns=['best_tool'])
    ret_df.index.name = 'sourcefile'
    for row in df.iterrows():
        sourcefile, results_df = row
        ret_df.set_value(sourcefile, 'best_tool', utils.derive_total_benchmark_order(results,
                                                                                     sourcefile,
                                                                                     compare_results)[0])
    return ret_df
