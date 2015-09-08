from PyPRSVT.preprocessing import utils

def create_benchmark_cputime_df(results):
    """
    Todo
    :param results:
    :return:
    """
    label_val = lambda df: df['cputime']
    return utils.create_benchmark_labeling_df(results,
                                              label_val,
                                              label_title='cputime')
