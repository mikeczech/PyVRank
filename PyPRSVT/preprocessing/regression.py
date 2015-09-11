from PyPRSVT.preprocessing import utils

def create_benchmark_cputime_dfdict(results):
    """
    Todo
    :param results:
    :return:
    """
    label_val = lambda df: df['cputime']
    return utils.create_benchmark_labeling_dfdict(results,
                                              label_val,
                                              label_title='cputime')
