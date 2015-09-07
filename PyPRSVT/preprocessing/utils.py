import pandas as pd

def create_benchmark_labeling_df(results, label_val, label_title='label'):
    """
    Todo
    :param results:
    :param label_val:
    :param label_title:
    :return:
    """
    ret = {}
    for benchmark in results.keys():
        ret_df = pd.DataFrame(columns=[label_title])
        ret_df.index.name = 'sourcefile'
        for row in results[benchmark].iterrows():
            sourcefile, df = row
            ret_df.set_value(sourcefile, label_title, label_val(df))
            ret[benchmark] = ret_df
    return ret


def derive_total_order_from_relations(relations):
    """
    Todo
    :param relations:
    :return:
    """
    pass
