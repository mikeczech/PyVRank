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


def derive_total_benchmark_order(results, compare_results):
    """
    Todo
    :param relations:
    :return:
    """
    higher_than_counter = {k: 0 for k in results.keys()}
    for i in results.keys():
        for j in results.keys():
            if i != j:
                if compare_results(results[i], results[j]) in [0, 1]:
                    higher_than_counter[i] += 1
    return sorted(results.keys(), reversed=True, key=lambda x: higher_than_counter[x])
