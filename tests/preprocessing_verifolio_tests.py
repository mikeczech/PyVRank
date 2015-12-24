from PyPRSVT.preprocessing import ranking, svcomp15
from PyPRSVT.preprocessing.unused import regression, classification
from PyPRSVT.preprocessing.verifolio import features as f


def verifolio_feature_extraction_test():
    file = 'static/sv-benchmarks/c/mixed-examples/data_structures_set_multi_proc_false-unreach-call_ground.i'
    metrics = f.extract_features(file)
    # Todo


def create_features_labels_df_test():
    cr = svcomp15.read_category('static/results-xml-raw', 'mixed-examples')
    sourcefiles = {i for k in cr.keys() for i in cr[k].index}
    features_df = f.create_feature_df(sourcefiles)
    # RPC dataset
    ranking_df = ranking.create_ranking_df(cr, svcomp15.compare_results)
    features_ranking_df = f.create_features_labels_df(features_df, ranking_df)

    # Learning via Utility Functions datasets
    score_dfdict = classification.create_benchmark_score_dfdict(cr, svcomp15.score)
    features_score_dfdict = {b: f.create_features_labels_df(features_df,
                                                            score_dfdict[b]) for b in score_dfdict.keys()}
    cputime_dfdict = regression.create_benchmark_cputime_dfdict(cr)
    features_cputime_dfdict = {b: f.create_features_labels_df(features_df,
                                                              cputime_dfdict[b]) for b in cputime_dfdict.keys()}
    # Todo
