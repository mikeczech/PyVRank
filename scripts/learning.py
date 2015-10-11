from PyPRSVT.ranking import rpc, distance_metrics
from PyPRSVT.preprocessing.ranking import Ranking
from ast import literal_eval
import pandas as pd
from sklearn import svm, cross_validation
import argparse
import numpy as np
import networkx as nx
import random
import logging


def k_fold_cv(gram_matrix, labels, tools, folds=2, shuffle=True):
    """
    K-fold cross-validation
    """
    scores = []
    loo = cross_validation.KFold(len(labels), folds, shuffle=shuffle, random_state=random.randint(0,100))
    spearman = distance_metrics.SpearmansRankCorrelation(tools)
    for train_index, test_index in loo:
        X_train, X_test = gram_matrix[train_index][:,train_index], gram_matrix[test_index][:, train_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf = rpc.RPC(tools, spearman, svm.SVC(C=1000, probability=True, kernel='precomputed'))
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)

    print("Mean accuracy: %f" %(np.mean(scores)))
    print("Stdv: %f" %(np.std(scores)))

    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label Ranking')
    parser.add_argument('-f', '--features', type=str, required=False)
    parser.add_argument('-g', '--graphs', type=str, required=False)
    parser.add_argument('-o', '--observations', nargs='+', type=str, required=True)
    args = parser.parse_args()

    if args.features:
        features_df = pd.DataFrame.from_csv(args.features)
        observations_df = pd.concat([pd.DataFrame.from_csv(o) for o in args.observations])
        tools = []
        for o in args.observations:
            with open(o + '.tools') as f:
                tools.extend([x .strip() for x in f.readline().split(',')])

        # Prepare data for RPC algorithm
        df = pd.concat([features_df, observations_df], axis=1)
        df.dropna(inplace=True)
        # df_shuffled = df.iloc[np.random.permutation(len(df))]
        df_shuffled = df

        X = df_shuffled.drop('ranking', 1).values
        y = [Ranking(literal_eval(r)) for r in df_shuffled['ranking'].tolist()]


    if args.graphs:
        observations_df = pd.concat([pd.DataFrame.from_csv(o) for o in args.observations])
        tools = []
        for o in args.observations:
            with open(o + '.tools') as f:
                tools.extend([x .strip() for x in f.readline().split(',')])

        graphs_df = pd.DataFrame.from_csv(args.graphs)

        df = pd.concat([graphs_df, observations_df], axis=1)
        df.dropna(inplace=True)
        # df_shuffled = df.iloc[np.random.permutation(len(df))]
        df_shuffled = df

        y = np.array([Ranking(literal_eval(r)) for r in df_shuffled['ranking'].tolist()])
        spearman = distance_metrics.SpearmansRankCorrelation(tools)

        graph_list = []
        for _, row in df_shuffled.iterrows():
            print('Processing ' + row.iloc[0])
            nx_digraph = nx.read_dot(row.iloc[0])
            graph_list.append(nx_digraph)
        print(len(graph_list))
        X = gk.compare_list_normalized(graph_list, h=2)
        k_fold_cv(X, y, tools)





