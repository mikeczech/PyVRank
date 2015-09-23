from PyPRSVT.ranking import rpc, distance_metrics
from PyPRSVT.preprocessing.ranking import Ranking
from ast import literal_eval
import pandas as pd
from sklearn import svm, cross_validation
import argparse
import numpy as np
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label Ranking')
    parser.add_argument('-f', '--features', type=str, required=True)
    parser.add_argument('-o', '--observations', nargs='+', type=str, required=True)
    args = parser.parse_args()

    features_df = pd.DataFrame.from_csv(args.features)
    observations_df = pd.concat([pd.DataFrame.from_csv(o) for o in args.observations])
    tools = set([])
    for o in args.observations:
        with open(o + '.tools') as f:
            tools |= set(f.readline().split(','))

    # Prepare data for RPC algorithm
    df = pd.concat([features_df, observations_df], axis=1)
    df.dropna(inplace=True)
    df_shuffled = df.iloc[np.random.permutation(len(df))]

    X = df_shuffled.drop('ranking', 1).values
    y = [Ranking(literal_eval(r)) for r in df_shuffled['ranking'].tolist()]
    spearman = distance_metrics.SpearmansRankCorrelation(tools)

    # Train, Test Split
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.4, random_state=0)
    clf = rpc.RPC(tools, spearman, svm.SVC(C=1, probability=True, kernel='rbf')).fit(X_train, y_train)
    print("Accuracy: %0.2f" % clf.score(X_test, y_test))

    # Cross Validation
    # scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))