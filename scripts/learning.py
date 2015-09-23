from PyPRSVT.ranking import rpc, distance_metrics
from PyPRSVT.ranking import cross_validation as cv
from PyPRSVT.preprocessing.ranking import Ranking
from ast import literal_eval
import pandas as pd
from sklearn import svm
import argparse
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
    clf = rpc.RPC(tools, svm.SVC, C=1, probability=True, kernel='rbf')
    X = df.drop('ranking', 1).values
    y = [Ranking(literal_eval(r)) for r in df['ranking'].tolist()]
    clf.fit(X, y)
    print(clf.score(X, y, distance_metrics.SpearmansRankCorrelation(tools)))
