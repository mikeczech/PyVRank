from PyPRSVT.ranking import rpc
from PyPRSVT.ranking import cross_validation as cv
import pandas as pd
from sklearn import svm
from PyPRSVT.preprocessing.ranking import SpearmansRankCorrelation
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
    X_df = df.drop('ranking', 1)
    y_df = df['ranking']
    # scores = cv.cross_val_score(clf, X_df, y_df, cv=5)
    clf.fit(X_df, y_df)
    #print(clf.predict([X_df.iloc[0].values, X_df.iloc[2].values]))
    #print(clf.predict([X_df.iloc[0].values]))
    print(clf.score(X_df, y_df, SpearmansRankCorrelation(tools)))
