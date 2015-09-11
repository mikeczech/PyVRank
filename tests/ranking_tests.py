import pandas as pd
from PyPRSVT.preprocessing import competition
from PyPRSVT.ranking import rpc
import sklearn

def rpc_test():
    clf = rpc.RPC(sklearn.svm.SVC())
