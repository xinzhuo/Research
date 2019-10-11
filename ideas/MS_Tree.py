import numpy as np
import pandas as pd
import datetime as dt
import sys
import math
from hft_rsys import *
import matplotlib.pyplot as plt
import copy
from Helper import *
from observe_order_book import HRB

import pickle
from Microstructure_Factors import *

number = str(5)
Month = "190" + number
with open(Month + '_test.pickle', 'rb') as handle:
    hrb = pickle.load(handle)
    msf = MSF_Library(hrb)

msf.df.drop([ 'volume', 'turnover'], axis = 1, inplace = True)
msf.get_all()
ndf = msf.df
pd.set_option('display.max_columns', 500)
unused_column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                        "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                        "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                        "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                        "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                        "turnover", "TimeStamp", "hCount",
                        "FallLimit", "RiseLimit", "TotalVolume", "Turnover",
                        "MidPrice", "bid_or_ask", "vwap"]

tdf = copy.deepcopy(ndf)
# remove first 50 rows
tdf = tdf.iloc[50:]
tdf = tdf[tdf["bid_or_ask"] != "NAN"]
tdf.drop(unused_column_names, axis = 1, inplace = True)

from sklearn import tree
feature_names = tdf.columns.values
predictors = tdf.values
responses = msf.response_list[50:]
responses = responses[~np.isnan(responses)]

"""
for name in feature_names:
    print(name + ":")
    print(tdf[name].isnull().any().any())
"""
print(len(responses))
print(predictors.shape)

clf = tree.DecisionTreeClassifier(max_depth = 6, min_samples_split = 100)
clf = clf.fit(predictors, responses)

fig = plt.figure(figsize=(14, 8))
tree.plot_tree(clf, feature_names = feature_names)
plt.show()