import numpy as np
import pandas as pd
import datetime as dt
import sys
import math
from hft_rsys import *

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)
def give_big_volume_direction(row):
    if row["Volume"] > row["Threshold"]:
        if row["PriceChg"] > 0:
            return row["Volume"]
        elif row["PriceChg"] < 0:
            return -1 * row["Volume"]
    return 0


def get_order_dir_sum(hrb, period):
    priceData = hrb.get_hft_data().loc[:, ["AskPrice1", "BidPrice1", "TotalVolume", "Turnover"]]
    priceDataV = priceData.values
    for i in range(priceDataV.shape[0]):
        if priceDataV[i, 0] == 0 and priceDataV[i, 1] != 0:
            priceDataV[i, 0] = priceDataV[i, 1]
        elif priceDataV[i, 0] != 0 and priceDataV[i, 1] == 0:
            priceDataV[i, 1] = priceDataV[i, 0]
        elif priceDataV[i, 0] == 0:
            priceDataV[i, 0] = priceDataV[i - 1, 0]
            priceDataV[i, 1] = priceDataV[i - 1, 1]
    priceData = pd.DataFrame(priceDataV, index = priceData.index, columns=["AskPrice1", "BidPrice1", "TotalVolume", "Turnover"])

    priceData["Mid"] = (priceData["AskPrice1"] + priceData["BidPrice1"])/2
    priceData["Volume"] = (priceData["TotalVolume"] - priceData["TotalVolume"].shift(1))/2
    priceData["Threshold"] = priceData["Volume"].rolling(period).mean() + 2 * priceData["Volume"].rolling(period).std()
    priceData["PriceChg"] = priceData["Mid"] - priceData["Mid"].shift(1)
    priceData["PriceChgInterval"] = priceData["Mid"] - priceData["Mid"].shift(period)
    priceData["LargeVol"] = priceData.apply(give_big_volume_direction, axis=1)

    priceData["NetLargeVol"] = priceData["LargeVol"].rolling(period).sum()
    priceData["LargeVolImpact"] = priceData["NetLargeVol"]/priceData["PriceChgInterval"]
    priceData["LargeVolImpact"].replace([np.inf, -np.inf], np.nan, inplace = True)
    priceData["LargeVolImpact"].fillna(method='ffill', inplace = True)

    return priceData.loc[:, ["LargeVolImpact"]]

if __name__ == '__main__':
    st = dt.datetime(2019, 1, 1)
    et = dt.datetime(2019, 2, 1)
    hrb = HRB.HRB(st, et, 'rb', '1905', 'l1_ctp')


    n = 2 * 60
    order_dir_sum = get_order_dir_sum(hrb, n)
    hrb.input_indicator(order_dir_sum, "orderDirSum")

    """
    hrb.get_status()
    hrb.indicator_observation('orderDirSum')
    hrb.indicator_distribution_observation('orderDirSum')
    hrb.indicator_distribution_observation('orderDirSum', plot_type='hist', hist_range=[-1, 1], hist_precision=0.1)
    hrb.indicator_linear_regression('orderDirSum', period=100)

    
    """
    hrb.generate_signal('orderDirSum', 'orderDirSumB', 200, 'independent', 200, 'intraday')
    #hrb.generate_signal('orderDirSum', 'orderDirSumS', -0.1, 'independent', 200, 'intraday')
    hrb.signal_show_response('orderDirSumB', period=2 * 600, direct = "buy", draw_signal=False)
    #hrb.signal_show_response('orderDirSumS', period=200, direct="sell")