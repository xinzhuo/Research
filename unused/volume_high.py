import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
from hft_rsys import *

# Data structure to store historial orderbook info
from Hist_OB import Cylinder
# Replicate between tick info
from Helper import *

# Utilities used:
from scipy.stats import linregress
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

"""
The general idea is to first identify abnormal volume,
and thus detect potential trend.

Some general ideas:
1. Use detail trade volumes
2. only use traded tick to eliminate noise

or:
Use regression  on buy volume, sell volume
                on 

or:
Identify temporary barrier
"""

def trial_1(hrb):
    column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                    "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                    "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                    "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                    "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                    "volume", "turnover", "TimeStamp", "hCount",
                    "NonZeroVol", "TotalVolume", "Turnover"]

    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    df = hrb.get_hft_data()

    df["hCount"] = hrb.get_hCount()
    df = filter_volume(df, multiplier, vol_side)
    df = df.loc[:, column_names]
    length = len(df.index)

    ###########################################################
    period = 30

    new_session_counter = 0

    last_row = None
    signal = np.zeros(length)                       # defined as buy/sell signal
    ask_volume_list, bid_volume_list = np.zeros(length), np.zeros(length)
                                                    # tick by tick volume
    true_ask_volume_list, true_bid_volume_list = [], []
                                                    # store only when volume > 0

    for index, row in df.iterrows():
        if index == 0 or row["volume"] == 0:
            last_row = row
            continue

        tick_volumes = analyze_between_tick(last_row, row, min_tick, multiplier, vol_side)

        # Here we extract information from our matching model:
        bid_size_sum, ask_size_sum = 0, 0       # total volume on bid and ask side
        bid_trade_level, ask_trade_level = 0, 0 # Number of levels traded on bid and ask side
                                                # Could also use highest (price - last_row[askprice1]) / min_tick

        for price, volume in tick_volumes.items():
            if price >= last_row["AskPrice1"]:      # Considered as buy volume
                ask_size_sum += volume
                ask_trade_level += 1
            elif price <= last_row["BidPrice1"]:    # Considered as sell volume
                bid_size_sum += volume
                bid_trade_level += 1
            else:                                   # ignore if volume in the middle
                pass

        ask_volume_list[index] = ask_size_sum
        bid_volume_list[index] = bid_size_sum
        true_ask_volume_list.append(ask_size_sum)
        true_bid_volume_list.append(bid_size_sum)

        if len(true_ask_volume_list) > period:
            regr = linear_model.LinearRegression(fit_intercept = False)
            regr.fit(np.reshape(true_bid_volume_list[-1 * period:], (period, 1)),
                     np.reshape(true_ask_volume_list[-1 * period:], (period, 1)))

            ask_pred = regr.predict(np.reshape(true_bid_volume_list[-1 * period:], (period, 1)))

            # The coefficients
            print('Coefficients: \n', regr.coef_)
            # The mean squared error
            print("Mean squared error: %.2f"
                  % mean_squared_error(np.reshape(true_ask_volume_list[-1 * period:], (period, 1)), ask_pred))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(np.reshape(true_ask_volume_list[-1 * period:], (period, 1)), ask_pred))

            # Plot outputs
            plt.scatter(true_bid_volume_list[-1 * period:], true_ask_volume_list[-1 * period:], color='black')
            plt.plot(true_bid_volume_list[-1 * period:], ask_pred, color='blue', linewidth=1)

            print(np.sum(true_bid_volume_list[-1 * period:])/np.sum(true_ask_volume_list[-1 * period:]))
            plt.show()

            input()

def trial_2(hrb):
    column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                    "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                    "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                    "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                    "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                    "volume", "turnover", "TimeStamp", "hCount",
                    "NonZeroVol", "TotalVolume", "Turnover", "MidPrice"]

    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    df = hrb.get_hft_data()

    df["hCount"] = hrb.get_hCount()
    df = filter_volume(df, multiplier, vol_side)
    df = df.loc[:, column_names]
    length = len(df.index)

    ###########################################################
    period = 30

    new_session_counter = 0

    last_row = None
    signal = np.zeros(length)                       # defined as buy/sell signal
    ask_volume_list, bid_volume_list = np.zeros(length), np.zeros(length)
                                                    # tick by tick volume
    true_ask_volume_list, true_bid_volume_list = [], []
                                                    # store only when volume > 0
    true_price_list = []                            # store corresponding price

    fig = plt.figure(figsize=(14, 8))
    for index, row in df.iterrows():
        if index == 0 or row["volume"] == 0:
            last_row = row
            continue

        tick_volumes = analyze_between_tick(last_row, row, min_tick, multiplier, vol_side)

        # Here we extract information from our matching model:
        bid_size_sum, ask_size_sum = 0, 0       # total volume on bid and ask side
        bid_trade_level, ask_trade_level = 0, 0 # Number of levels traded on bid and ask side
                                                # Could also use highest (price - last_row[askprice1]) / min_tick

        for price, volume in tick_volumes.items():
            if price >= last_row["AskPrice1"]:      # Considered as buy volume
                ask_size_sum += volume
                ask_trade_level += 1
            elif price <= last_row["BidPrice1"]:    # Considered as sell volume
                bid_size_sum += volume
                bid_trade_level += 1
            else:                                   # ignore if volume in the middle
                pass

        ask_volume_list[index] = ask_size_sum
        bid_volume_list[index] = bid_size_sum
        true_ask_volume_list.append(ask_size_sum)
        true_bid_volume_list.append(bid_size_sum)
        true_price_list.append(row["MidPrice"])

        if len(true_ask_volume_list) > period:
            regr = linear_model.LinearRegression(fit_intercept = False)
            test = np.subtract(true_ask_volume_list[-1 * period:], true_bid_volume_list[-1 * period:])
            regr.fit(np.reshape(list(range(period)), (period, 1)), np.reshape(test, (period, 1)))

            pred = regr.predict(np.reshape(list(range(period)), (period, 1)))

            # The coefficients
            print('Coefficients: \n', regr.coef_)
            # The mean squared error
            print("Mean squared error: %.2f"
                  % mean_squared_error(np.reshape(true_ask_volume_list[-1 * period:], (period, 1)), pred))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(test, pred))
            ax1 = fig.add_subplot(111)
            # Plot outputs
            ax1.scatter(list(range(period)), test, color='black')
            ax1.plot(list(range(period)), pred, color='blue', linewidth=1)

            ax2 = ax1.twinx()
            ax2.plot(true_price_list[-period:])

            print(row["TimeStamp"])
            print((true_price_list[-1] - true_price_list[-1 * period - 1])/min_tick)

            #print(np.sum(true_bid_volume_list[-1 * period:])/np.sum(true_ask_volume_list[-1 * period:]))
            plt.draw()
            plt.pause(0.001)
            input()
            fig.clf()


if __name__ == '__main__':
    st = datetime(2019, 3, 18, 10, 0, 0)
    et = datetime(2019, 3, 20)
    #hrb = HRB.HRB(st, et, 'IF', '1904', 'l2_cffex', 0)
    hrb = HRB.HRB(st, et, 'jm', '1905', 'l2_dce', 0) # Use only L2 Data!!!
    trial_2(hrb)