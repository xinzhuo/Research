import numpy as np
import pandas as pd
import datetime as dt
import sys
import math
from hft_rsys import HRB
import matplotlib.pyplot as plt
import copy
from Helper import *
from observe_order_book import *

import pickle

def spread(hrb):
    column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                    "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                    "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                    "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                    "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                    "volume", "turnover", "TimeStamp", "hCount",
                    "FallLimit", "RiseLimit", "TotalVolume", "Turnover",
                    "MidPrice", "sprd"]

    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    df = hrb.get_hft_data()
    # ["volume", "turnover", "vwap"]
    get_vwap(df, multiplier, vol_side)

    df["hCount"] = hrb.get_hCount()
    # ignored daily limit case
    df["sprd"] = ((df["AskPrice1"] - df["BidPrice1"])/min_tick).round()

    interval = 20
    #Trade Through:  ["BidTT", "AskTT"]
    tt = copy.deepcopy(get_trade_through(df, multiplier, vol_side))
    tt["BidTTMean"], tt["AskTTMean"] = tt["BidTT"].rolling(interval).mean(), tt["AskTT"].rolling(interval).mean()
    bidTTMean, askTTMean = tt["BidTTMean"].values, tt["AskTTMean"].values

    # Sprd narrow: ["BidSN", "AskSN"]
    sn = copy.deepcopy(get_sprd_narrow(df))
    sn["BidSNMean"], sn["AskSNMean"] = sn["BidSN"].rolling(interval).mean(), sn["AskSN"].rolling(interval).mean()
    bidSNMean, askSNMean = sn["BidSNMean"].values, sn["AskSNMean"].values

    # Trade on best quote: ["BidTR", AskTR"]
    tr = copy.deepcopy(get_bid_ask_trade(df, multiplier, vol_side))
    tr["BidTRMean"], tr["AskTRMean"] = tr["BidTR"].rolling(interval).mean(), tr["AskTR"].rolling(interval).mean()
    bidTRMean, askTRMean = tr["BidTRMean"].values, tr["AskTRMean"].values

    df = df.loc[:, column_names]

    #print(get_sprd_narrow(df, min_tick))
    if df.index.values[0] != 0:
        df.reset_index(inplace=True)
    dfv = df.values
    #print(dfv[1])
    in_position = 0
    period_count, period = 0, 10
    case_total, case_fail = 0, 0

    order_done_count = 0            # Count how many orders are done based on our criteria

    order_price = -1
    last_row = None

    case = 24
    print("Case: " + str(case))
    for row in df.itertuples():
        if row.Index <= 120:
            last_row = row
            continue
        last_sprd = round((last_row.AskPrice1 - last_row.BidPrice1)/min_tick)
        sprd = round((row.AskPrice1 - row.BidPrice1)/min_tick)

        if in_position == 0:
            if case == 1:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 <= last_row.AskPrice1 + min_tick:     #Place buy order 1 tick above BidPrice1
                        order_price = row.BidPrice1 + min_tick
                        in_position = 1
                    elif row.BidPrice1 >= last_row.BidPrice1 - min_tick:   #Place sell order 1 tick below AskPrice1
                        order_price = row.AskPrice1 - min_tick
                        in_position = -1
            elif case == 101:
                if 2 * min_tick > round(row.AskPrice1 - last_row.AskPrice1, 1) >= min_tick \
                        and round(last_row.BidPrice1 - row.BidPrice1) >= 4 * min_tick:     #Place buy order 1 tick above BidPrice1
                    order_price = round(row.BidPrice1 + min_tick, 1)
                    in_position = 1
                elif 2 * min_tick > round(last_row.BidPrice1 - row.BidPrice1, 1) >= min_tick \
                        and round(row.AskPrice1 - last_row.AskPrice1, 1) >= 4 * min_tick:
                    order_price = round(row.AskPrice1 - min_tick, 1)
                    in_position = -1
            elif case == 102:
                if 2 * min_tick > round(row.AskPrice1 - last_row.AskPrice1, 1) >= min_tick \
                        and round(last_row.BidPrice1 - row.BidPrice1) >= 5 * min_tick:     #Place buy order 1 tick above BidPrice1
                    order_price = round(row.BidPrice1 + min_tick, 1)
                    in_position = 1
                elif 2 * min_tick > round(last_row.BidPrice1 - row.BidPrice1, 1) >= min_tick \
                        and round(row.AskPrice1 - last_row.AskPrice1, 1) >= 5 * min_tick:
                    order_price = round(row.AskPrice1 - min_tick, 1)
                    in_position = -1
            elif case == 2:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if row.volume != 0 and row.turnover / row.volume < last_row.BidPrice1:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if row.volume != 0 and row.turnover / row.volume > last_row.AskPrice1:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 3:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if row.volume != 0 and row.turnover / row.volume >= last_row.BidPrice1\
                                and row.turnover / row.volume <= last_row.AskPrice1:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if row.volume != 0 and row.turnover / row.volume <= last_row.AskPrice1\
                                and row.turnover / row.volume >= last_row.BidPrice1:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 4:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if row.volume == 0:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if row.volume == 0:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 5:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if dfv[row.Index - 1][29] - dfv[row.Index - 2][29] >= 0: # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if dfv[row.Index - 1][29] - dfv[row.Index - 2][29] <= 0:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 6:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if dfv[row.Index][29] - dfv[row.Index - 120][29] >= 0: # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if dfv[row.Index][29] - dfv[row.Index - 120][29] <= 0:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 7:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if last_sprd <= 1: # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if last_sprd <= 1:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 8:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if np.mean(dfv[row.Index - 20: row.Index, 29]) > 3: # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if np.mean(dfv[row.Index - 20: row.Index, 29]) > 3 :
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 9:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if np.std(dfv[row.Index - 20: row.Index, 29]) < 1.5:  # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if np.std(dfv[row.Index - 20: row.Index, 29]) < 1.5:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 10:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    volume_sum = np.sum(dfv[row.Index - 10: row.Index, 21])
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if volume_sum != 0 and row.volume/volume_sum >= 0.3:  # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if volume_sum != 0 and row.volume/volume_sum >= 0.3:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 11:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if round((row.AskPrice2 - row.AskPrice1)/min_tick) <= 1:  # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if round((row.BidPrice1 - row.BidPrice2)/min_tick) <= 1:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 12:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if round((row.BidPrice1 - row.BidPrice2) / min_tick) <= 1:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1

                        if round((row.AskPrice2 - row.AskPrice1) / min_tick) <= 1:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 13:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if round((last_row.BidPrice1 - last_row.BidPrice2) / min_tick) > 4:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if round((last_row.AskPrice2 - last_row.AskPrice1) / min_tick) > 4:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 14:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if row.BidSize1/(row.BidSize1 + row.AskSize1) >= 0.7:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if row.AskSize1/(row.BidSize1 + row.AskSize1) >= 0.7:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 15:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if bidTTMean[row.Index] <= 0.4:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if askTTMean[row.Index] <= 0.4:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 16:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if askTTMean[row.Index] <= 0.1:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if bidTTMean[row.Index] <= 0.1:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 17:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if bidSNMean[row.Index] <= 0.1:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if askSNMean[row.Index] <= 0.1:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 18:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if askSNMean[row.Index] <= 0.1:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if bidSNMean[row.Index] <= 0.1:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 19:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if askTRMean[row.Index] >= 0.8:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if bidTRMean[row.Index] >= 0.8:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 20:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if bidTRMean[row.Index] <= 0.5:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if askTRMean[row.Index] <= 0.5:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 21:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if np.std(dfv[row.Index - 20: row.Index, 8]) < 2 * min_tick:  # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if np.std(dfv[row.Index - 20: row.Index, 10]) < 2 * min_tick:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 22:
                if sprd - last_sprd >= 4:  # check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:  # Place buy order 1 tick above BidPrice1
                        if np.std(dfv[row.Index - 20: row.Index, 10]) < 1 * min_tick:  # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:  # Place sell order 1 tick below AskPrice1
                        if np.std(dfv[row.Index - 20: row.Index, 8]) < 1 * min_tick:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -10
            elif case == 23:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if sprd >= 8: # Positive ret
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if sprd >= 8:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 24: # Combine #3, #4 and # 22
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if row.volume == 0 or (round(row.turnover / row.volume, 3) >= last_row.BidPrice1
                                               and np.std(dfv[row.Index - 20: row.Index, 10]) < 2 * min_tick):
                            order_price = round(row.BidPrice1 + min_tick,1)
                            in_position = 1


                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if row.volume == 0 or (round(row.turnover / row.volume, 3) <= last_row.AskPrice1
                                               and np.std(dfv[row.Index - 20: row.Index, 8]) < 2 * min_tick):
                            order_price = round(row.AskPrice1 - min_tick, 1)
                            in_position = -1

            elif case == 25: # Combine #4, #18 and # 21C
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if row.volume == 0 or (askSNMean[row.Index] <= 0.3
                                               and np.std(dfv[row.Index - 20: row.Index, 8]) < 3 * min_tick):
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if row.volume == 0 or (bidSNMean[row.Index] <= 0.3
                                               and np.std(dfv[row.Index - 20: row.Index, 10]) < 3 * min_tick):
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 26: # Combine #4, #18 and # 21C
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if row.volume == 0 or (askSNMean[row.Index] <= 0.3
                                               and np.std(dfv[row.Index - 20: row.Index, 8]) < 3 * min_tick
                                               and row.turnover / row.volume >= last_row.BidPrice1):
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if row.volume == 0 or (bidSNMean[row.Index] <= 0.3
                                               and np.std(dfv[row.Index - 20: row.Index, 10]) < 3 * min_tick
                                               and row.turnover / row.volume <= last_row.AskPrice1):
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 27:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if row.volume == 0 or (askSNMean[row.Index] <= 0.3
                                               and round((last_row.BidPrice1 - last_row.BidPrice2) / min_tick) > 3):
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if row.volume == 0 or (bidSNMean[row.Index] <= 0.3
                                               and round((last_row.AskPrice2 - last_row.AskPrice1) / min_tick) > 3):
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
            elif case == 28:
                if sprd - last_sprd >= 4:                       #check for spread widen, only consider single side widen here
                    if row.AskPrice1 == last_row.AskPrice1:     #Place buy order 1 tick above BidPrice1
                        if row.volume <= 10.5 \
                                and (dfv[row.Index][28] - dfv[row.Index - 10][28])/min_tick > -3.75\
                                and np.std(dfv[row.Index - 19: row.Index + 1, 8]) <= 2.363 * min_tick:
                            order_price = row.BidPrice1 + min_tick
                            in_position = 1
                    elif row.BidPrice1 == last_row.BidPrice1:   #Place sell order 1 tick below AskPrice1
                        if row.volume <= 10.5 \
                                and (dfv[row.Index][28] - dfv[row.Index - 10][28])/min_tick * -1 > -3.75\
                                and np.std(dfv[row.Index - 19: row.Index + 1, 10]) <= 2.363 * min_tick:
                            order_price = row.AskPrice1 - min_tick
                            in_position = -1
        else:

            #Oppo price reaches limit:
            if in_position > 0:
                if period_count == 0: # position entered last tick
                    mean_sprd = np.mean(dfv[row.Index - 5: row.Index, 29])
                    mean_volume = np.mean(dfv[row.Index - 5: row.Index, 20])
                    volume_per_sprd = mean_volume/mean_sprd
                    order_done = False
                    if row.volume != 0:
                        current_vwap = row.turnover / row.volume
                        half_sprd = (current_vwap - order_price)/min_tick

                        if row.volume/2 >= half_sprd * volume_per_sprd: # Here we consider our order done
                            order_done_count += 1
                            order_done = True

                    if not order_done:
                        in_position = 0
                        last_row = row
                        continue

                if row.AskPrice1 < order_price:             # AskPrice lower than our buy price
                    case_fail += 1
                    case_total += 1
                    #print("Index: " + str(row.Index))
                    #print("Count: " + str(period_count))
                    in_position = 0
                    period_count = 0
                else:
                    period_count += 1
            elif in_position < 0:
                if period_count == 0:  # position entered last tick
                    mean_sprd = np.mean(dfv[row.Index - 5: row.Index, 29])
                    mean_volume = np.mean(dfv[row.Index - 5: row.Index, 20])
                    volume_per_sprd = mean_volume / mean_sprd
                    order_done = False
                    if row.volume != 0:
                        current_vwap = row.turnover / row.volume
                        half_sprd = (order_price - current_vwap) / min_tick

                        if row.volume / 2 >= half_sprd * volume_per_sprd:  # Here we consider our order done
                            order_done_count += 1
                            order_done = True
                    if not order_done:
                        in_position = 0
                        last_row = row
                        continue

                if row.BidPrice1 > order_price:             # BidPrice higher than our sell price
                    case_fail += 1
                    case_total += 1
                    #print("Index: " + str(row.Index))
                    #print("Count: " + str(period_count))
                    in_position = 0
                    period_count = 0
                else:
                    period_count += 1
            """

             # mean of 5 ticks mid price
            if in_position > 0:
                if period_count == 0: # position entered last tick
                    mean_sprd = np.mean(dfv[row.Index - 5: row.Index, 29])
                    mean_volume = np.mean(dfv[row.Index - 5: row.Index, 20])
                    volume_per_sprd = mean_volume/mean_sprd
                    order_done = False
                    if row.volume != 0:
                        current_vwap = row.turnover / row.volume
                        half_sprd = round((current_vwap - order_price)/min_tick,3)

                        if row.volume/2 >= half_sprd * volume_per_sprd: # Here we consider our order done
                            order_done_count += 1
                            order_done = True

                    if not order_done:
                        in_position = 0
                        last_row = row
                        continue
                if np.mean(dfv[row.Index - 4: row.Index + 1, 28]) < order_price:  # AskPrice lower than our buy price
                    case_fail += 1
                    case_total += 1
                    # print("Index: " + str(row.Index))
                    # print("Count: " + str(period_count))
                    in_position = 0
                    period_count = 0
                else:
                    period_count += 1
            elif in_position < 0:
                if period_count == 0:  # position entered last tick
                    mean_sprd = np.mean(dfv[row.Index - 5: row.Index, 29])
                    mean_volume = np.mean(dfv[row.Index - 5: row.Index, 20])
                    volume_per_sprd = mean_volume / mean_sprd
                    order_done = False
                    if row.volume != 0:
                        current_vwap = row.turnover / row.volume
                        half_sprd = round((order_price - current_vwap) / min_tick, 3)

                        if row.volume / 2 >= half_sprd * volume_per_sprd:  # Here we consider our order done
                            order_done_count += 1
                            order_done = True
                    if not order_done:
                        in_position = 0
                        last_row = row
                        continue

                if np.mean(dfv[row.Index - 4: row.Index + 1, 28]) > order_price:  # BidPrice higher than our sell price
                    case_fail += 1
                    case_total += 1
                    # print("Index: " + str(row.Index))
                    # print("Count: " + str(period_count))
                    in_position = 0
                    period_count = 0
                else:
                    period_count += 1
            """
            """
            # Oppo price reach limit twice in 8 ticks
            if in_position > 0:
                if period_count == 0: # position entered last tick
                    mean_sprd = np.mean(dfv[row.Index - 5: row.Index, 29])
                    mean_volume = np.mean(dfv[row.Index - 5: row.Index, 20])
                    volume_per_sprd = mean_volume/mean_sprd
                    order_done = False
                    if row.volume != 0:
                        current_vwap = row.turnover / row.volume
                        half_sprd = (current_vwap - order_price)/min_tick

                        if row.volume/2 >= half_sprd * volume_per_sprd: # Here we consider our order done
                            order_done_count += 1
                            order_done = True

                    if not order_done:
                        in_position = 0
                        last_row = row
                        continue
                if row.AskPrice1 < order_price:             # AskPrice lower than our buy price
                    if flag == 1:
                        case_fail += 1
                        case_total += 1
                        #print("Index: " + str(row.Index))
                        #print("Count: " + str(period_count))
                        in_position = 0
                        period_count = 0
                        flag = 0
                    else:
                        flag = 1
                else:
                    period_count += 1
            elif in_position < 0:
                if period_count == 0:  # position entered last tick
                    mean_sprd = np.mean(dfv[row.Index - 5: row.Index, 29])
                    mean_volume = np.mean(dfv[row.Index - 5: row.Index, 20])
                    volume_per_sprd = mean_volume / mean_sprd
                    order_done = False
                    if row.volume != 0:
                        current_vwap = row.turnover / row.volume
                        half_sprd = (order_price - current_vwap) / min_tick

                        if row.volume / 2 >= half_sprd * volume_per_sprd:  # Here we consider our order done
                            order_done_count += 1
                            order_done = True
                    if not order_done:
                        in_position = 0
                        last_row = row
                        continue
                if row.BidPrice1 > order_price:             # BidPrice higher than our sell price
                    if flag == 1:
                        case_fail += 1
                        case_total += 1
                        #print("Index: " + str(row.Index))
                        #print("Count: " + str(period_count))
                        in_position = 0
                        period_count = 0
                        flag = 0
                    else:
                        flag = 1
                else:
                    period_count += 1

            """
            """
            # mean of 5 ticks bid price
            if in_position > 0:
                if period_count == 0: # position entered last tick
                    mean_sprd = np.mean(dfv[row.Index - 5: row.Index, 29])
                    mean_volume = np.mean(dfv[row.Index - 5: row.Index, 20])
                    volume_per_sprd = mean_volume/mean_sprd
                    order_done = False
                    if row.volume != 0:
                        current_vwap = row.turnover / row.volume
                        half_sprd = (current_vwap - order_price)/min_tick

                        if row.volume/2 >= half_sprd * volume_per_sprd: # Here we consider our order done
                            order_done_count += 1
                            order_done = True

                    if not order_done:
                        in_position = 0
                        last_row = row
                        continue
                if np.mean(dfv[row.Index - 4: row.Index + 1, 8]) < order_price - min_tick:  # AskPrice lower than our buy price
                    case_fail += 1
                    case_total += 1
                    # print("Index: " + str(row.Index))
                    # print("Count: " + str(period_count))
                    in_position = 0
                    period_count = 0
                else:
                    period_count += 1
            elif in_position < 0:
                if period_count == 0:  # position entered last tick
                    mean_sprd = np.mean(dfv[row.Index - 5: row.Index, 29])
                    mean_volume = np.mean(dfv[row.Index - 5: row.Index, 20])
                    volume_per_sprd = mean_volume / mean_sprd
                    order_done = False
                    if row.volume != 0:
                        current_vwap = row.turnover / row.volume
                        half_sprd = (order_price - current_vwap) / min_tick

                        if row.volume / 2 >= half_sprd * volume_per_sprd:  # Here we consider our order done
                            order_done_count += 1
                            order_done = True
                    if not order_done:
                        in_position = 0
                        last_row = row
                        continue
                if np.mean(dfv[row.Index - 4: row.Index + 1, 10]) > order_price + min_tick:  # BidPrice higher than our sell price
                    case_fail += 1
                    case_total += 1
                    # print("Index: " + str(row.Index))
                    # print("Count: " + str(period_count))
                    in_position = 0
                    period_count = 0
                else:
                    period_count += 1
            """

            if period_count == period:
                flag = 0
                in_position = 0
                period_count = 0
                case_total += 1

        last_row = row
    print("Total:   " + str(case_total))
    print("Fail:    " + str(case_fail))
    print("Done:    " + str(order_done_count))


if __name__ == '__main__':
    numbers = ["6"]
    for number in numbers:
        Month = "190" + number
        print("Month: " + Month)
        """
        st = dt.datetime(2019, int(number) - 1, 15)
        et = dt.datetime(2019, int(number), 15)
        contract = "IF"
        source = "l2_cffex"
        #hrb = HRB.HRB(st, et, 'IF', '1905', 'l2_cffex', 0)
        hrb = HRB.HRB(st, et, contract, Month, source, 0)

        with open(Month + '.pickle', 'wb') as handle:
            pickle.dump(hrb, handle)
        """
        with open(Month + '.pickle', 'rb') as handle:
            hrb = pickle.load(handle)
            spread(hrb)


    #visualize_ob_history_2(contract, Month, st, et, source)
    #volume_residual(hrb, 120)