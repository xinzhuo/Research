import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy as sp
"""
Interpret trading volume on different price levels between two ticks
return: dict with price as key and volume as value
"""

LAMBDA = 0.00001

price_columns = ["BidPrice5", "BidPrice4", "BidPrice3", "BidPrice2", "BidPrice1",
                 "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5"]
volume_columns = ["BidSize5", "BidSize4", "BidSize3", "BidSize2", "BidSize1",
                  "AskSize1", "AskSize2", "AskSize3", "AskSize4", "AskSize5"]
def analyze_between_tick(tick1, tick2,  min_step):
    volume = tick2["volume"]
    # Case 1:
    # No further action needed if volume = 0, also most common case
    if volume == 0:
        return dict()

    # Map price with size for both ticks into two dicts
    tick1_dict, tick2_dict = {}, {}
    for index, price in enumerate(price_columns):
        if price[:3] == "Bid":
            tick1_dict[tick1[price]] = -1 * tick1[volume_columns[index]]
            tick2_dict[tick2[price]] = -1 * tick2[volume_columns[index]]
        else:
            tick1_dict[tick1[price]] = tick1[volume_columns[index]]
            tick2_dict[tick2[price]] = tick2[volume_columns[index]]

    # Here we assume volume and turnover for each tick are pre - calculated
    # volume residual: sum of relative position in minimum steps to BidPrice1 of previous tick
    vol_residual = round((tick2["turnover"] - (volume * tick1["BidPrice1"])) / min_step)

    # This dict store size done on each price level
    result_dict = {}

    # Case 2:
    # When best bid&ask unchanged
    if tick1["BidPrice1"] == tick2["BidPrice1"] and tick1["AskPrice1"] == tick2["AskPrice1"] \
            and tick1["AskPrice1"] - tick1["BidPrice1"] == min_step:
        if 0 <= vol_residual <= volume:  # vol_resid should be bounded by 0 and volume in this case
            result_dict[0] = int(volume - vol_residual)
            result_dict[1] = int(vol_residual)
        else:  # A strange case
            result_dict[int(round(vol_residual / volume))] = int(volume)

        return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)

    # Assumption: no price fluctuation between tick
    lowBound = min(tick1["BidPrice1"], tick2["BidPrice1"])
    upBound = max(tick1["AskPrice1"], tick2["AskPrice1"])
    # all possible traded price under our assumption
    price_list = np.arange(lowBound, upBound + min_step, min_step)
    bucket_dict = {}

    # map position relative to the best bid to volume traded inferred by the orderbook change
    non_zero_keys, non_zero_values = [], []
    for price in price_list:
        # All the orders at the level was consumed
        position = round((price - tick1["BidPrice1"]) / min_step)
        if price not in tick1_dict:
            bucket_dict[position] = 0
        elif price not in tick2_dict:
            if price > tick2["AskPrice5"] or price < tick2["BidPrice5"]:
                bucket_dict[position] = 0
            else:
                bucket_dict[position] = abs(tick1_dict[price])
        elif tick1_dict[price] * tick2_dict[price] <= 0:
            bucket_dict[position] = abs(tick1_dict[price])
        else:
            if abs(tick1_dict[price]) > abs(tick2_dict[price]):
                bucket_dict[position] = abs(tick1_dict[price]) - abs(tick2_dict[price])
            else:
                bucket_dict[position] = 0

        if bucket_dict[position] != 0:
            non_zero_keys.append(position)
            non_zero_values.append(bucket_dict[position])

    # Case 3:
    # Even if best price changed, we can tell there are 2 possible trades price based on OB info
    if len(non_zero_keys) == 2:
        x1 = (non_zero_keys[1] * volume - vol_residual) / (non_zero_keys[1] - non_zero_keys[0])
        # Here we check if x1 is an integer, meaning that we have have the 2 correct prices
        if round(x1) - LAMBDA < x1 < round(x1) + LAMBDA:
            x2 = volume - x1
            if x1 >= 0 and x2 >= 0:
                result_dict[non_zero_keys[0]] = round(x1)
                result_dict[non_zero_keys[1]] = round(x2)
            else:
                result_dict[int(round(vol_residual / volume))] = volume
            return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)

    """
    Here we want to first re-arrange the order of the dictionary, 
    so the algo access the prices that are more likely to be traded first. 
    Mid-price up, we look at bid prices first and vice versa.
    There are 2 situation when the spread remains the same, 
    3 situations when the spread widens and 2 situations when the spread narrows.
    Also here, we don't consider unchanged mid_price specifically

    If Mid price unchanged, we compare v
    """
    bucket_dict_temp = OrderedDict()
    mid_price1 = (tick1["BidPrice1"] + tick1["AskPrice1"]) / 2
    mid_price2 = (tick2["BidPrice1"] + tick2["AskPrice1"]) / 2
    if mid_price1 > mid_price2 or \
            (mid_price1 == mid_price2 and tick2["vwap"] <= mid_price1):
        # rank bid before ask if mid is lower, position relative to Tick1[Bid1]
        price_posit = 0
        while price_posit in bucket_dict:
            bucket_dict_temp[price_posit] = bucket_dict[price_posit]
            price_posit = price_posit - 1

        price_posit = 1
        while price_posit in bucket_dict:
            bucket_dict_temp[price_posit] = bucket_dict[price_posit]
            price_posit = price_posit + 1
    elif mid_price1 < mid_price2 or \
            (mid_price1 == mid_price2 and tick2["vwap"] > mid_price1):
        price_posit = int((tick1["AskPrice1"] - tick1["BidPrice1"]) / min_step)
        while price_posit in bucket_dict:
            bucket_dict_temp[price_posit] = bucket_dict[price_posit]
            price_posit = price_posit + 1

        price_posit = int((tick1["AskPrice1"] - tick1["BidPrice1"]) / min_step) - 1
        while price_posit in bucket_dict:
            bucket_dict_temp[price_posit] = bucket_dict[price_posit]
            price_posit = price_posit - 1
    bucket_dict = bucket_dict_temp

    for price_posit, value in bucket_dict.items():
        if value != 0:
            # Here checks if there are enough volume for the ob inferred volume
            # Should we also check for volume residual here??????
            if volume < value:
                break
            result_dict[price_posit] = int(value)
            volume -= value
            vol_residual -= price_posit * value

    # Case 4:
    # When volume are consumed exactly during the process
    if volume == 0:
        return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)
        # Need to check if volResidual == 0 in the future update

    price_list = list(bucket_dict.keys())
    # Case 5:
    # When There are "three" possible prices, this should represent a lot of the remaining case
    if len(price_list) == [3, 4]:
        solution3 = solve_3uk_2eq(price_list[:3], [int(volume), vol_residual])
        if solution3:
            for index, value in enumerate(price_list[:3]):
                if value in result_dict:
                    result_dict[value] += round(solution3[index])
                else:
                    result_dict[value] = round(solution3[index])
            return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)

    # Case 6
    """
     There are two cases in the next step:
     1. Total volume inferred by orderbook is less than Volume, so we allocate the volume to ob first and some more
        However, problem could arise when volume residual presented in ob exceeded calculated residual.
     2. Total volume inferred by orderbook is more than Volume. We fill the volume inferred partly based on our
        ranking.

    Solution: we assign leftover volume to the top two possible prices
    Possible problem: the leftover volume and vol_residual might not make senses - giving negative solutions
    """
    x1 = (price_list[1] * volume - vol_residual) / (price_list[1] - price_list[0])
    x2 = volume - x1

    if x1 >= 0 and x2 >= 0:
        if price_list[0] in result_dict:
            result_dict[price_list[0]] += round(x1)
        else:
            result_dict[price_list[0]] = round(x1)
        if price_list[1] in result_dict:
            result_dict[price_list[1]] += round(x2)
        else:
            result_dict[price_list[1]] = round(x2)
    else:
        price_posit = int(round(vol_residual / volume))
        if price_posit in result_dict:
            result_dict[price_posit] += volume
        elif price_posit in bucket_dict:
            result_dict[price_posit] = volume
        elif price_posit + 1 in bucket_dict or price_posit - 1 in bucket_dict:
            result_dict[price_posit] = volume
    return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)

def process_between_tick_result(result_dict, base, min_step):
    """
    This method process the final return item in
    :param base:
    :param min_tick:
    :return:
    """
    new_result_dict = {}
    for key, value in result_dict.items():
        if value != 0:
            new_result_dict[base + key * min_step] = int(value)
    return new_result_dict

def solve_3uk_2eq(x, y):
    """
    x1 + x2 + x3 = volume
    x1 * p1 + x2 * p2 + x3 * p3 = residual
    :param x: x here is the residual for each price
    :param y: y is [volume, residual]
    :return:
    """
    assert len(x) == 3 and len(y) == 2
    for c in range(y[0]):
        a = ((y[0] - c) * x[1] - (y[1] - c * x[2])) / (x[1] - x[0])
        # Check if a is integer >= 0
        if a >= 0 and round(a) - LAMBDA < a < round(a) + LAMBDA:
            # check if b is positive
            b = y[0] - c - a
            if b >= 0:  # we have our solution
                return [a, b, c]

    return None
"""
Process and filter out volume = 0
Return same DataFrame with binary column indicating whether volume is non-zero
as well as volume and turnover by tick
"""
def filter_volume(priceData, multiplier, vol_side = 2):
    priceData.reset_index(inplace=True)

    priceData["volume"] = priceData["TotalVolume"].diff() / vol_side
    priceData["turnover"] = priceData["Turnover"].diff() / multiplier / vol_side
    priceData["volume"] = priceData["volume"].clip(0)
    priceData["turnover"] = priceData["turnover"].clip(0)

    priceData["NonZeroVol"] = np.where(priceData["volume"] != 0, 1, 0)

    return priceData

"""
Re-sample data set by volume/turnover
Data here should have a column labeling different trading sessions
TWo column were added: 
["bar_label"]:  Indicate end of bar (0 or 1)
["bar_size"]:   Non-zero only if bar_label == 1
Return: (same DataFrame with two new column, expected bar_size)
as well as volume and turnover by tick
"""
def get_alt_clock(priceData, multiplier, method = "volume", vol_side = 2, bar_size = 20):
    priceData.reset_index(inplace=True)

    priceData["vol"] = priceData["TotalVolume"].diff()
    priceData["to"] = priceData["Turnover"].diff() / multiplier / vol_side
    priceData["vol"] = priceData["vol"].clip(0)
    priceData["to"] = priceData["to"].clip(0)
    #priceData[["vol", "to"]] = priceData[["vol", "to"]].fillna(value=0)

    col_name, col_id = "vol", 2
    if method == "volume":
        col_name = "vol"
        col_id = 2
    elif method == "turnover":
        col_name = 'to'
        cod_id = 3

    # Count of Non_zero
    b_count = priceData[col_name].fillna(0).astype(bool).sum()
    b_sum = priceData[col_name].sum()
    b_size = b_sum/b_count * bar_size


    priceDataV = priceData.loc[:, ["TotalVolume", "Turnover", "vol", "to", "hCount"]].values
    bar_label = [[0, 0]] #First: 1 or 0 indicates end of a bar, Second: Bar size
    last_index = 0
    csum = 0
    #print(priceDataV)
    for index, row in enumerate(priceDataV):
        if index == 0:
            continue
        """
        if row[col_id] >= b_size: #if current value > b_size
            if bar_label
        """
        csum += row[col_id]
        if priceDataV[last_index][4] != row[4]:
            bar_label[-1] = [1, csum - row[col_id]]
            csum = row[col_id]

        if csum >= b_size: #Current tick as end of bar
            bar_label.append([1, csum])
            csum = 0
        else:
            bar_label.append([0, 0])


        last_index = index

    bar_label = np.array(bar_label)
    priceData["bar_label"], priceData["bar_size"] = bar_label[:, 0], bar_label[:, 1]
    print(priceData["vol"].astype(float).nlargest(40))
    tmp = priceData["bar_size"].values
    tmp = list(filter(lambda  a: a != 0, tmp))
    #priceData["bar_size"].hist(grid=True, bins=40)
    plt.hist(tmp, bins = 100)
    plt.show()
    return priceData, b_size

# ["BidTT", "AskTT"]
def get_trade_through(priceData, multiplier, vol_side):
    if priceData.index.values[0] != 0:
        priceData.reset_index(inplace=True)
    if "vwap" not in priceData.columns:
        # added ["volume", "turnover", "vwap"]
        if "volume" in priceData.columns:
            del priceData["volume"]
        if "turnover" in priceData.columns:
            del priceData["turnover"]
        get_vwap(priceData, multiplier, vol_side)

    def check_trade_through(row):
        att, btt = 0, 0
        if row["AskPrice1"] > row["LastAsk"] and row["vwap"] > row["LastBid"]:
            att = 1
        if row["BidPrice1"] < row["LastBid"] and row["vwap"] < row["LastAsk"]:
            btt = 1
        return btt, att

    priceData["LastBid"] = priceData["BidPrice1"].shift()
    priceData["LastAsk"] = priceData["AskPrice1"].shift()
    priceData["BidTT"], priceData["AskTT"] = zip(*priceData.apply(check_trade_through, axis = 1))
    priceData.drop(["LastBid","LastAsk"], axis=1, inplace=True)
    return priceData.loc[:, ["BidTT", "AskTT"]]

# ["BidSN", "AskSN"]
def get_sprd_narrow(priceData):
    if priceData.index.values[0] != 0:
        priceData.reset_index(inplace=True)
    priceData["LastBid"] = priceData["BidPrice1"].shift()
    priceData["LastAsk"] = priceData["AskPrice1"].shift()
    def check_sprd_narrow(row):
        bsn, asn = 0, 0                       # Only care about independent side
        if row["AskPrice1"] < row["LastAsk"] and row["BidPrice1"] >= row["LastBid"]:
            asn = 1
        if row["BidPrice1"] > row["LastBid"] and row["AskPrice1"] <= row["LastAsk"]:
            bsn = 1
        return bsn, asn

    priceData["BidSN"], priceData["AskSN"] = zip(*priceData.apply(check_sprd_narrow, axis = 1))
    priceData.drop(["LastBid", "LastAsk"], axis=1, inplace=True)
    return priceData.loc[:, ["BidSN", "AskSN"]]

# ["BidTR", "AskTR"]: simply method to determine trade on best bid/ask
def get_bid_ask_trade(priceData, multiplier, vol_side):
    if priceData.index.values[0] != 0:
        priceData.reset_index(inplace=True)

    if "vwap" not in priceData.columns:
        # added ["volume", "turnover", "vwap"]
        if "volume" in priceData.columns:
            del priceData["volume"]
        if "turnover" in priceData.columns:
            del priceData["turnover"]
        get_vwap(priceData, multiplier, vol_side)

    def check_best_quote_trade(row):
        btr, atr = 0, 0
        if row["volume"] == 0:
            btr, atr = 0, 0
        elif row["vwap"] <= row["LastBid"]:
            btr = 1
        elif row["vwap"] >= row["LastAsk"]:
            atr = 1
        else:
            btr, atr = 1, 1
        return btr, atr

    priceData["LastBid"] = priceData["BidPrice1"].shift()
    priceData["LastAsk"] = priceData["AskPrice1"].shift()

    priceData["BidTR"], priceData["AskTR"] = zip(*priceData.apply(check_best_quote_trade, axis = 1))
    priceData.drop(["LastBid", "LastAsk"], axis=1, inplace=True)

    return priceData.loc[:, ["BidTR", "AskTR"]]

# ["volume", "turnover", "vwap"]
def get_vwap(priceData, multiplier, vol_side):
    if priceData.index.values[0] != 0:
        priceData.reset_index(inplace=True)

    priceData["volume"] = priceData["TotalVolume"].diff() / vol_side
    priceData["turnover"] = priceData["Turnover"].diff() / multiplier / vol_side
    priceData["volume"] = priceData["volume"].clip(0)
    priceData["turnover"] = priceData["turnover"].clip(0)

    def get_vwap(row):
        if row["volume"] == 0:
            return np.nan
        else:
            return row["turnover"]/row["volume"]

    priceData["vwap"] = priceData.apply(get_vwap, axis = 1)
    priceData.loc[0, "vwap"] = (priceData.at[0, "BidPrice1"] + priceData.at[0, "AskPrice1"])/2
    priceData["vwap"] = priceData["vwap"].fillna(method="ffill")
    priceData["vwap"] = priceData["vwap"].round(3)
    priceData["volume"] = priceData["volume"].fillna(0)
    priceData["turnover"] = priceData["turnover"].fillna(0)
    return priceData.loc[:, ["volume", "turnover", "vwap"]]


###########################################ORIGINAL#####################################
# Interpret trading volume directions between two ticks
"""
Record specific trade volumes at bid and ask
"""
def get_volume_ba(priceData, min_step):
    trimmed_names = ["AskPrice1", "AskSize1", "AskPrice2", "AskSize2",
                     "AskPrice3", "AskSize3", "AskPrice4", "AskSize4",
                     "AskPrice5", "AskSize5", "BidPrice1", "BidSize1",
                     "BidPrice2", "BidSize2", "BidPrice3", "BidSize3",
                     "BidPrice4", "BidSize4", "BidPrice5", "BidSize5",
                     "TimeStamp", "LastPrice", "hCount", "dCount",
                     "volume", "turnover", "vwap"]
    last_tick, this_tick = None, None

    dfV = priceData.values
    orders_sep = [[0, 0] for i in range(len(dfV))]
    tick_count = 0
    for index, row in enumerate(dfV):
        if tick_count == 0:
            last_tick = dict(zip(trimmed_names, row))
            tick_count += 1
            continue
        elif tick_count >= 1:
            this_tick = dict(zip(trimmed_names, row))

        if this_tick["volume"] == 0:
            continue

        trade_record = analyze_between_tick(last_tick, this_tick, min_step)

        bid_volume, ask_volume = 0, 0
        for price, volume in trade_record.items():
            if price >= last_tick["AskPrice1"]:
                ask_volume += volume
            elif price <= last_tick["BidPrice1"]:
                bid_volume += volume
            else:
                if this_tick["vwap"] > last_tick["vwap"]:
                    ask_volume += volume
                elif this_tick["vwap"] < last_tick["vwap"]:
                    bid_volume += volume
                else:
                    ask_volume += volume/2
                    bid_volume += volume/2

        orders_sep[index] = [bid_volume, ask_volume]
    return pd.DataFrame(orders_sep, columns=["BidVolume", "AskVolume"])
"""
Record willingness to cross the spread by market participants
"""
def get_spread_crossed(priceData, min_step):
    trimmed_names = ["AskPrice1", "AskSize1", "AskPrice2", "AskSize2",
                     "AskPrice3", "AskSize3", "AskPrice4", "AskSize4",
                     "AskPrice5", "AskSize5", "BidPrice1", "BidSize1",
                     "BidPrice2", "BidSize2", "BidPrice3", "BidSize3",
                     "BidPrice4", "BidSize4", "BidPrice5", "BidSize5",
                     "TimeStamp", "LastPrice", "hCount", "dCount",
                     "volume", "turnover", "vwap"]
    last_tick, this_tick = None, None

    dfV = priceData.values
    orders_sep = [[0, 0] for i in range(len(dfV))]
    tick_count = 0
    for index, row in enumerate(dfV):
        if tick_count == 0:
            last_tick = dict(zip(trimmed_names, row))
            tick_count += 1
            continue
        elif tick_count >= 1:
            this_tick = dict(zip(trimmed_names, row))

        if this_tick["volume"] == 0:
            continue

        trade_record = analyze_between_tick(last_tick, this_tick, min_step)

        bid_volume, ask_volume = 0, 0
        for price, volume in trade_record.items():
            if price >= last_tick["AskPrice1"]:
                ask_volume += volume * int((price - last_tick["BidPrice1"])/min_step)
            elif price <= last_tick["BidPrice1"]:
                bid_volume += volume * int((last_tick["AskPrice1"] - price)/min_step)


        orders_sep[index] = [bid_volume, ask_volume]

    return pd.DataFrame(orders_sep, columns=["BidSprdCr", "AskSprdCr"])






