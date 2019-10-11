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

"""Misc:
Many factors below are calculated under three options:
1. Compute regardless any condition         - "none"
    - Available for all columns
2. Compute correspond to bid/ask column     - "ba"
    - Result based on bid, ask
    - NA if "ba" is NA
3. Compute correspond to a specifies column - "col"
"""
class MSF_Library():
    def __init__(self, hrb):
        contract_info = hrb.get_contract_data()
        self.multiplier = contract_info.multiplier
        self.min_tick = contract_info.step
        self.vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1

        self.df = hrb.get_hft_data()
        def get_fake_bid_ask(row):
            if row["BidSize1"] % 2 == 1:
                return "bid"
            else:
                return "ask"

        self.df["bid_or_ask"] = self.df.apply(get_fake_bid_ask, axis=1)
        if self.df.index.values[0] != 0:
            self.df.reset_index(inplace=True)

        column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                        "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                        "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                        "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                        "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                        "volume", "turnover", "TimeStamp", "hCount",
                        "FallLimit", "RiseLimit", "TotalVolume", "Turnover",
                        "MidPrice", "bid_or_ask"]
        self.df["hCount"] = hrb.get_hCount()
        self.df = self.df.loc[:, column_names]

        self.condition = "ba"
        self.period = 10

    def get_all(self):
        get_vwap_position(self.df, self.multiplier, self.vol_side, self.condition)
        get_ret_direction(self.df, self.min_tick, self.period, self.condition, "side")
        # more ret_direction should be added
        get_sprd_ma(self.df, self.min_tick, self.period)
        get_sprd_stdev(self.df, self.min_tick, self.period)

        get_volume_frac(self.df, self.period)

        get_lv1_distance(self.df, self.min_tick, self.condition)
        get_widen_distance(self.df, self.min_tick, self.condition, True)
        get_best_quote_ratio(self.df, self.condition)

        get_TT_mean(self.df, self.multiplier, self.vol_side, self.period, self.condition)
        get_SN_mean(self.df, self.period, self.condition)
        get_TR_mean(self.df, self.multiplier, self.vol_side, self.period, self.condition)

        get_price_stdev(self.df, self.min_tick, self.period, self.condition)

        print(self.df.columns)

def get_vwap_position(df, multiplier, vol_side, condition = "ba"):
    """Case 2: Return position of VWAP relative to bid and ask last tick
        -inf < VWAP < BidPrice1 <= VWAP <= AskPrice1 < VWAP < inf
    :param df:
    :param multiplier:
    :param vol_side:
    :param condition:
    :return: Label Encoded corresponding to about VWAP position
            bid direction:  1, 2, 3
            ask direction:  3, 2, 1
            both:           -1, 0, 1
    """
    if condition == "ba":                       # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

    if "vwap" not in df.columns:
        get_vwap(df, multiplier, vol_side)      # ["volume", "turnover", "vwap"] added to df

    def vwap_position(row):
        if row["volume"] == 0:
            return np.nan
        elif condition == "ba":
            if row["bid_or_ask"] == "bid":
                if row["vwap"] < row["BidPrice1"]:
                    return 1
                elif row["vwap"] > row["AskPrice1"]:
                    return 3
                else:
                    return 2
            elif row["bid_or_ask"] == "bid":
                if row["vwap"] < row["BidPrice1"]:
                    return 3
                elif row["vwap"] > row["AskPrice1"]:
                    return 1
                else:
                    return 2
            else:
                return np.nan
        else:                                   # Normal case
            if row["vwap"] < row["BidPrice1"]:
                return -1
            elif row["vwap"] > row["AskPrice1"]:
                return 0
            else:
                return 1

    df["vwap_position"] = df.apply(vwap_position, axis = 1)

    return df.loc[:, ["vwap_position"]]

def get_ret_direction(df, min_tick, period = 10, condition = "ba", type = "side"):
    """Case 5: Return status of selected return based on criteria
                For "ba", if "ask", positive return will be -1 and vice versa
    :param df:
    :param min_tick:
    :param period:
        -1 : return from t-2 to t-1
        n  : return from t-n to t
    :param condition:
        "ba": based on bid or ask
    :param type:
        tick: return tick return
        side: return -1, 0, 1
    :return: return in ticks
    """
    if condition == "ba":                       # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

    if period == -1:
        df["ret"] = (df["MidPrice"].shift(1) - df["MidPrice"].shift(2))*(1 / min_tick).round()
    elif period >= 0:
        df["ret"] = (df["MidPrice"] - df["MidPrice"].shift(period)) * (1 / min_tick).round()
    else:
        print("Select proper period")
        return None

    def ret_direction(row):
        if row["ret"] > 0:
            if condition == "ba":
                if row["bid_or_ask"] == "bid":
                    return 1 if type == "side" else row["ret"]
                elif row["bid_or_ask"] == "ask":
                    return -1 if type == "side" else -1 * row["ret"]
                else:
                    return np.nan
            else:
                return 1 if type == "side" else row["ret"]
        elif row["ret"] < 0:
            if condition == "ba":
                if row["bid_or_ask"] == "bid":
                    return -1 if type == "side" else row["ret"]
                elif row["bid_or_ask"] == "ask":
                    return 1 if type == "side" else -1 * row["ret"]
                else:
                    return np.nan
            else:
                return -1 if type == "side" else row["ret"]
        else:
            return 0

    df["ret_direction"] = df.apply(ret_direction, axis = 1)

    return df.loc[:, ["ret_direction"]]

def get_sprd(df, min_tick):
    """ To get spreads in ticks on each update

    :param df:
    :param min_tick:
    :return: ["sprd"]
    """
    if "sprd" not in df.columns:
        df["sprd"] = ((df["AskPrice1"] - df["BidPrice1"]) / min_tick).round()

    return df.loc[:, ["sprd"]]

def get_sprd_ma(df, min_tick, period):
    """ Get moving average of spread, exclusive of current update -Case 8
    :param df:
    :param min_tick:
    :param period:
    :return: ["sprd_ma"]
    """
    if "sprd" not in df.columns:
        get_sprd(df, min_tick)
    df["sprd_ma"] = df["sprd"].rolling(period).mean().shift()

    return df.loc[:, ["sprd_ma"]]

def get_sprd_stdev(df, min_tick, period):
    """ Get moving stdev of spread, exclusive of current update - Case 9
    :param df: DataFrame
    :param min_tick:
    :param period:
    :return: ["sprd_stdev"]
    """
    if "sprd" not in df.columns:
        get_sprd(df, min_tick)
    df["sprd_stdev"] = df["sprd"].rolling(period).std().shift()

    return df.loc[:, ["sprd_stdev"]]

def get_volume_frac(df, period = 10):
    """ Volume as a percentage sum of past n period volume - Case 10
    :param df: DataFrame
    :param period:
    :return:  ["volume_frac"]
    """
    if "volume" not in df.columns:
        print("Can not locate volume column")
        return None
    df["volume_frac"] = df["volume"]/(df["volume"].rolling(period).sum())

    return df.loc[:, ["volume_frac"]]

def get_lv1_distance(df, min_tick, condition = "ba"):
    """ Get distance between first and second level quotes in tick - Case 11
    :param df:
    :param min_tick:
    :param condition:
        "ba": same side, oppo side
        other: bid side, ask side
    :return:
        "ba":   ["base_lv1_distance", "oppo_lv1_distance"]
        other:  ["bid_lv1_distance", "ask_lv1_distance"]
    """

    if condition == "ba":                       # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

        def lv1_distance(row):
            if row["bid_or_ask"] == "bid":
                return round((row["BidPrice1"] - row["BidPrice2"])/min_tick), \
                       round((row["AskPrice2"] - row["AskPrice1"])/min_tick)
            elif row["bid_or_ask"] == "ask":
                return round((row["AskPrice2"] - row["AskPrice1"])/min_tick), \
                       round((row["BidPrice1"] - row["BidPrice2"])/min_tick)
            else:
                return np.nan, np.nan

        df["base_lv1_distance"], df["oppo_lv1_distance"] = zip(*df.apply(lv1_distance, axis = 1))
        return df.loc[:, ["base_lv1_distance", "oppo_lv1_distance"]]
    else:
        df["bid_lv1_distance"], df["ask_lv1_distance"] = ((df["BidPrice1"] - df["BidPrice2"])/min_tick).round(), \
                                                         ((df["AskPrice2"] - df["AskPrice1"])/min_tick).round()
        return df.loc[:, ["bid_lv1_distance", "ask_lv1_distance"]]

def get_widen_distance(df, min_tick, condition = "ba", zero_bound = True):
    """ Get widen distance of bid/ask side from last tick - Case 1c
        :param df:
        :param min_tick:
        :param condition:
            "ba": same side, oppo side
            other: bid side, ask side
        :param zero_bound: restrict negative distance
        :return:
            "ba":   ["base_widen_distance", "oppo_widen_distance"]
            other:  ["bid_widen_distance", "ask_widen_distance"]
        """

    if condition == "ba":  # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

        def widen_distance(row):
            if row["bid_or_ask"] == "bid":
                return round((row["LastBid"] - row["BidPrice1"]) / min_tick), \
                       round((row["AskPrice1"] - row["LastAsk"]) / min_tick)
            elif row["bid_or_ask"] == "ask":
                return round((row["AskPrice1"] - row["LastAsk"]) / min_tick), \
                       round((row["LastBid"] - row["BidPrice1"]) / min_tick)
            else:
                return np.nan, np.nan

        df["LastBid"], df["LastAsk"] = df["BidPrice1"].shift(), df["AskPrice1"].shift()
        df["base_widen_distance"], df["oppo_widen_distance"] = zip(*df.apply(widen_distance, axis=1))
        df.drop(["LastBid", "LastAsk"], axis=1, inplace=True)

        if zero_bound:
            df["base_widen_distance"] = df["base_widen_distance"].clip(0)
            df["oppo_widen_distance"] = df["oppo_widen_distance"].clip(0)

        return df.loc[:, ["base_widen_distance", "oppo_widen_distance"]]
    else:
        df["LastBid"], df["LastAsk"] = df["BidPrice1"].shift(), df["AskPrice1"].shift()
        df["bid_widen_distance"], df["ask_widen_distance"] = \
            ((df["LastBid"] - df["BidPrice1"]) / min_tick).round(), \
            ((df["AskPrice1"] - df["LastAsk"]) / min_tick).round()
        df.drop(["LastBid", "LastAsk"], axis=1, inplace=True)

        if zero_bound:
            df["bid_widen_distance"] = df["bid_widen_distance"].clip(0)
            df["ask_widen_distance"] = df["ask_widen_distance"].clip(0)

        return df.loc[:, ["bid_widen_distance", "ask_widen_distance"]]

def get_best_quote_ratio(df, condition = "ba"):
    """ Get best quote ratio based on bid_or_ask - Case 14
        return bid ratio if condition is not "ba"
    :param df:
    :return: ["best_quote_ratio"]
    """
    if condition == "ba":  # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

        def best_quote_ratio(row):
            if row["bid_or_ask"] == "bid":
                return row["BidSize1"]/(row["BidSize1"] + row["AskSize1"])
            elif row["bid_or_ask"] == "ask":
                return row["AskSize1"] / (row["BidSize1"] + row["AskSize1"])
            else:
                return np.nan

        df["best_quote_ratio"] = df.apply(best_quote_ratio, axis = 1)
    else:
        df["best_quote_ratio"] = df["BidSize1"]/(df["BidSize1"] + df["AskSize1"])
    return df.loc[:, ["best_quote_ratio"]]

def get_TT_mean(df, multiplier, vol_side, period = 10, condition = "ba"):
    """ Get the average number of TradeThrough per tick in given period - Case 15
        Uses get_trade_through in Helper.py
    :param df:
    :param multiplier:
    :param vol_side:
    :param period:      period to calculate mean
    :return:
    """
    if "BidTT" not in df.columns:
        get_trade_through(df, multiplier, vol_side)

    df["bid_TT_mean"], df["ask_TT_mean"] =  df["BidTT"].rolling(period).mean(), df["AskTT"].rolling(period).mean()

    if condition == "ba":  # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

        def TT_mean(row):
            if row["bid_or_ask"] == "bid":
                return row["bid_TT_mean"], row["ask_TT_mean"]
            elif row["bid_or_ask"] == "ask":
                return row["ask_TT_mean"], row["bid_TT_mean"]
            else:
                return np.nan, np.nan

        df["base_TT_mean"], df["oppo_TT_mean"] = zip(*df.apply(TT_mean, axis=1))
        df.drop(["bid_TT_mean", "ask_TT_mean"], axis=1, inplace=True)
        return df.loc[:, ["base_TT_mean", "oppo_TT_mean"]]
    else:
        return df.loc[:, ["bid_TT_mean", "ask_TT_mean"]]

def get_SN_mean(df, period = 10, condition = "ba"):
    """ Get the average number of Spread Narrow per tick in given period - Case 17
        Watch out here! : Make sure if you want to control the other side- check original code
    :param df:
    :param period:      period to calculate mean
    :return:
    """
    if "BidSN" not in df.columns:
        get_sprd_narrow(df)

    df["bid_SN_mean"], df["ask_SN_mean"] =  df["BidSN"].rolling(period).mean(), df["AskSN"].rolling(period).mean()

    if condition == "ba":  # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

        def SN_mean(row):
            if row["bid_or_ask"] == "bid":
                return row["bid_SN_mean"], row["ask_SN_mean"]
            elif row["bid_or_ask"] == "ask":
                return row["ask_SN_mean"], row["bid_SN_mean"]
            else:
                return np.nan, np.nan

        df["base_SN_mean"], df["oppo_SN_mean"] = zip(*df.apply(SN_mean, axis=1))
        df.drop(["bid_SN_mean", "ask_SN_mean"], axis=1, inplace=True)
        return df.loc[:, ["base_SN_mean", "oppo_SN_mean"]]
    else:
        return df.loc[:, ["bid_SN_mean", "ask_SN_mean"]]

def get_TR_mean(df, multiplier, vol_side, period = 10, condition = "ba"):
    """ Get the average number of trade per tick in given period on each side - Case 19
    :param df:
    :param period:      period to calculate mean
    :return:
    """
    if "BidTR" not in df.columns:
        get_bid_ask_trade(df, multiplier, vol_side)

    df["bid_TR_mean"], df["ask_TR_mean"] =  df["BidTR"].rolling(period).mean(), df["AskTR"].rolling(period).mean()
    if condition == "ba":  # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

        def TR_mean(row):
            if row["bid_or_ask"] == "bid":
                return row["bid_TR_mean"], row["ask_TR_mean"]
            elif row["bid_or_ask"] == "ask":
                return row["ask_TR_mean"], row["bid_TR_mean"]
            else:
                return np.nan, np.nan

        df["base_TR_mean"], df["oppo_TR_mean"] = zip(*df.apply(TR_mean, axis=1))
        df.drop(["bid_TR_mean", "ask_TR_mean"], axis=1, inplace=True)
        return df.loc[:, ["base_TR_mean", "oppo_TR_mean"]]
    else:
        return df.loc[:, ["bid_TR_mean", "ask_TR_mean"]]

def get_price_stdev(df, min_tick, period = 10, condition = "ba"):
    """ Get the stdev in ticks for base and oppo price in case of "ba"
        bid and ask otherwise
    :param df:
    :param period:
    :param condition:
    :return:
    """
    df["bid_price_stdev"], df["ask_price_stdev"] = \
        df["BidPrice1"].rolling(period).std()/min_tick, df["AskPrice1"].rolling(period).std()/min_tick

    if condition == "ba":  # Check if bid/ask specified in "ba" column
        if "bid_or_ask" not in df.columns:
            print("Please provide column (bid_or_ask)")
            return None

        def price_stdev(row):
            if row["bid_or_ask"] == "bid":
                return row["bid_price_stdev"], row["ask_price_stdev"]
            elif row["bid_or_ask"] == "ask":
                return row["ask_price_stdev"], row["bid_price_stdev"]
            else:
                return np.nan, np.nan

        df["base_price_stdev"], df["oppo_price_stdev"] = zip(*df.apply(price_stdev, axis=1))
        df.drop(["bid_price_stdev", "ask_price_stdev"], axis=1, inplace=True)
        return df.loc[:, ["base_price_stdev", "oppo_price_stdev"]]
    else:
        return df.loc[:, ["bid_price_stdev", "ask_price_stdev"]]

# For testing purpose
if __name__ == "__main__":
    number = str(5)
    Month = "190" + number
    with open(Month + '_test.pickle', 'rb') as handle:
        hrb = pickle.load(handle)
        """
        contract_info = hrb.get_contract_data()
        multiplier = contract_info.multiplier
        min_tick = contract_info.step
        vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1

        df = hrb.get_hft_data()
        df["sprd"] = ((df["AskPrice1"] - df["BidPrice1"]) / min_tick).round()
        def get_fake_bid_ask(row):
            if row["BidSize1"] % 2 == 1:
                return "bid"
            else:
                return "ask"

        df["bid_or_ask"] = df.apply(get_fake_bid_ask, axis = 1)
        if df.index.values[0] != 0:
            df.reset_index(inplace=True)
        column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                        "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                        "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                        "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                        "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                        "volume", "turnover", "TimeStamp", "hCount",
                        "FallLimit", "RiseLimit", "TotalVolume", "Turnover",
                        "MidPrice", "sprd", "bid_or_ask"]
        df["hCount"] = hrb.get_hCount()
        df = df.loc[:, column_names]
        print(df)
        print(get_price_stdev(df, min_tick))
        """
        msf = MSF_Library(hrb)
        msf.get_all()
        print(msf.df)