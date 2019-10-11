import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import sys
import math
from hft_rsys import *
import matplotlib.pyplot as plt
import copy
from Helper import *
from observe_order_book import HRB

import pickle
from Microstructure_Factors import *



def get_fee_sprd_ratio():
    columns = ["BidPrice1", "AskPrice1", "MidPrice"]
    st = dt.datetime(2019, 5, 21)
    et = dt.datetime(2019, 8, 21)
    contracts = ["cu", "al", "zn", "pb", "ni", "sn", "au", "ag", "rb", "hc", "fu", "bu", "ru", "sp"]
    #contracts = ["IC", "IF", "IH"]
    source = "l1_ctp"
    months = ["0", "1"]
    result = [["contract", "month", "feetype", "fee", "sprd"
                  , "mean_fee per hand", "sprd_fee_ratio"]]
    for contract in contracts:
        for month in months:

            hrb = HRB.HRB(st, et, contract, month, source, 0)

            contract_info = hrb.get_contract_data()
            feetype = contract_info.feetype
            fee = contract_info.fee

            df = hrb.get_hft_data()
            df = df[df.AskPrice1 * df.BidPrice1 != 0]
            sprd = (df["AskPrice1"] - df["BidPrice1"]).mean()

            if contract_info.feetype == "vol":
                mean_fee = contract_info.fee / contract_info.multiplier
            else:
                mean_fee = (df["MidPrice"].mean()) * contract_info.fee / 10000
            entry = [contract, month, feetype, fee, sprd, mean_fee * contract_info.multiplier, sprd/mean_fee]
            print(entry)
            result.append(entry)

    pd.DataFrame(result
                 ).to_csv("sprd_fee_ratio_summary.csv")


def find_and_print():
    #columns = ["BidPrice1", "AskPrice1", "MidPrice"]
    st = dt.datetime(2019, 8, 25)
    et = dt.datetime(2019, 9, 2)
    contract = "IC"
    month = "0"
    # contracts = ["IC", "IF", "IH"]
    source = "l2_cffex"

    hrb = HRB.HRB(st, et, contract, month, source, 0)
    df = hrb.get_hft_data()
    df.reset_index(inplace=True, drop=True)
    get_SBO_order_1(df, hrb.get_contract_data().step)
    #df["MidPrice-5"] = df["MidPrice"].shift(10)
    #df["BigJump"] = np.where(np.abs(df["MidPrice-5"] - df["MidPrice"]) >= 10 * hrb.get_contract_data().step, "1", "0")

    #print(df.loc[df["BigJump"] == "1", ["BigJump"]])
    print(df.loc[df["narrow_sprd_count"] >= 0.5, ["narrow_sprd_count"]].iloc[::100, :])

def check_trading_code():
    start_time = datetime(2019, 9, 17, 9, 0, 0)
    end_time = datetime(2019, 9, 18, 9, 0, 0)
    hrb = HRB.HRB(start_time, end_time, "IF", "1910", "cffex_l2", True, True)
    df = hrb.get_hft_data()
    print(df.columns.values)
    df = df.loc[:, ["BidPrice1", "AskPrice1", "TimeStamp"]]

    hrb2 = HRB.HRB(start_time, end_time, "IC", "1910", "cffex_l2", True, True)
    df2 = hrb2.get_hft_data()
    df2 = df2.loc[:, ["BidPrice1", "AskPrice1"]]

    df["BidPrice2"], df["AskPrice2"] = df2["BidPrice1"], df2["AskPrice1"]

    def ts_to_dt(row):
        current_time = datetime.fromtimestamp(int(str(int(row["TimeStamp"]))[:10]))
        current_time = current_time + timedelta(
            milliseconds=int(str(int(row["TimeStamp"]))[10:13]))
        return current_time
    df["F1_ma"] = ((df["AskPrice1"] + df["BidPrice1"])/2).rolling(25).mean().shift()
    df["F2_ma"] = ((df["AskPrice2"] + df["BidPrice2"])/2).rolling(25).mean().shift()

    df["F1_std"] = ((df["AskPrice1"] + df["BidPrice1"])/2).rolling(25).std().shift()
    df["F2_std"] = ((df["AskPrice2"] + df["BidPrice2"])/2).rolling(25).std().shift()

    df["corr"] = ((df["AskPrice1"] + df["BidPrice1"])/2).rolling(25).corr((df["AskPrice2"] + df["BidPrice2"])/2).shift()
    df["dt"] = df.apply(ts_to_dt, axis = 1)
    df.to_csv("Sample.csv")

if __name__ == "__main__":
    # columns = ["BidPrice1", "AskPrice1", "MidPrice"]
    st = dt.datetime(2019, 8, 23, 21, 0, 0)
    et = dt.datetime(2019, 8, 26, 10, 0, 0)
    contract = "ni"
    month = "1911"
    # contracts = ["IC", "IF", "IH"]
    source = "xele_l2"

    hrb = HRB.HRB(st, et, contract, month, source, True, True)
    df = hrb.get_hft_data()
    df.reset_index(inplace=True, drop=True)
    contract_info = hrb.get_contract_data()
    get_vwap(df, contract_info.multiplier, 2)
    min_step = contract_info.step

    df["LastBid"] = df["BidPrice1"].shift(1)
    df["LastAsk"] = df["AskPrice1"].shift(1)

    def ts_to_dt(row):
        current_time = datetime.fromtimestamp(int(str(int(row["TimeStamp"]))[:10]))
        current_time = current_time + timedelta(
            milliseconds=int(str(int(row["TimeStamp"]))[10:13]))
        return current_time


    df["dt"] = df.apply(ts_to_dt, axis=1)
    #print(df["dt"])
    print(df.loc[(df["vwap"] <= df["LastBid"] - 2 * min_step) | (df["vwap"] <= df["LastBid"] - 2 * min_step), ["vwap"]])