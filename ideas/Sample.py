import numpy as np
import pandas as pd
import datetime as dt
import sys
sys.path.append("/home/dxz/Code")
import math
from hft_rsys import *
import matplotlib.pyplot as plt
import copy
import Helper, Microstructure_Factors


from Microstructure_Factors import *

if __name__ == "__main__":
    start_time = datetime(2019, 6, 15, 9, 0, 0)
    end_time = datetime(2019, 7, 16, 9, 0, 0)
    contract = "j"
    month = "1909"
    source = "l2_dce"
    # Create instance of RS just like before
    hrb = HRB.HRB(start_time, end_time, contract, month, source, 0)

    contract = hrb.tInfo.fSymbol + hrb.tInfo.fContract
    contract_info = hrb.get_contract_data()

    # Needed to be update
    min_step = contract_info.step
    multiplier = contract_info.multiplier

    feetype = contract_info.feetype
    fee = contract_info.fee
    vol_side = 1 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 2

    df = hrb.get_hft_data()
    df["hCount"] = hrb.get_hCount()
    df["dCount"] = hrb.get_dCount()
    Helper.get_vwap(df, multiplier, vol_side)

    trimmed_names = ["AskPrice1", "AskSize1", "AskPrice2", "AskSize2",
                     "AskPrice3", "AskSize3", "AskPrice4", "AskSize4",
                     "AskPrice5", "AskSize5", "BidPrice1", "BidSize1",
                     "BidPrice2", "BidSize2", "BidPrice3", "BidSize3",
                     "BidPrice4", "BidSize4", "BidPrice5", "BidSize5",
                     "TimeStamp", "LastPrice", "hCount", "dCount",
                     "volume", "turnover", "vwap"]

    df = df[trimmed_names]

"""
Here we want to explore different ways to separate volumes
"""

def gen_volume_1(df, min_step):
    last_tick, this_tick = None, None
    new_order_list = {}
    count = 0
    for row in self.bDataV:
        if self.tick_count == 0:
            last_tick = dict(zip(self.trimmed_names, row))
            self.tick_count += 1
            continue
        elif self.tick_count >= 1:
            this_tick = dict(zip(self.trimmed_names, row))