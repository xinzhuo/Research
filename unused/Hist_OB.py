import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from collections import deque
from hft_rsys import *

class Cylinder:
    def __init__(self, n_tick, min_tick, low_limit, high_limit, ob_depth = 5):
        self.n_tick = n_tick
        self.n_level = int(round((high_limit - low_limit) / min_tick)) + 1
        self.min_tick_div = 1.0 / min_tick
        self.ob_depth = ob_depth

        self.low_limit = low_limit

        self.record = [deque([None] * n_tick) for i in range(self.n_level)]
        self.last_row = [None] * self.n_level

        self.non_zero_left = self.n_level - 1
        self.non_zero_right = 0

        # Track last tick's best bid and ask
        self.last_bid_1, self.last_ask_1 = None, None

    def update(self, tick):

        for i in range(5, 0, -1):
            bid_posit = int(round((tick["BidPrice" + str(i)] - self.low_limit) * self.min_tick_div))
            ask_posit = int(round((tick["AskPrice" + str(i)] - self.low_limit) * self.min_tick_div))

            if i == 5:
                self.last_row[bid_posit: ask_posit + 1] = [0] * (ask_posit - bid_posit + 1)
                self.non_zero_left = min(self.non_zero_left, bid_posit)
                self.non_zero_right = max(self.non_zero_right, ask_posit)

            self.last_row[bid_posit] = -1 * tick["BidSize" + str(i)]
            self.last_row[ask_posit] = tick["AskSize" + str(i)]

        # Here we check if large move can result in ask volume on the left of bid volume and vice versa
        if self.last_bid_1:
            if tick["BidPrice5"] > self.last_ask_1:
                # Start to replace bad value with None, until negative value is reached
                posit = int(round((self.last_ask_1 - self.low_limit) * self.min_tick_div))
                print(self.last_row[self.non_zero_left: self.non_zero_right + 1])
                while self.last_row[posit] is None or self.last_row[posit] >= 0:
                    self.last_row[posit] = None
                    posit += 1
                print(self.last_row[self.non_zero_left: self.non_zero_right + 1])
            elif tick["AskPrice5"] < self.last_bid_1:
                posit = int(round((self.last_bid_1 - self.low_limit) * self.min_tick_div))
                print(self.last_row[self.non_zero_left: self.non_zero_right + 1])
                while self.last_row[posit] is None or self.last_row[posit] <= 0:
                    self.last_row[posit] = None
                    posit -= 1
                print(self.last_row[self.non_zero_left: self.non_zero_right + 1])

        self.last_bid_1 = tick["BidPrice1"]
        self.last_ask_1 = tick["AskPrice1"]

        for i in range(self.non_zero_left, self.non_zero_right + 1):
            self.record[i].popleft()
            self.record[i].append(self.last_row[i])



    def retrieve(self, price):
        posit = int(round((price - self.low_limit) * self.min_tick_div))
        return self.record[posit]



if __name__ == '__main__':
    st = datetime(2019, 3, 19)
    et = datetime(2019, 3, 30)
    hrb = HRB.HRB(st, et, 'pp', '1905', 'l2_dce', 0)

    priceData = hrb.get_hft_data()
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step

    n_tick = 10 # Set your own here

    priceData["hCount"] = hrb.get_hCount()
    priceData.reset_index(inplace=True)
    cy = None

    last_row = None
    for index, row in priceData.iterrows():
        if index == 0:
            cy = Cylinder(n_tick, min_tick, row["FallLimit"], row["RiseLimit"])
        elif last_row and last_row["hCount"] != row["hCount"]:
            print("Reset Data Structure for new trading day")
            cy = Cylinder(n_tick, min_tick, row["FallLimit"], row["RiseLimit"])

        cy.update(row)
        last_row = row
        #print(row["BidPrice1"])
        #print(cy.last_row)
        print(cy.retrieve(row["BidPrice1"]))




