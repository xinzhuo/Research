import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
from hft_rsys import *
import sys
sys.path.append("/home/dxz/Code")
import Helper
import Backtest_System
import time

EPSILON = 0.000001

class My_Strategy(Backtest_System.Strategy):
    #####################################################################
    #                Custom variables defined by user here
    #####################################################################
    def init_variables(self):
        # This dictionary record time when order was sent, index was stored
        self.orders_place_time = dict()
        self.last_order_time = -1

        # For vol_dir_ratio
        self.vol_dir_record = np.array([[0, 0]])
        self.last_signal_index = 0

        # For signals:
        self.bid_stdev = self.bData["BidPrice1"].rolling(20).std(ddof=0).shift().values
        self.ask_stdev = self.bData["AskPrice1"].rolling(20).std(ddof=0).shift().values

        self.ret_10 = (self.bData["MidPrice"] - self.bData["MidPrice"].shift(10))/self.min_step
        self.sprd = ((self.bData["AskPrice1"] - self.bData["BidPrice1"])/self.min_step).round()


        self.mean_mid = self.bData["MidPrice"].rolling(5).mean().shift(-5).values
        self.last_order_direction = 0
        self.last_order_price = 0.0
        self.counter = 0
        self.last_order_done = False
        self.tick_entries = []

    #####################################################################
    #                Custom function defined by user here
    #####################################################################
    def strategy(self):
        #self.index_future_sprd_IC()
        #self.pp_sprd_widen()
        self.pp_gap()

    def pp_gap(self):
        loc = self.index
        # Record order time and cancel order after 20 secs

        cancel_id_list = []
        for order_id in self.order_list:
            if order_id not in self.orders_place_time:
                self.orders_place_time[order_id] = self.index - 1

            # Stop loss by time and loss
            if (self.orders_place_time[order_id] - self.index > 1200) \
                    or (
                    self.order_list[order_id].sign > 0 and (
                    self.bData.iloc[loc]["AskPrice1"] - self.order_list[order_id].price) > 10 * self.min_step) \
                    or (
                    self.order_list[order_id].sign < 0 and (
                    self.bData.iloc[loc]["BidPrice1"] - self.order_list[order_id].price) < -10 * self.min_step): #1
                self.send_market_order(self.order_list[order_id].quantity)
                cancel_id_list.append(order_id)
                del self.orders_place_time[order_id]

        for order_id in cancel_id_list:
            self.cancel_order_by_id(order_id)


        # Sample strategy here
        if self.bData.iloc[loc]["AskPrice2"] - self.bData.iloc[loc]["AskPrice1"] > 1 * self.min_step + EPSILON: #2
            if self.index - self.last_order_time >= 10:                                                         #3
                if self.bData.iloc[loc]["AskSize1"] <= 11:                                                      #4
                    self.send_limit_order(self.bData.iloc[loc]["AskPrice1"], 1)  #5self.bData.iloc[loc]["AskSize1"]
                    self.send_limit_order(self.bData.iloc[loc]["AskPrice1"] + self.min_step * 1, -1)
                    self.last_order_time = self.index
                    #print(self.index)

        elif self.bData.iloc[loc]["BidPrice1"] - self.bData.iloc[loc]["BidPrice2"] > 1* self.min_step + EPSILON:
            if self.index - self.last_order_time >= 10:
                if self.bData.iloc[loc]["BidSize1"] <= 11:
                    self.send_limit_order(self.bData.iloc[loc]["BidPrice1"], -1)
                    self.send_limit_order(self.bData.iloc[loc]["BidPrice1"] - self.min_step * 1,1 )
                    self.last_order_time = self.index
                    #print(self.index)

    def save_csv(self):
        pd.DataFrame(self.tick_entries,
                     columns=["Time", "Index", "Direction", "sprd", "volume", "bid_stdev", "ask_stdev",
                              "bid_price", "ask_price", "timestamp", "bid_widen", "ask_widen"]
                     ).to_csv("python_result_IC.csv")


    def index_future_sprd_IF(self):
        if self.index < 120:
            return
        side = "NAN"
        tick1 = self.bData.iloc[self.index - 1]
        tick2 = self.bData.iloc[self.index]

        if self.last_order_time != -1: # order exist:
            if self.last_order_time + 1 == self.index: # check next tick
                if not self.order_list: # empty order_list -> last order done
                    print("Order Done: " + str(self.index))
                    if self.last_order_direction == 1:
                        self.send_force_order(self.mean_mid[self.index], -1)
                    elif self.last_order_direction == -1:
                        self.send_force_order(self.mean_mid[self.index], 1)
                else:                   # order not done
                    print("Order not Done: " + str(self.index))
                    self.cancel_all_orders()
            elif self.last_order_time + 5 < self.index:
                self.last_order_time = -1
                self.last_order_direction = 0

        else:
            if round((tick1["BidPrice1"] - tick2["BidPrice1"])/self.min_step) >= 4 \
                    and tick2["AskPrice1"] == tick1["AskPrice1"]:
                if tick2["volume"] <= 10.5 and self.sprd[self.index] > 5.5 and \
                        self.ret_10[self.index] > -3.75 and self.ask_stdev[self.index] <= 2.363 * self.min_step:
                    new_order_id = self.send_limit_order(round(tick2["BidPrice1"] + self.min_step, 1), 1)


                    side = "buy"
                    self.counter += 1
                    self.last_order_time = self.index
                    self.last_order_direction = 1

            elif round((tick2["AskPrice1"] - tick1["AskPrice1"])/self.min_step) >= 4 \
                    and tick2["BidPrice1"] == tick1["BidPrice1"]:
                if tick2["volume"] <= 10.5 and self.sprd[self.index] > 5.5 and \
                        self.ret_10[self.index] < 3.75 and self.bid_stdev[self.index] <= 2.363 * self.min_step:
                    new_order_id = self.send_limit_order(round(tick2["AskPrice1"] - self.min_step, 1), -1)
                    #print(self.sprd[self.index])
                    side = "sell"
                    self.counter += 1
                    self.last_order_time = self.index
                    self.last_order_direction = -1

        #if side != "NAN":
        if self.sprd[self.index] - self.sprd[self.index - 1] >= 4 \
                and (tick1["BidPrice1"] == tick2["BidPrice1"] or tick1["AskPrice1"] == tick2["AskPrice1"]):
            if not tick2["TimeStamp"]:
                return "NA"
            ts_str = '{:f}'.format(tick2["TimeStamp"])
            current_time = datetime.fromtimestamp(int(str(ts_str)[:10]))
            current_time = current_time + timedelta(
                milliseconds=int(str(ts_str)[10:13]))
            self.tick_entries.append(
                [str(current_time), str(self.index + 1), side, str(self.sprd[self.index]),
                 str(tick2["volume"]), self.bid_stdev[self.index], self.ask_stdev[self.index],
                 str(tick2["BidPrice1"]), str(tick2["AskPrice1"]),
                 str(tick2["TimeStamp"]), str(round((tick1["BidPrice1"]-tick2["BidPrice1"])/self.min_step)),
                 str(round((tick2["AskPrice1"]-tick1["AskPrice1"])/self.min_step))])

    #####################################################################
    #                Custom configuration set by user here
    #####################################################################
    def set_config(self):
        # Positions covered and orders cancelled 120 ticks before EOD
        self.cancel_all_end = True
        # Include fee in pnl calculation
        self.include_fee = True
        # Show orders direction in plots
        self.show_orders = True
        # Show fee cost over time in plots
        self.show_fee = True
        # Show order cancellation in max drawdown plot
        self.show_cancel = True

        self.show_plot = False
    #####################################################################
    #                 Configuration End Here
    #####################################################################



if __name__ == "__main__":
    start_time = datetime(2019, 6, 15, 9, 0, 0)
    end_time = datetime(2019, 7, 16, 9, 0, 0)
    # Create instance of RS just like before
    hrb = HRB.HRB(start_time, end_time, "j", "1909", "l2_dce", 0)
    # Create instance of BT system with hrb as a param
    start = time.time()
    bt = Backtest_System.Backtester(hrb,  "ic_2019013", "normal")
    bt.fee = 0.6

    strat = My_Strategy(bt)
    # Pass defined strategy to Backtest_System
    bt.run(strat)
    #strat.save_csv()

    end = time.time()
    print("Time:")
    print(end-start)