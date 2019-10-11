import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
from hft_rsys import *
import Backtest_System
import Helper
import copy

EPSILON = 0.000001

class My_Strategy(Backtest_System.Strategy):
    #####################################################################
    #                Custom variables defined by user here
    #####################################################################
    def init_variables(self):
        # This dictionary record time when order was sent, index was stored
        self.orders_place_time = dict()
        self.last_order_time = -1
        self.last_signal_index = 0

        # For signals:

        self.sprd = ((self.bData["AskPrice1"] - self.bData["BidPrice1"])/self.min_step).round()

        self.last_order_direction = 0
        self.last_order_price = 0.0
        self.counter = 0
        self.last_order_done = False
        self.tick_entries = []
        self.barrier_order_entries = dict()


        # params for index_future_barrier_break
        self.signal_count = 0
        self.initial_orders = {}

        # rolling min for mid price:
        self.roll_mid_min = self.bData["MidPrice"].rolling(10).min().shift().values
        self.roll_mid_max = self.bData["MidPrice"].rolling(10).max().shift().values

        self.signal_list = np.full(len(self.bData.index), "NAN")
        self.response_list = np.full(len(self.bData.index), np.nan)
    #####################################################################
    #                Custom function defined by user here
    #####################################################################
    def strategy(self):
        self.index_future_barrier_break()
        #self.index_future_barrier_IF()

    def save_csv(self):
        pd.DataFrame(self.tick_entries,
                     columns=["Time", "Index", "Direction", "sprd", "volume", "bid_stdev", "ask_stdev",
                              "bid_price", "ask_price", "timestamp", "bid_widen", "ask_widen"]
                     ).to_csv("python_result_IC.csv")

    def index_future_barrier_break(self):
        if self.index < 120:
            return
        side = "NAN"
        tick1 = self.bData.iloc[self.index - 1]
        tick2 = self.bData.iloc[self.index]

        tick2_dict = {}
        price_columns = ["BidPrice5", "BidPrice4", "BidPrice3", "BidPrice2", "BidPrice1",
                         "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5"]
        volume_columns = ["BidSize5", "BidSize4", "BidSize3", "BidSize2", "BidSize1",
                          "AskSize1", "AskSize2", "AskSize3", "AskSize4", "AskSize5"]

        # build two dictionaries to map price by index
        for index, price_posit in enumerate(price_columns):
            if price_posit[:3] == "Bid":
                tick2_dict[tick2[price_posit]] = -1 * tick2[volume_columns[index]]
            else:
                tick2_dict[tick2[price_posit]] = tick2[volume_columns[index]]

        # Here we maintain positions first
        initial_orders_tmp = copy.deepcopy(self.initial_orders)
        for order_id, order in self.initial_orders.items():
            if order == None:
                if order_id in self.order_list:
                    order = self.order_list[order_id]
                    del initial_orders_tmp[order_id]

                else:
                    order = self.trade_record_on_tick[order_id]
                    initial_orders_tmp[order_id] = order

            if order.done_index != 0 and order.done_index + 5 < self.index:     #When order is done for at least 5 ticks
                if order.sign > 0 and self.bData.iloc[self.index]["MidPrice"] < self.roll_mid_min[self.index]: #stop buy orders
                    self.send_market_order(order.sign * -1 * 1)
                    del initial_orders_tmp[order_id]
                    print("Stop Loss")
                    print(self.index)
                elif order.sign < 0 and self.bData.iloc[self.index]["MidPrice"] > self.roll_mid_max[self.index]: #stop buy orders
                    self.send_market_order(order.sign * -1 * 1)
                    del initial_orders_tmp[order_id]
                    print("Stop Loss")
                    print(self.index)

        self.initial_orders = initial_orders_tmp


        # Then we check for initial signal if no initial signal and first order
        #print(self.barrier_order_entries)
        #print(self.initial_orders)
        if not self.barrier_order_entries and not self.initial_orders:         #For now, only send order when all orders are completed or cancelled
            if tick2["BidSize1"] > 25 and tick2["AskSize1"] < tick2["BidSize1"] / 4 and self.sprd[self.index] == 1:
                self.barrier_order_entries[self.signal_count] = [self.index, "buy", tick2["BidPrice1"], -1 * tick2["BidSize1"]]
                self.signal_count += 1

            elif tick2["AskSize1"] > 25 and tick2["BidSize1"] < tick2["AskSize1"] / 4 and self.sprd[self.index] == 1:
                self.barrier_order_entries[self.signal_count] = [self.index, "sell", tick2["AskPrice1"], tick2["AskSize1"]]
                self.signal_count += 1
            return

        # Followed signal with actual orders
        if not self.barrier_order_entries:
            return
        order_entries_tmp = copy.deepcopy(self.barrier_order_entries)
        for order_id, order_entry in self.barrier_order_entries.items():
            #initial order is done
            if order_entry[0] + 30 < self.index:  #time limit
                del order_entries_tmp[order_id]
            elif order_entry[2] not in tick2_dict:
                del order_entries_tmp[order_id]
            elif tick2_dict[order_entry[2]] * order_entry[3] < 0:
                del order_entries_tmp[order_id]
            elif order_entry[2] in tick2_dict and abs(tick2_dict[order_entry[2]]) < abs(0.3 * order_entry[3]):
                del order_entries_tmp[order_id]
                id = self.send_limit_order(order_entry[2], order_entry[3]/abs(order_entry[3]))
                self.initial_orders[id] = None
                print("####################################################################")
                print("Open")
                print(self.index)

        self.barrier_order_entries = order_entries_tmp

    def index_future_barrier_IF(self):
        if self.index < 120:
            return
        side = "NAN"
        tick1 = self.bData.iloc[self.index - 1]
        tick2 = self.bData.iloc[self.index]

        tick2_dict = {}
        price_columns = ["BidPrice5", "BidPrice4", "BidPrice3", "BidPrice2", "BidPrice1",
                         "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5"]
        volume_columns = ["BidSize5", "BidSize4", "BidSize3", "BidSize2", "BidSize1",
                          "AskSize1", "AskSize2", "AskSize3", "AskSize4", "AskSize5"]

        # build two dictionaries to map price by index
        for index, price_posit in enumerate(price_columns):
            if price_posit[:3] == "Bid":
                tick2_dict[tick2[price_posit]] = -1 * tick2[volume_columns[index]]
            else:
                tick2_dict[tick2[price_posit]] = tick2[volume_columns[index]]

        if not self.order_list and not self.trade_record_on_tick:         #For now, only send order when all orders are completed or cancelled
            if tick2["BidSize1"] > 25 and tick2["AskSize1"] < tick2["BidSize1"] / 7 and self.sprd[self.index] == 1:
                order_id1 = self.send_limit_order(tick2["AskPrice1"], 1)
                order_id2 = self.send_limit_order(round(tick2["AskPrice1"] + 4 * self.min_step, 1), -1)
                self.barrier_order_entries[order_id1] = [0, order_id2, 0, self.index, "buy", tick2["BidPrice1"], -1 * tick2["BidSize1"]]
                print("####################################################################")
                print("Open")
                print(self.index)
                self.signal_list[self.index] = 'bid'
                self.last_order_time = self.index
            elif tick2["AskSize1"] > 25 and tick2["BidSize1"] < tick2["AskSize1"] / 7 and self.sprd[self.index] == 1:
                order_id1 = self.send_limit_order(tick2["BidPrice1"], -1)
                order_id2 = self.send_limit_order(round(tick2["BidPrice1"] - 4 * self.min_step, 1), 1)
                self.barrier_order_entries[order_id1] = [0, order_id2, 0, self.index, "sell", tick2["AskPrice1"], tick2["AskSize1"]]
                print("####################################################################")
                print("Open")
                print(self.index)
                self.signal_list[self.index] = 'ask'
                self.last_order_time = self.index
            return

        if not self.barrier_order_entries:
            return
        order_entries_tmp = copy.deepcopy(self.barrier_order_entries)

        #if self.trade_record_on_tick: print(self.trade_record_on_tick)
        for order_id, order_entry in self.barrier_order_entries.items():
            if order_id not in self.order_list and order_entry[1] not in self.order_list and not self.trade_record_on_tick:
                del order_entries_tmp[order_id]
                continue
            if order_entries_tmp[order_id][0] == 0:    #if the order was not done last tick
                if order_id in self.trade_record_on_tick:
                    order_entries_tmp[order_id][0] = 1 #mark initial order as done
                else:                                  # Cancel order if initial order was not done
                    self.cancel_order_by_id(order_id)
                    self.cancel_order_by_id(order_entry[1])
                    del order_entries_tmp[order_id]
                    continue
            if order_entry[1] in self.trade_record_on_tick: # When followed order done
                if order_entries_tmp[order_id][0] != 1:                 # If initial order not done but followed order done
                    # cover it with market order for now
                    del order_entries_tmp[order_id]
                    self.cancel_order_by_id(order_id)
                    self.send_market_order(-1 * self.order_list[order_entry[1]].quantity)
                    print("Followed order done before initial order")
                    print(self.index)
                else:                                   # When both order done
                    order_entries_tmp[order_id][2] = 1 #mark followed order as done
                    del order_entries_tmp[order_id]
                    print("Both order completed")
                    print(self.index)
                    if self.last_order_time != -1:
                        self.response_list[self.last_order_time] = 1
                        self.last_order_time = -1
            elif order_entries_tmp[order_id][0] == 1:                   #initial order is done
                if order_entry[3] + 240 < self.index:  #time limit
                    del order_entries_tmp[order_id]
                    self.send_market_order(self.order_list[order_entry[1]].quantity)
                    self.cancel_order_by_id(order_entry[1])
                    print("Stop loss due to time limit")
                    print(self.index)
                    if self.last_order_time != -1:
                        self.response_list[self.last_order_time] = 0
                        self.last_order_time = -1
                elif order_entry[5] in tick2_dict and tick2_dict[order_entry[5]] * order_entry[6] < 0:
                    del order_entries_tmp[order_id]
                    if order_entry[1] in self.order_list:
                        self.send_market_order(self.order_list[order_entry[1]].quantity)
                    self.cancel_order_by_id(order_entry[1])
                    print("Stop loss due to price")
                    print(self.index)
                    if self.last_order_time != -1:
                        self.response_list[self.last_order_time] = 0
                        self.last_order_time = -1
                #elif order_entry[5] in tick2_dict and abs(tick2_dict[order_entry[5]]) > abs(order_entry[6]):
                #    order_entry[6] = tick2_dict[order_entry[5]]

                elif order_entry[5] in tick2_dict and abs(tick2_dict[order_entry[5]]) < abs(0.3 * order_entry[6]):
                    del order_entries_tmp[order_id]
                    if order_entry[1] in self.order_list:
                        self.send_market_order(self.order_list[order_entry[1]].quantity)
                    self.cancel_order_by_id(order_entry[1])
                    print("Stop loss due to price2")
                    print(self.index)
                    if self.last_order_time != -1:
                        self.response_list[self.last_order_time] = 0
                        self.last_order_time = -1

        self.barrier_order_entries = order_entries_tmp
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
    start_time = datetime(2019, 7, 20, 9, 0, 0)
    end_time = datetime(2019, 8, 18, 9, 0, 0)
    # Create instance of RS just like before
    hrb = HRB.HRB(start_time, end_time, "IF", "1908", "l2_cffex", 0)
    # Create instance of BT system with hrb as a param
    bt = Backtest_System.Backtester(hrb,  "ic_2019013", "normal")
    #bt.fee = 0.3

    strat = My_Strategy(bt)
    # Pass defined strategy to Backtest_System
    bt.run(strat)

    pd.DataFrame({"bid_or_ask": strat.signal_list,
                  "responses": strat.response_list
                  }).to_csv("signal_response_IC.csv")
    #strat.save_csv()

