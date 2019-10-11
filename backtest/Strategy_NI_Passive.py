import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
from hft_rsys import *
import sys

import Backtest_System
import time
import copy

EPSILON = 0.000001

class My_Strategy(Backtest_System.Strategy):
    #####################################################################
    #                Custom variables defined by user here
    #####################################################################
    def init_variables(self):
        self.signals = pd.read_csv("ni1911_20190826_trade_price.csv", index_col=1, header=0)

        # order1 is the last initial order, order2 is the cover order ID
        self.order_id1 = None
        self.order1 = None
        self.order_id2 = None
        self.order2 = None

        self.gap = False            #indicating data gap (signal)
        # Parameters:
        self.param1 = 4            # Minimum profit target

    #####################################################################
    #                Custom function defined by user here
    #####################################################################
    def strategy(self):
        #self.index_future_sprd_IC()
        #self.pp_sprd_widen()
        self.ni_sprd()
    def reset(self):
        # order1 is the last initial order, order2 is the cover order ID
        self.order_id1 = None
        self.order1 = None
        self.order_id2 = None
        self.order2 = None
    def ni_sprd(self):

        loc = self.index
        if self.index in self.signals.index:
            self.gap = False
            open_short, open_long, close_short, close_long = self.signals["Open_Short"].loc[self.index], \
                                                             self.signals["Open_Long"].loc[self.index], \
                                                             self.signals["Close_Short"].loc[self.index], \
                                                             self.signals["Close_Long"].loc[self.index]
        else:
            if self.gap == False:
                self.gap = True
                print("Signal not available @ " + str(self.index))
            self.clear_inventory()
            self.reset()
            return

        posit = self.trade_record_history[self.index].position

        #if self.index > 65:
        #    quit()
        # First: we deal with existing limit orders
        #print(self.index)
        #print("Exisiting orders: ")
        #print(self.order_list)

        # Second we process done orders
        if len(self.trade_record_on_tick) > 0:
            if posit > 0:
                # for simplicity, we cancel all orders when posit != 0
                self.cancel_all_orders()
                print("All order cancelled")

                if self.index in self.signals.index:  #opened long
                    for order_id, order in self.trade_record_on_tick.items():
                        self.order1 = self.completed_orders[order_id]
                        self.order_id1 = order_id
                        print("Initial order done @ " + str(self.index) + " price " + str(order.price))
                    self.order_id2 = self.send_limit_order(close_short, posit * -1)
                    print("Cover order sent @ " + str(self.index) + " price " + str(close_short))
                else:
                    print("Short cover signal not available")
                    self.clear_inventory()
                    self.reset()
            elif posit < 0:
                # for simplicity, we cancel all orders when posit != 0
                self.cancel_all_orders()
                if self.index in self.signals.index:  #opened short
                    for order_id, order in self.completed_orders.items():
                        self.order1 = self.completed_orders[order_id]
                        self.order_id1 = order_id
                        print("Initial order done @ " + str(self.index) + " price " + str(order.price))
                    else:
                        print("Error checking completed orders")
                    self.order_id2 = self.send_limit_order(close_long, posit * -1)
                    print("Cover order sent @ " + str(self.index))
                else:
                    print("Long cover signal not available")
                    self.clear_inventory()
                    self.reset()
            else: # reset if no position
                print("Cover order done @ " + str(self.index))
                self.reset()

        if posit != 0 and self.order1 is not None:
            if self.order1 is not None:
                #stop loss by checking time elapsed from last tick
                if (self.order1.sign > 0 and
                        close_short - self.order1.price < -2 * self.min_step) \
                    or (self.order1.sign < 0 and
                        close_long - self.order1.price > 2 * self.min_step):
                    print("Stop loss by loss @ " + str(self.index))
                    self.clear_inventory()
                    self.reset()
                    return

            if self.order2 is None:
                if self.order_id2 in self.order_list:
                    self.order2 = self.order_list[self.order_id2]
                else:
                    print("Could not find order2 @ " + str(self.index))
            else:           # time exceeded, so we replace with new cover order
                if self.order2.sent_index + 10 < self.index:
                    self.cancel_order_by_id(self.order_id2)
                    print("Cover order cancelled by time @ " + str(self.index) + " price " + str(self.order2.price))
                    if self.order2.sign < 0:
                        self.order_id2 = self.send_limit_order(close_short, self.order2.quantity)
                    else:
                        self.order_id2 = self.send_limit_order(close_long, self.order2.quantity)
                    print("New cover order sent @ " + str(self.index) + " price " + str(close_short))
                    self.order2 = None
        # We enter new position when none exist
        elif posit == 0:
            current_long_exist = False
            tmp_order_list = copy.deepcopy(self.order_list)

            for order_id, order in tmp_order_list.items():
                # only consider long position for now
                if order.sign > 0:
                    if order.price == open_long:
                        current_long_exist = True
                    # Cancel existing orders that can't satisfy minimum profit target
                    if round((close_short - order.price)/self.min_step) < self.param1:
                        print("Initial order cancelled by loss @ " + str(self.index) + " price " + str(order.price))
                        self.cancel_order_by_id(order_id)
                    """
                    elif round((close_short - order.price)/self.min_step) < self.param1 + 2 \
                            and (round((self.bData.iloc[loc]["AskPrice1"] - self.bData.iloc[loc]["BidPrice1"]) / self.min_step) >= 2 or
                                 round((self.bData.iloc[loc]["BidPrice1"] - self.bData.iloc[loc]["BidPrice2"]) / self.min_step) >= 2):
                        print("Initial order cancelled by loss @ " + str(self.index) + " price " + str(order.price))
                        self.cancel_order_by_id(order_id)
                    """
            if not current_long_exist:
                print("New Initial order sent @ " + str(self.index)  + " price " + str(open_long))
                self.order_id1 = self.send_limit_order(open_long, 1)




    #####################################################################
    #                Custom configuration set by user here
    #####################################################################
    def set_config(self):
        # Positions covered and orders cancelled 120 ticks before EOD
        self.cancel_all_end = True
        # Include fee in pnl calculation
        self.include_fee = False
        # Show orders direction in plots
        self.show_orders = True
        # Show fee cost over time in plots
        self.show_fee = True
        # Show order cancellation in max drawdown plot
        self.show_cancel = True
        # Display plots at the end of backtest
        self.show_plot = True
    #####################################################################
    #                 Configuration End Here
    #####################################################################



if __name__ == "__main__":
    start_time = datetime(2019, 8, 23, 21, 0, 0)
    end_time = datetime(2019, 8, 26, 15, 0, 0)
    contract = "ni"
    month = "1911"
    source = "xele_l2"
    filled = True
    # Create instance of RS just like before
    # Here use the new MongoDB database
    hrb = HRB.HRB(start_time, end_time, contract, month, source, filled, True)
    # Create instance of BT system with hrb as a param
    start = time.time()
    bt = Backtest_System.Backtester(hrb,  "NI_Passive", "normal")
    #bt.fee = 0.6

    strat = My_Strategy(bt)
    # Pass defined strategy to Backtest_System
    bt.run(strat)
    #strat.save_csv()

    end = time.time()
    print("Time:")
    print(end-start)