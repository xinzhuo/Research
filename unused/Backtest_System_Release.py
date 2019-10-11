import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as sp
from scipy.stats import norm
import os
from datetime import datetime, date, time, timedelta
from collections import OrderedDict
import math
import random
import copy
from hft_rsys import *
import seaborn as sns


class Strategy:
    def __init__(self, bt):
        self.name = bt.strat_name                        # Strategy Name
        self.hrb = bt.hrb                          # Instance of hrb
        self.bData = bt.bData                      # Data in form of DataFrame
        self.min_step = bt.min_step
        self.multiplier = bt.multiplier

        self.order_list = dict()                # Maintain incomplete orders
        self.new_order_list = dict()            # Maintain new orders, refreshed at every tick update
        self.cancel_all_eod = True              # At EOD, cancel all orders and cover all positions

        self.order_id_count = 0                 # Counter used to assign ID to orders
        self.cancelled_count = 0
        self.index = 0                          # Counter indicates the location of current tick

        self.trade_record_on_tick = dict()      # trade_record used to record trade at each tick update
        self.trade_record_history = [TickTradeRecord()]     # used to record trade at each tick update

        self.no_trade = False                   # indicating no trade period, controller
        self.init_variables()                   # User defined variables initialized here
        self.set_config()                       # User defined configurations

    def init_variables(self):
        print("No custom variable defined.")

    def strategy(self):
        print("Strategy not defined.")

    def set_config(self):
        # Positions covered and orders cancelled 120 ticks before EOD
        self.cancel_all_end = True

        self.include_fee = True
        self.show_orders = True
        self.show_fee = True
        self.show_cancel = True
        print("Config not defined. Using default settings.")

    def process_tick(self, loc):
        self.index = loc  # Do not alter!
        self.new_order_list = dict()  # reinitialize new order list for this tick

        # We don't trade the last 121 ticks of the data
        if self.index > len(self.bData.index) - 121:
            if self.index == len(self.bData.index) - 1:
                self.clear_inventory()
            return self.new_order_list, {}

        # Default is to cancel and cover all at EOD before 120 ticks
        if self.bData.iloc[self.index + 120]["hCount"] != self.bData.iloc[self.index]["hCount"] and self.cancel_all_eod:
            if not self.no_trade:
                print("End of Trading Session")
                self.clear_inventory()
                self.no_trade = True
            return self.new_order_list, {}
        else:
            if self.no_trade:
                self.no_trade = False

        self.strategy()
        return self.new_order_list, self.order_list  # Do not alter!

    def clear_inventory(self):
        print("Clearing all orders and positions.")
        # First cancel all orders
        self.cancel_all_orders()
        # Then cover all positions
        if self.trade_record_history[-1].position != 0:
            self.send_market_order(-1 * self.trade_record_history[-1].position)

    def cancel_all_orders(self):
        self.order_list = {}
        self.new_order_list = {}

    def update_order_list(self, orders):
        self.order_list = orders

    def send_FAK_order(self, price, quantity):
        # assume all orders have size 0, use -1 (sell) and +1 (buy) to indicate order direction
        self.new_order_list["Order__" + str(self.order_id_count)] = (
        price, int(math.copysign(1, quantity)), quantity, "FAK")
        # print("New order: Order__" + str(self.order_id_count) + " " + str(price) + " " + str(quantity))
        # increment order id count for the next order
        order_id = "Order__" + str(self.order_id_count)
        self.order_id_count = self.order_id_count + 1
        return order_id

    def send_FOK_order(self, price, quantity):
        # assume all orders have size 0, use -1 (sell) and +1 (buy) to indicate order direction
        self.new_order_list["Order__" + str(self.order_id_count)] = (
        price, int(math.copysign(1, quantity)), quantity, "FOK")
        # print("New order: Order__" + str(self.order_id_count) + " " + str(price) + " " + str(quantity))
        # increment order id count for the next order
        order_id = "Order__" + str(self.order_id_count)
        self.order_id_count = self.order_id_count + 1
        return order_id

    def send_market_order(self, quantity):
        # price = self.bData.iloc[self.index]["BidPrice1"] if quantity > 0 else self.bData.iloc[self.index]["AskPrice1"]
        self.new_order_list["Order__" + str(self.order_id_count)] = (
        0, int(math.copysign(1, quantity)), quantity, "market")
        # print("New order: Order__" + str(self.order_id_count) + " " + str(price) + " " + str(quantity))
        order_id = "Order__" + str(self.order_id_count)
        self.order_id_count = self.order_id_count + 1
        return order_id

    def send_limit_order(self, price, quantity):
        # assume all orders have size 0, use -1 (sell) and +1 (buy) to indicate order direction
        self.new_order_list["Order__" + str(self.order_id_count)] = (
        price, int(math.copysign(1, quantity)), quantity, "limit")
        # print("New order: Order__" + str(self.order_id_count) + " " + str(price) + " " + str(quantity))
        # increment order id count for the next order
        order_id = "Order__" + str(self.order_id_count)
        self.order_id_count = self.order_id_count + 1
        return order_id

    def cancel_order_by_id(self, id):
        if id in self.order_list:
            del self.order_list[id]
            print("Order id: " + id + " cancelled")
            self.cancelled_count += 1
        else:
            print("Order id not valid for cancellation")

    def cancel_order_by_price(self, price):
        cancelled_id_list = []
        order_list_temp = copy.deepcopy(self.order_list)
        for order_id, order in self.order_list.items():
            if order.price == price:
                del order_list_temp[order_id]
                print("Order id: " + order_id + " cancelled")
                cancelled_id_list.append(order_id)
                self.cancelled_count += 1
        self.order_list = order_list_temp
        return cancelled_id_list

class TickTradeRecord:
    def __init__(self, timestamp=0, position=0, cashflow=0, pnl=0, fee=0):
        self.position = position
        self.cashflow = cashflow
        self.pnl = pnl
        self.fee = fee
        self.order_dir = 0
        self.cancel_flag = 0

        self.timestamp = timestamp
        self.volume = 0

    def set_cancellation_flag(self):
        self.order_cancelled = 1

    # These are cumulative value
    def get_key_info(self):
        return self.position, self.cashflow, self.pnl, self.fee

    # Property here should match ones in compute_pnl_summary
    def to_dict(self):
        return {
            "TimeStamp": self.timestamp,
            "Position": self.position,
            "Cashflow": self.cashflow,
            "PnL": self.pnl,
            "Fee": self.fee,
            "Volume": self.volume,
            "OrderDir": self.order_dir,
            "CancelFlag": self.cancel_flag,
        }

    def __str__(self):
        return "Position: " + str(self.position) + ", cashflow: " + str(self.cashflow) \
               + ", pnl: " + str(self.pnl) + ", Fee: " + str(self.fee)

class Order:
    def __init__(self, price, cur_posit, queue_size, sign, quantity):
        self.price = price  # Price of the order
        self.cur_posit = cur_posit  # Position within the queue
        self.queue_size = queue_size  # Latest queue size
        self.sign = sign  # Direction of the order, -1 -> sell, +1 -> buy
        self.quantity = quantity

    def update(self, cur_posit, queue_size):
        self.cur_posit = cur_posit  # Position within the queue
        self.queue_size = queue_size  # Latest queue size

    def __str__(self):
        return str(self.price) + " " + str(self.cur_posit) + " " + str(self.queue_size) + " " + str(self.quantity)

    def __repr__(self):
        return str(self.price) + " " + str(self.cur_posit) + " " + str(self.queue_size) + " " + str(self.quantity)

class Backtester:
    def __init__(self, hrb, name, mode):
        self.hrb = hrb
        self.strat_name = name
        self.mode = mode
        self.bData = pd.DataFrame()
        self.order_list = dict()
        # trade_record used to record trade at each tick update
        self.trade_record_on_tick = dict()
        # (position, cashflow, PnL, fee)
        self.trade_record_history = [TickTradeRecord()]
        self.data_type = "l2"
        self.cancelled_count = 0
        self.trimmed_names = ["AskPrice1", "AskSize1", "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                              "AskPrice4", "AskSize4", "AskPrice5", "AskSize5", "BidPrice1", "BidSize1", "BidPrice2",
                              "BidSize2", "BidPrice3", "BidSize3", "BidPrice4", "BidSize4", "BidPrice5", "BidSize5",
                              "TotalVolume", "Turnover", "TimeStamp", "LastPrice", "hCount", "dCount"]
        self.trimmed_names_l1 = ["AskPrice1", "AskSize1", "BidPrice1", "BidSize1",
                                 "TotalVolume", "Turnover", "TimeStamp", "LastPrice", "hCount", "dCount"]
        self.load_data()

    def trim_df(self):
        if self.data_type == "l2":
            raise Exception("Please use l1 data")
            self.trimmed_names = self.trimmed_names
        elif self.data_type == "l1":
            self.trimmed_names = self.trimmed_names_l1
        else:
            print("Wrong data source")
            return
        self.bData = self.bData[self.trimmed_names]

    def load_data(self):
        self.data_type = self.hrb.tInfo.counter[:2]
        # Backtest should be default to use non-tick-filled data
        df = self.hrb.get_hft_data()
        df["hCount"] = self.hrb.get_hCount()
        df["dCount"] = self.hrb.get_dCount()

        self.bData = df
        self.contract = self.hrb.tInfo.fSymbol + self.hrb.tInfo.fContract
        contract_info = self.hrb.get_contract_data()

        # Needed to be update
        self.min_step = contract_info.step
        self.multiplier = contract_info.multiplier
        self.margin_ratio = contract_info.leverage

        self.feetype = contract_info.feetype
        self.fee = contract_info.fee
        print("Fee type: " + str(self.feetype))
        print("Fee: " + str(self.fee))

    def match_old_orders_l1(self, tick1, tick2):
        if len(self.order_list) == 0:
            return

        tick1_volume, tick2_volume = {}, {}
        price_columns = ["BidPrice1",
                         "AskPrice1"]
        volume_columns = ["BidSize1",
                          "AskSize1"]

        # build two dictionaries to map price by index
        for index, price in enumerate(price_columns):
            if price[:3] == "Bid":
                tick1_volume[tick1[price]] = int(-1 * tick1[volume_columns[index]])
                tick2_volume[tick2[price]] = int(-1 * tick2[volume_columns[index]])
            else:
                tick1_volume[tick1[price]] = int(tick1[volume_columns[index]])
                tick2_volume[tick2[price]] = int(tick2[volume_columns[index]])

        # Intepreted volume: key = price, value = volume
        traded_dict = self.analyze_between_tick_l1(tick1, tick2)
        order_list_temp = copy.deepcopy(self.order_list)

        """
        Three cases:
        Case a: order price in OB
        Case b: order price in OB range -> gap price
        Case c: order price out of OB scope
        """

        # we loop through orders in "order_list"
        for order_id, order in self.order_list.items():
            # Check if the current price can be done on the OB
            if (order.sign > 0 and order.price >= tick2["AskPrice1"]) or \
                    (order.sign < 0 and order.price <= tick2["BidPrice1"]):
                message = order_id + " done @ " + str(order.price) + " time: " + str(tick2["TimeStamp"]) + " Code--2"
                self.trade_record_on_tick[order_id] = order
                del order_list_temp[order_id]
                # print(message)
            # Case when order_price was initially at a "gap" in OB, b -> a, b, c
            elif order.cur_posit == 0 and order.queue_size == 0:
                order_done = False
                for price_traded in traded_dict:
                    # Consider the order done if price better than given order's was done
                    if (order.sign > 0 and price_traded <= order.price) or \
                            (order.sign < 0 and price_traded >= order.price):
                        message = order_id + " done @ " + str(order.price) + " time: " + str(
                            tick2["TimeStamp"]) + " Code--3"
                        self.trade_record_on_tick[order_id] = order
                        del order_list_temp[order_id]
                        # print(message)
                        order_done = True
                        break

                # if order was not done and price appear in tick2,
                if not order_done and order.price in tick2_volume:
                    # check if new limit order is in same direction as our order, update queue_size
                    if tick2_volume[order.price] * (order.sign * -1) > 0:
                        order_list_temp[order_id].queue_size = tick2_volume[order.price]
                    # Consider order done, if new limit order in the opposite direction
                    else:
                        message = order_id + " done @ " + str(order.price) + " time: " + str(
                            tick2["TimeStamp"]) + " Code--4"
                        self.trade_record_on_tick[order_id] = order
                        del order_list_temp[order_id]
                        # print(message)
            # Case when order_price was initially out of previous OB scopes, c->a, b, c
            elif order.queue_size is None:
                # When out of scope price is now in tick2 OB
                if order.price in tick2_volume:
                    order_list_temp[order_id].update(tick2_volume[order.price], tick2_volume[order.price])
                # When out of scope price is now a "gap" price in tick2 OB, meaning 0 limit orders presented
                elif tick2["BidPrice1"] < order.price < tick2["AskPrice1"]:
                    order_list_temp[order_id].update(0, 0)
            # Normal case when we need to update our position in queue based on interpreted volume
            else:
                # Deal with special case first
                # If normal order went out of scope, a->b, c
                # Out of scope order (can't be gap price) remains out of scope
                if order.price not in tick2_volume:
                    # If order price was among possible traded price, consider the order done
                    if min(tick1["BidPrice1"], tick2["BidPrice1"]) < order.price < max(tick1["AskPrice1"],
                                                                                       tick2["AskPrice1"]):
                        message = order_id + " done @ " + str(order.price) + " time: " + str(
                            tick2["TimeStamp"]) + " Code--5"
                        self.trade_record_on_tick[order_id] = order
                        del order_list_temp[order_id]
                        # print(message)
                    continue
                # If the bid/ask side flipped, consider the order done, a->a from now on
                elif order.price in tick1_volume and tick1_volume[order.price] * tick2_volume[order.price] < 0:
                    message = order_id + " done @ " + str(order.price) + " time: " + str(
                        tick2["TimeStamp"]) + " Code--6"
                    self.trade_record_on_tick[order_id] = order
                    del order_list_temp[order_id]
                    # print(message)
                    continue

                # Normal cases now
                traded_volume = traded_dict[order.price] if order.price in traded_dict else 0
                # if size < volume: consider all done or position in queue < volume
                if traded_volume > 0 and abs(order.cur_posit) < traded_volume:
                    message = order_id + " done @ " + str(order.price) + " time: " + str(
                        tick2["TimeStamp"]) + " Code--7"
                    self.trade_record_on_tick[order_id] = order
                    del order_list_temp[order_id]
                    # print(message)
                    continue

                net_volume = abs(order.queue_size) - traded_volume - abs(tick2_volume[order.price])
                # For simplicity, the assumption here is that newly placed order and cancelled order don't co-exist between two tick
                updated_posit = 0
                if net_volume <= 0:  # calculated newly placed limit order size
                    updated_posit = abs(order.cur_posit) - traded_volume  # >= 0  by definition
                else:  # calculated cancelled order size
                    # Determine where the cancelled order happened based on the position of our order
                    if abs(order.cur_posit) <= abs(order.queue_size) / 2 or self.mode == "harsh":
                        updated_posit = abs(order.cur_posit) - traded_volume
                    else:
                        updated_posit = abs(order.cur_posit) - traded_volume - net_volume

                # Bound by 0 and queue size
                if updated_posit < 0:
                    message = order_id + " done @ " + str(order.price) + " time: " + str(
                        tick2["TimeStamp"]) + " Code--8"
                    self.trade_record_on_tick[order_id] = order
                    del order_list_temp[order_id]
                    # print(message)
                else:
                    updated_posit = min(updated_posit, abs(tick2_volume[order.price])) * -1 * order.sign
                    order_list_temp[order_id].update(updated_posit, tick2_volume[order.price])

        self.order_list = order_list_temp

    def match_new_orders_l1(self, tick2, new_order_list):
        # new_order_list = { order_id: (price, sign, quantity, order_type)}
        for order_id, order_info in new_order_list.items():
            if order_info[-1] == "market":
                if order_info[1] > 0:
                    if tick2["AskPrice1"] != 0:
                        price = tick2["AskPrice1"]
                    else:
                        print("Price unavailable to market order: " + order_id)
                        continue
                else:
                    if tick2["AskPrice1"] != 0:
                        price = tick2["BidPrice1"]
                    else:
                        print("Price unavailable to market order: " + order_id)
                        continue
                message = order_id + " done @ " + str(price) + " time: " + str(tick2["TimeStamp"]) + " Code--9"
                self.trade_record_on_tick[order_id] = Order(price, None, None, order_info[1], order_info[2])
                continue

            # Check if the order can be done immediately on orderbook
            if (order_info[1] > 0 and order_info[0] >= tick2["AskPrice1"]) or \
                    (order_info[1] < 0 and order_info[0] <= tick2["BidPrice1"]):
                if order_info[1] > 0:
                    price = tick2["AskPrice1"]
                else:
                    price = tick2["BidPrice1"]
                self.trade_record_on_tick[order_id] = Order(price, None, None, order_info[1], order_info[2])

                if order_info[-1] == "FAK" or order_info[-1] == "FOK":
                    message = order_id + " done @ " + str(order_info[0]) + " time: " + str(
                        tick2["TimeStamp"]) + " Code--10"
                else:
                    message = order_id + " done @ " + str(order_info[0]) + " time: " + str(
                        tick2["TimeStamp"]) + " Code--1"
            # Need to update order_list otherwise
            else:
                if order_info[-1] == "FAK" or order_info[-1] == "FOK":
                    message = order_id + " FAK/FOK order cancelled"
                    continue
                tick2_volume = {}
                price_columns = ["BidPrice1",
                                 "AskPrice1"]
                volume_columns = ["BidSize1",
                                  "AskSize1"]
                # build a dict to map prices to OB sizes
                for index, price in enumerate(price_columns):
                    if price[:3] == "Bid":
                        tick2_volume[tick2[price]] = int(-1 * tick2[volume_columns[index]])
                    else:
                        tick2_volume[tick2[price]] = int(tick2[volume_columns[index]])

                # if order_price in the OB: we record [price, current_position, queue_size, sign]
                if order_info[0] in tick2_volume:
                    self.order_list[order_id] = \
                        Order(order_info[0], tick2_volume[order_info[0]], tick2_volume[order_info[0]], order_info[1],
                              order_info[2])
                # if order_price at a gap in OB: we record [price, 0, 0, sign]
                elif tick2["BidPrice1"] < order_info[0] < tick2["AskPrice1"]:
                    self.order_list[order_id] = Order(order_info[0], 0, 0, order_info[1], order_info[2])
                # if order_price out of current tick2 scope, we record [price, None, None, sign], we update when the price emerge in ob
                else:
                    self.order_list[order_id] = Order(order_info[0], None, None, order_info[1], order_info[2])

    def analyze_between_tick_l1(self, tick1, tick2):
        """
        :param tick1:
        :param tick2:
        :return: dict with price as key and volume as value
        """
        min_step = self.min_step
        multiplier = self.multiplier

        volume = (tick2["TotalVolume"] - tick1["TotalVolume"]) / 2
        cashVolume = ((tick2["Turnover"] - tick1["Turnover"]) / multiplier) / 2
        BidPrice = tick2["BidPrice1"]
        AskPrice = tick2["AskPrice1"]
        if BidPrice == 0 and AskPrice == 0:
            return {}
        elif BidPrice == 0:
            return {AskPrice: volume}
        elif AskPrice == 0:
            return {BidPrice: volume}
        # if only 2 non-zero volume prices
        # BidPrice1 * x1 + AskPrice1 * x2 = Turnover
        # x1 + x2 = volume
        a = np.array([[tick2["BidPrice1"], tick2["AskPrice1"]], [1, 1]])
        b = np.array([cashVolume, volume])
        x = np.linalg.solve(a, b)

        if x[0] < 0:
            x = [0, volume]
        elif x[1] < 0:
            x = [volume, 0]

        trade_record = {
            tick2["BidPrice1"]: int(round(x[0])),
            tick2["AskPrice1"]: int(round(x[1]))
        }

        return trade_record

    # This calculate PnL on tick based on cashflow, it assume the trades on tick don't affect PnL
    def build_tick_trade_record(self, tick2):
        trade_record_last = self.trade_record_history[-1].get_key_info()
        position_last, cashflow_last, pnl_last, fee_last = trade_record_last

        # Positive position use best bid to calculate value and vice versa
        if position_last == 0:
            pnl_on_tick = cashflow_last
        elif position_last > 0:
            pnl_on_tick = position_last * tick2["BidPrice1"] + cashflow_last
        else:
            pnl_on_tick = position_last * tick2["AskPrice1"] + cashflow_last

        # If there is no trade this tick, simply update MTM PnL
        if len(self.trade_record_on_tick) == 0 or tick2["BidPrice1"] == 0 or tick2["AskPrice1"] == 0:
            self.trade_record_history.append(TickTradeRecord(tick2["TimeStamp"], *trade_record_last))
            if not(tick2["BidPrice1"] == 0 or tick2["AskPrice1"] == 0):
                self.trade_record_history[-1].pnl = pnl_on_tick
            return

        cashflow_on_tick = 0
        position_on_tick = 0
        turnover_on_tick = 0
        volume_on_tick = 0

        for order_id, order in self.trade_record_on_tick.items():
            cashflow_on_tick += -1 * order.quantity * order.price
            position_on_tick += order.quantity
            turnover_on_tick += abs(order.quantity) * order.price
            volume_on_tick += abs(order.quantity)

        fee_on_tick = 0
        if self.feetype == "vol":
            fee_on_tick = volume_on_tick * self.fee / self.multiplier
        else:  # Feetype = ratio
            fee_on_tick = turnover_on_tick * self.fee / 10000

        if position_on_tick > 0:
            sum_dir_on_tick = 1
        elif position_on_tick < 0:
            sum_dir_on_tick = -1
        else:
            sum_dir_on_tick = 0

        cashflow_on_tick += cashflow_last
        position_on_tick += position_last
        fee_on_tick += fee_last

        self.trade_record_history.append(TickTradeRecord(tick2["TimeStamp"],
                                                         position_on_tick, cashflow_on_tick, pnl_on_tick,
                                                         fee_on_tick))
        self.trade_record_history[-1].order_dir = sum_dir_on_tick
        self.trade_record_history[-1].volume = volume_on_tick

        if self.strategy.cancelled_count > self.cancelled_count:
            print("Order cancelled")
            self.trade_record_history[-1].cancel_flag = 1

    # row_wise Create the DirChg column, used only in compute_pnl_summary()
    def create_DirChg(self, row):
        if row["LastPosit"] == 0 and row["Position"] == 0:
            return 0
        elif row["LastPosit"] * row["Position"] <= 0:
            return int(math.copysign(1, (row["Position"] - row["LastPosit"])))

    def compute_pnl_summary(self, include_fee=True, show_orders=True, show_fee=True, show_cancel = True):
        df = pd.DataFrame.from_records([record.to_dict() for record in self.trade_record_history])
        if include_fee:
            df["PnL"] = df["PnL"] - df["Fee"]
        df["Drawdown"] = (df["PnL"] - df["PnL"].cummax())
        df["LastPosit"] = df["Position"].shift(1)
        # When our position switched sign
        df["DirChg"] = df.apply(self.create_DirChg, axis=1)
        df["DirChg"].fillna(0, inplace=True)
        df["hCount"] = self.bData["hCount"].values
        df["dCount"] = self.bData["dCount"].values
        df.loc[df['Volume'] == 0, 'TradeDone'] = 0
        df.loc[df['Volume'] != 0, 'TradeDone'] = 1
        df.to_csv(self.strat_name + ".csv")

        # Correct
        # PnL by day
        pnl_by_day = df.groupby("dCount").tail(1).PnL.values * self.multiplier
        # Number of trade by day
        trade_count_by_day = df.groupby("dCount").sum().TradeDone.values

        cycle_start_ts = -1
        cycle_start_pnl = 0
        cycle_holding_period = []
        cycle_pnl = []
        for row in df.itertuples():
            # Skip first row
            if row.Index == 0: continue
            # Position opened
            if row.LastPosit == 0 and row.Position != 0:
                cycle_start_ts = row.Index
                cycle_start_pnl = row.PnL
            # Position closed
            elif row.LastPosit != 0 and row.Position == 0:
                cycle_holding_period.append(row.Index - cycle_start_ts)
                cycle_start_ts = -1

                cycle_pnl.append(row.PnL - cycle_start_pnl)
                cycle_start_pnl = 0

            # Position reversed (closed and opened)
            elif row.LastPosit * row.Position < 0:
                cycle_holding_period.append(row.Index - cycle_start_ts)
                cycle_start_ts = row.Index

                cycle_pnl.append(row.PnL - cycle_start_pnl)
                cycle_start_pnl = row.PnL

        """
        The first plot: PnL, Max Drawdown, Price Curve
        """
        fig = plt.figure(figsize=(14, 8))
        ax_pnl = fig.add_subplot(311)
        (df["PnL"]*self.multiplier).plot()
        if show_fee: (df["Fee"]*self.multiplier).plot()
        ax_pnl.set_title("Cumulative PnL")

        ax_dd = fig.add_subplot(312, sharex=ax_pnl)
        (df["Drawdown"]*self.multiplier).plot()
        ax_dd.set_title("Max Drawdown")

        if show_cancel:
            for index, cancel_flag in enumerate(df["CancelFlag"].values):
                if cancel_flag == 1:
                    ax_dd.axvline(x=index, linestyle='--', color="red", lw=0.5)

        ax_lp = fig.add_subplot(313, sharex=ax_pnl)
        plt.plot(self.bData["LastPrice"].values[:len(df)])
        ax_lp.set_title("Last Price")

        if show_orders:
            for index, order_dir in enumerate(df["OrderDir"].values):
                if order_dir == 1:
                    ax_lp.axvline(x=index, linestyle='--', color="red", lw=0.5)
                elif order_dir == -1:
                    ax_lp.axvline(x=index, linestyle='--', color="green", lw=0.5)
        fig.tight_layout()

        """
        The second plot: PnL, Max Drawdown, Price Curve
        """
        fig2 = plt.figure(figsize=(14, 8))
        ax_pnl = fig2.add_subplot(311)
        color = 'tab:blue'
        ax_pnl.set_xlabel("Days")
        ax_pnl.set_ylabel("Daily PnL", color=color)
        ax_pnl.plot(range(1, len(pnl_by_day) + 1), pnl_by_day, color=color)
        ax_pnl.set_title("Daily Stats")
        ax_pnl.tick_params(axis='y', labelcolor=color)
        ax_pnl.set_xticks(range(1, len(pnl_by_day) + 1))
        color = 'tab:red'
        ax_trade_count = ax_pnl.twinx()
        ax_trade_count.set_ylabel("# of trade", color=color)
        ax_trade_count.bar(range(1, len(pnl_by_day) + 1), trade_count_by_day, color=color, alpha=0.3, linewidth=1)
        ax_trade_count.tick_params(axis='y', labelcolor=color)

        ax_hp = fig2.add_subplot(312)  # Hist for Holding period in ticks
        ax_hp.set_xlabel("Holding Periods")
        ax_hp.set_ylabel("Probability")
        ax_hp.set_title("Histogram of Holding Periods (ticks)")
        #ax_hp = sns.distplot(cycle_holding_period, fit= norm, kde=False)
        ax_hp.hist(cycle_holding_period, 50, density=True, alpha=0.75)

        ax_ret = fig2.add_subplot(313)  # Hist for Holding period in ticks
        ax_ret.set_xlabel("PnL")
        ax_ret.set_ylabel("Probability")
        ax_ret.set_title("Histogram of PnL for each round trip")
        ax_ret = sns.distplot(cycle_pnl, 50, fit=norm, kde=False)
        #ax_ret.hist(cycle_pnl, 50, density=True, alpha=0.75)
        fig2.tight_layout()

        """
        Here we generate the summary statistics of our backtest
        """
        pnl = df.iloc[-1]["PnL"]
        max_posit = max(df['Position'].max(), abs(df['Position'].min()))
        last_price = self.bData["LastPrice"].max()
        max_margin = (max_posit * last_price * self.margin_ratio)
        ret = pnl / max_margin
        day_count = self.bData.iloc[-1]["dCount"] + 1
        fee = df.iloc[-1]["Fee"]
        trade_count = df["Volume"].astype(bool).sum(axis=0)
        open_count = df.loc[(df["Position"] != 0) & (df["DirChg"] != 0), "TradeDone"].sum()
        SR = np.mean(pnl_by_day / max_margin) / np.std(pnl_by_day / max_margin)

        print("Strategy                 :" + self.strat_name)
        print("Contract                 :" + self.contract)
        print("Start Time               :" + str(self.hrb.tInfo.startTime))
        print("End Time                 :" + str(self.hrb.tInfo.endTime))
        tmp = pnl * self.multiplier
        print("PnL (post fee)           :" + f'{tmp:.0f}' + " (" + '{:.2%}'.format(ret) + ")")
        tmp = pnl / (df["Volume"].sum() / 2) * self.multiplier
        print("PnL per hand             :" + f'{tmp:.2f}')
        tmp = df["Drawdown"].min() * self.multiplier
        print("Max Drawdown             :" + f'{tmp:.0f}' + " (" + '{:.2%}'.format(
            df["Drawdown"].min() / max_margin) + ")")
        print("Sharpe Ratio             :" + f'{SR:.2f}')
        tmp = fee * self.multiplier
        print("Total Fee                :" + f'{tmp:.0f}' + " (" + '{:.2%}'.format(
            fee / (pnl + fee)) + " of gross PnL)")
        tmp = trade_count / day_count
        print("Total trades             :" + str(trade_count) + " (" + f'{tmp:.0f}' + "/day)")
        tmp = open_count / day_count
        print("Total open               :" + str(int(open_count)) + " (" + f'{tmp:.0f}' + "/day)")
        # To fix
        tmp = np.mean(cycle_holding_period)
        print("Average Holding Period   :" + f'{tmp:.0f}' + " (ticks)")
        tmp = self.strategy.cancelled_count / self.strategy.order_id_count
        print("Cancellation Rate        :" + '{:.2%}'.format(tmp))

        plt.show()

    def run(self, strategy):
        if self.bData.empty:
            print("Please load data first")
            return

        # Set up strategy here
        self.strategy = strategy

        # Trim dataframe to what we use only after passing original DataFrame to "Strategy"
        self.trim_df()

        tick_count = 0
        last_tick, this_tick, next_tick = None, None, None
        for row in self.bData.values:
            if tick_count == 0:
                last_tick = dict(zip(self.trimmed_names, row))
                tick_count += 1
                continue
            elif tick_count == 1:
                this_tick = dict(zip(self.trimmed_names, row))
                tick_count += 1
                continue
            elif tick_count >= 2:
                next_tick = dict(zip(self.trimmed_names, row))

            # Step 2 & 3
            if self.data_type == "l2":
                break
            else:
                self.match_old_orders_l1(last_tick, this_tick)
            # After matching done for old orders, we can
            self.build_tick_trade_record(this_tick)

            # Step 4, 5 & 6
            self.strategy.update_order_list(self.order_list)

            # Feed transaction record from the matching back to Strategy
            self.strategy.trade_record_history = self.trade_record_history
            self.strategy.trade_record_on_tick = self.trade_record_on_tick


            # Clear trade record upon arrival of the new tick
            self.trade_record_on_tick = dict()
            self.cancelled_count = self.strategy.cancelled_count

            # Feed new tick
            new_order_list, temp_order_list = self.strategy.process_tick(tick_count - 1)

            # Step 8
            self.order_list = temp_order_list

            # Step 9 & 10
            if self.mode != "harsh":
                if self.data_type == "l2":
                    break
                else:
                    self.match_new_orders_l1(this_tick, new_order_list)
            else:  # "harsh" mode use next tick update to match new orders
                if self.data_type == "l2":
                    break
                else:
                    self.match_new_orders_l1(next_tick, new_order_list)

            last_tick = this_tick
            this_tick = next_tick
            tick_count += 1

        self.trade_record_history.append(TickTradeRecord(0, *self.trade_record_history[-1].get_key_info()))
        self.compute_pnl_summary(self.strategy.include_fee, self.strategy.show_orders,
                                 self.strategy.show_fee,    self.strategy.show_cancel)

