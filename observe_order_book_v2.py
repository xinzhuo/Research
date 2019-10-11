import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import scipy as sp
import os
from datetime import datetime, date, time, timedelta
from hft_rsys import *

# Data structure to store historial orderbook info
from Hist_OB import Cylinder
import tkinter
from collections import OrderedDict
import matplotlib
matplotlib.use('TkAgg')

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
# Called only by between_tick_analyze
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

def analyze_between_tick(tick1, tick2, min_step):
    """
    :param tick1:
    :param tick2:
    :return: dict with price as key and volume as value
    """
    #min_step = self.min_step
    #multiplier = self.multiplier
    #vol_side = 2 if self.double_side_volume else 1
    tick1_dict, tick2_dict = {}, {}
    price_columns = ["BidPrice5", "BidPrice4", "BidPrice3", "BidPrice2", "BidPrice1",
                     "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5"]
    volume_columns = ["BidSize5", "BidSize4", "BidSize3", "BidSize2", "BidSize1",
                      "AskSize1", "AskSize2", "AskSize3", "AskSize4", "AskSize5"]

    # build two dictionaries to map price by index
    for index, price_posit in enumerate(price_columns):
        if price_posit[:3] == "Bid":
            tick1_dict[tick1[price_posit]] = -1 * tick1[volume_columns[index]]
        else:
            tick1_dict[tick1[price_posit]] = tick1[volume_columns[index]]

        if price_posit[:3] == "Bid":
            tick2_dict[tick2[price_posit]] = -1 * tick2[volume_columns[index]]
        else:
            tick2_dict[tick2[price_posit]] = tick2[volume_columns[index]]

    # calculate volume residual
    #volume = (tick2["TotalVolume"] - tick1["TotalVolume"]) / vol_side
    #cashVolume = ((tick2["Turnover"] - tick1["Turnover"]) / multiplier) / vol_side
    volume = tick2["volume"]
    cashVolume = tick2["turnover"]
    volResidual = int((cashVolume - (volume * tick1["BidPrice1"])) / min_step)

    # Assumption: no price fluctuation between tick
    lowBound = min(tick1["BidPrice1"], tick2["BidPrice1"])
    upBound = max(tick1["AskPrice1"], tick2["AskPrice1"])

    # possible traded price
    price_list = np.arange(lowBound, upBound + min_step, min_step)
    bucket_dict = {}
    result_dict = {}

    # If bid&ask remain unchanged, under our assumption, we have a relative "accurate" estimation of trade size
    if len(price_list) == 2 or volume == 0:
        if int(volume - volResidual) >= 0 and int(volResidual) >= 0:
            result_dict[0] = int(volume - volResidual)
            result_dict[1] = int(volResidual)
        else:
            result_dict[int(round(volResidual / volume))] = int(volume)
        return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)

    # map position relative to the best bid to volume traded inferred in the orderbook
    non_zero_key = []
    non_zero_value = []
    for price_posit in price_list:
        # All the orders at the level was consumed
        spot = round((price_posit - tick1["BidPrice1"]) / min_step)
        if price_posit not in tick1_dict:
            bucket_dict[spot] = 0
        elif price_posit not in tick2_dict:
            if price_posit > tick2["AskPrice5"] or price_posit < tick2["BidPrice5"]:
                bucket_dict[spot] = 0
            else:
                bucket_dict[spot] = abs(tick1_dict[price_posit])
        elif tick1_dict[price_posit] * tick2_dict[price_posit] <= 0:
            bucket_dict[spot] = abs(tick1_dict[price_posit])
        else:
            if abs(tick1_dict[price_posit]) > abs(tick2_dict[price_posit]):
                bucket_dict[spot] = abs(tick1_dict[price_posit]) - abs(tick2_dict[price_posit])
            else:
                bucket_dict[spot] = 0
        if bucket_dict[spot] != 0:
            non_zero_key.append(spot)
            non_zero_value.append(bucket_dict[spot])

    # if only 2 non-zero volume prices
    # non_zero_key[0] * x1 + non_zero_key[1] * x2 = volResidual
    # x1 + x2 = volume
    if len(non_zero_key) == 2:
        a = np.array([[non_zero_key[0], non_zero_key[1]], [1, 1]])
        b = np.array([volResidual, volume])
        x = np.linalg.solve(a, b)

        if x[0] >= 0 and x[1] >= 0:
            result_dict[non_zero_key[0]] = int(x[0])
            result_dict[non_zero_key[1]] = int(x[1])
        else:
            result_dict[int(round(volResidual / volume))] = int(volume)

        return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)

    # Here we want to first re-arrange the order of the dictionary, so the algo access the prices that are more likely
    # to be traded first. Mid-price up, we look at bid prices first and vice versa
    # There are 2 situation when the spread remains the same, 3 situations when the spread widens and 2 situations
    # when the spread narrows.
    # We rank the items based on the change in the mid price
    bucket_dict_temp = OrderedDict()

    if (tick1["BidPrice1"] + tick1["AskPrice1"]) >= (tick2["BidPrice1"] + tick2["AskPrice1"]):
        # rank bid before ask if mid is lower, position relative to Tick1[Bid1]
        price_posit = 0
        while price_posit in bucket_dict:
            bucket_dict_temp[price_posit] = bucket_dict[price_posit]
            price_posit = price_posit - 1

        price_posit = 1
        while price_posit in bucket_dict:
            bucket_dict_temp[price_posit] = bucket_dict[price_posit]
            price_posit = price_posit + 1
    else:
        price_posit = int((tick1["AskPrice1"] - tick1["BidPrice1"]) / min_step)
        while price_posit in bucket_dict:
            bucket_dict_temp[price_posit] = bucket_dict[price_posit]
            price_posit = price_posit + 1

        price_posit = int((tick1["AskPrice1"] - tick1["BidPrice1"]) / min_step) - 1
        while price_posit in bucket_dict:
            bucket_dict_temp[price_posit] = bucket_dict[price_posit]
            price_posit = price_posit - 1

    bucket_dict = bucket_dict_temp

    # There are two cases in the next step:
    # 1. Total volume inferred by orderbook is less than Volume, so we allocate the volume to ob first and some more
    #    However, problem could arise when volume residual presented in ob exceeded calculated residual.
    #    Here we figure out ways to deal with this situation later: filling priority?
    # 2. Total volume inferred by orderbook is more than Volume. We fill the volume inferred partly based on our
    #    ranking.

    # print(bucket_dict_temp)
    for key, value in bucket_dict.items():
        if value != 0:
            # Here checks if there are enough volume for the ob inferred volume
            # Should we also check for volume residual here??????
            if volume < value:
                break
            result_dict[key] = int(value)
            volume -= value
            volResidual -= key * value
    if volume == 0:
        return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)
        # Need to check if volResidual == 0 in the future update

    # After this point,the uncertainly arises, we can only designed a simple rule
    # Here we assume the remaining volume are traded at the first 2 items of the ordereddict
    items = list(bucket_dict.items())

    a = np.array([[items[0][0], items[1][0]], [1, 1]])
    b = np.array([volResidual, volume])
    x = np.linalg.solve(a, b)

    # Check if solved result are value (negative)
    if x[0] >= 0 and x[1] >= 0:
        if items[0][0] in result_dict:
            result_dict[items[0][0]] += int(x[0])
        else:
            result_dict[items[0][0]] = int(x[0])

        if items[1][0] in result_dict:
            result_dict[items[1][0]] += int(x[1])
        else:
            result_dict[items[1][0]] = int(x[1])
    # If result not valid, assign all volume to the closest average price
    else:
        price_posit = int(round(volResidual / volume))
        if price_posit in bucket_dict:
            if price_posit in result_dict:
                result_dict[price_posit] += volume
            else:
                result_dict[price_posit] = volume

    return process_between_tick_result(result_dict, tick1["BidPrice1"], min_step)

def plot_single(ax, row, row_sup, min_tick, no_gap = False):
    price_columns = ["BidPrice5", "BidPrice4", "BidPrice3", "BidPrice2", "BidPrice1",
             "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5"]
    sup_values = [row_sup[name] for name in price_columns] # need to convert to str later
    volume_columns = ["BidSize5", "BidSize4", "BidSize3", "BidSize2", "BidSize1",
               "AskSize1", "AskSize2", "AskSize3", "AskSize4", "AskSize5"]

    od = {}
    for index, price in enumerate(price_columns):
        if index == 0:
            od[np.round(row[price], 1)] = -1 * row[volume_columns[index]]
        else:

            if price[:3] == "Bid":
                od[np.round(row[price], 1)] = -1 * row[volume_columns[index]]
            else:
                od[np.round(row[price], 1)] = row[volume_columns[index]]

    min_price = min(row["BidPrice5"], row_sup["BidPrice5"])
    max_price = max(row["AskPrice5"], row_sup["AskPrice5"])
    min_tick_size = int(round((max_price - min_price)/min_tick)) + 1
    print(min_tick_size)

    # np_gap: don't want to see the gap between price, especially for if,
    if not no_gap:
        for value in [x * min_tick + min_price for x in range(min_tick_size)]:
            tmp = np.round(value, 1)
            if tmp not in od:
                od[tmp] = 0

    od = OrderedDict(sorted(od.items(), key=lambda t: t[0]))
    price_values = []
    volume_values = []
    for key, value in od.items():
        price_values.append(str(key))
        volume_values.append(value)


    barlist = ax.barh(price_values, volume_values, align='center')
    for i in range(len(volume_values)):
        if volume_values[i] < 0:
            barlist[i].set_color('r')
    ax.set_yticks(np.arange(len(volume_values)))
    ax.set_yticklabels(price_values)
    current_time = datetime.fromtimestamp(int(str(int(row["TimeStamp"]))[:10]))
    current_time = current_time + timedelta(
        milliseconds=int(str(int(row["TimeStamp"]))[10:13]))
    ax.set_xlabel(str(current_time))
    ax.margins(0.25, 0)
    ax.axvline(0, color='black')

    ax.grid(True)
    for i, v in enumerate(volume_values):
        if volume_values[i] < 0:
            plt.text(v - 3, i, str(v), va="center", ha="right")
        elif volume_values[i] > 0:
            plt.text(v + 3, i, str(v), va="center", ha="left")

def visualize_order_book(contract_id, contract_month, start_time, end_time, source):
    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, source, 0)
    contract_info = hrb.get_contract_data()
    df = hrb.get_hft_data()
    df["VolDiff"] = df["TotalVolume"].diff()
    df["TODiff"] = df["Turnover"].diff()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    counter = 1
    fig = plt.figure(figsize=(14, 8))
    fig.tight_layout()
    gs = gridspec.GridSpec(2, 2)
    while counter < len(df.index):
        if counter > 0:
            ax1 = fig.add_subplot(gs[0, 0])
            plot_single(ax1, df.iloc[counter - 1], df.iloc[counter], min_tick)
            ax1.yaxis.tick_right()
            ax2 = fig.add_subplot(gs[0, 1])
            plot_single(ax2, df.iloc[counter], df.iloc[counter - 1], min_tick)
            fig.tight_layout()

            ax3 = fig.add_subplot(gs[1, 0])
            if counter > 2400:
                ax3.plot(df["LastPrice"][counter - 2400:counter + 1].values)
            ax3.grid(True)
            ax3.yaxis.tick_right()


            ax4 = fig.add_subplot(gs[1, 1])
            ax_volume = ax4.twinx()
            if counter > 240:
                ax4.plot(df["LastPrice"][counter - 240:counter + 1].values)
            ax4.grid(True)

            ax_volume.set_ylabel("Volume")
            if counter > 240:
                ax_volume.bar(range(1, 242), df["VolDiff"][counter - 240:counter + 1].values,
                              alpha = 0.6, color = "red", width = 0.5)

            plt.draw()

            volume = (df.iloc[counter]["TotalVolume"] - df.iloc[counter - 1]["TotalVolume"])/vol_side
            cashVolume = ((df.iloc[counter]["Turnover"] - df.iloc[counter - 1]["Turnover"]) / multiplier)/vol_side
            textstr = "\n".join((
                "Volume: " + str(int(volume)),
                "CashVolume: " + str(cashVolume),
                "VolumeResid: " + str(int((cashVolume-(volume * df.iloc[counter -1]["BidPrice1"]))/min_tick)),
                "LastVolume: " + str(df.iloc[counter]["LastVolume"]/2),
                "LastPrice: " + str(df.iloc[counter]["LastPrice"]))
            )
            props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
            ax4.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
                     verticalalignment='top', bbox=props)

            plt.pause(0.001)
        else:
            counter += 1
            continue
        choice = 2
        try:
            choice = int(input("1 for last tick, 2 for next tick: "))
        except:
            print("")


        fig.clf()
        if choice == 0:
            counter -= 20
        elif choice == 1:
            counter -= 1
        elif choice == 2:
            counter += 1
        elif choice == 3:
            counter += 240
        elif choice == 4:
            counter += 14400
        else:
            counter += 1

def visualize_ob_history(contract_id, contract_month, start_time, end_time, source):
    #mpl.style.use("seaborn")
    if source[:2] != 'l2':
        print("Please use L2 data source")
        return

    column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                    "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                    "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                    "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                    "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                    "volume", "turnover", "TimeStamp", "hCount",
                    "FallLimit", "RiseLimit", "TotalVolume", "Turnover"]

    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, source, 0)
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    df = hrb.get_hft_data()
    df["volume"] = df["TotalVolume"].diff()/vol_side
    df["turnover"] = df["Turnover"].diff()/vol_side/multiplier
    df["hCount"] = hrb.get_hCount()

    df = df.loc[:, column_names]

    # Select only tick with Volume != 0
    #df = df.loc[df["volume"] != 0]
    df.reset_index(inplace=True)

    vwap_list = []
    # Setup plot here
    fig = plt.figure(figsize=(20, 8))
    plot_tick = 30 # number of tick to be plotted

    # Setup Cylinder OB structure here
    cy = None
    n_tick = plot_tick

    transaction_record = []
    last_row = None
    for index, row in df.iterrows():
        ax = fig.add_subplot(111)
        # Check for new trading sessions and Define VWAP price here
        if index == 0:
            cy = Cylinder(n_tick, min_tick, row["FallLimit"], row["RiseLimit"])
            transaction_record.append(dict())
        elif last_row is not None and last_row["hCount"] != row["hCount"]:
            print("Reset Data Structure for new trading session")
            cy = Cylinder(n_tick, min_tick, row["FallLimit"], row["RiseLimit"])
            transaction_record.append(dict())
        else:
            transaction_record.append(
                analyze_between_tick(last_row, row, min_tick, multiplier, vol_side)
            )

        last_row = row
        cy.update(row)

        # Define VWAP price here
        if row["volume"] != 0:
            vwap_list.append(row["turnover"]/row["volume"] )
        else:
            if len(vwap_list) >= 1:
                vwap_list.append(vwap_list[-1])
            else:
                vwap_list.append((row["BidPrice1"] + row["AskPrice1"])/2)

        # After complete updating the data structure, we can plot
        bidPrice5, askPrice5 = row["BidPrice5"], row["AskPrice5"]
        if index > plot_tick:
            bidPrice5 = np.min(df.loc[index - plot_tick + 1:index + 1, ["BidPrice5"]].values)
            askPrice5 = np.max(df.loc[index - plot_tick + 1:index + 1, ["AskPrice5"]].values)


        min_tick_size = int(round((askPrice5 - bidPrice5) / min_tick)) + 1

        for value in [x * min_tick + bidPrice5 for x in range(min_tick_size)]:
            ax.scatter(list(range(plot_tick)), [value] * plot_tick,
                       marker = "o", alpha = 0)
            #print(cy.retrieve(value))
            for i, txt in enumerate(list(cy.retrieve(value))[-1 * plot_tick:]):
                if txt is not None:
                    color = "crimson"
                    if float(txt) > 0:
                        color = "mediumblue"
                    elif float(txt) == 0:
                        color = "silver"
                    ax.annotate(txt, (i, value) , size = 12, color = color)

        for i in range(plot_tick):
            ax.axvline(i, color='black', linestyle = ':', alpha = 0.3)
        if index > plot_tick:
            # Plot VWAP Price
            if vol_side != 1:
                ax.plot(list(range(plot_tick)), vwap_list[-plot_tick:],
                        marker = "^", linestyle = "-.")

            # Plot transaction record:
            for i in range(1, plot_tick + 1):
                 # start from -1
                if vol_side == 2:
                    for key, value in transaction_record[i * -1].items():
                        ax.arrow(plot_tick - i - 1 + 0.25, key, 0.5, 0, head_width = 0.1)
                        ax.annotate(str(value), (plot_tick - i - 0.5, key), size = 15, color = "forestgreen")
                elif vol_side == 1:
                    ax.arrow(plot_tick - i - 1 + 0.25, vwap_list[-1 * i], 0.5, 0, head_width=0.1)
                    ax.annotate(int(df.iloc[index - i + 1]["volume"]), (plot_tick - i - 0.5, vwap_list[-1 * i]), size=15, color="forestgreen")

        current_time = datetime.fromtimestamp(int(str(int(row["TimeStamp"]))[:10]))
        current_time = current_time + timedelta(
            milliseconds=int(str(int(row["TimeStamp"]))[10:13]))
        ax.set_title(str(current_time))

        print(str(current_time))
        print("Volume: " + str(row["volume"]))

        ax.set_xticks(np.arange(plot_tick))
        ax.set_xticklabels(np.arange(index - plot_tick + 1, index + 1))
        ax.set_yticks([x * min_tick + bidPrice5 for x in range(min_tick_size)])
        ax.set_xlabel("Tick")
        ax.set_ylabel("Price")
        #ax.grid(True)
        plt.draw()
        plt.pause(0.001)
        input()
        fig.clf()

# Use given index
# Have the option to plot a time series along with the plot, should have aligned index
def visualize_ob_history_2(contract_id, contract_month, start_time, end_time, source, filled,
                           my_series = None, my_series2 = None, my_series_time = None):
    if source[:2] != 'l2' and source[-2:] != 'l2':
        print("Please use L2 data source")
        return

    column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                    "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                    "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                    "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                    "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                    "volume", "turnover", "TimeStamp", "hCount",
                    "FallLimit", "RiseLimit", "vwap"]
    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, source, filled, True)
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    df = hrb.get_hft_data()
    #print(df)
    df["volume"] = df["TotalVolume"].diff()/vol_side
    df["turnover"] = df["Turnover"].diff()/vol_side/multiplier
    df["hCount"] = hrb.get_hCount()

    get_vwap(df, multiplier, vol_side)

    df = df.loc[:, column_names]
    # Select only tick with Volume != 0

    df.reset_index(inplace=True)
    # Setup plot here


    ts = input("Enter index: ")  # Index selected
    fig = plt.figure(figsize=(17, 10))
    plot_tick = 40  # number of tick to be plotted


    while ts != "exit":
        ts = int(float(ts))
        # Setup Cylinder OB structure here
        ax = fig.add_subplot(111)
        cy = None
        n_tick = plot_tick

        vwap_list = []
        transaction_record = []
        last_row = None

        for index in range(ts - plot_tick + 1, ts + 1):
            row = df.iloc[index]

            # Check for new trading sessions and Define VWAP price here
            if index == 0 or last_row is None:
                cy = Cylinder(n_tick, min_tick, row["FallLimit"], row["RiseLimit"])
                transaction_record.append(dict())
            elif last_row is not None and last_row["hCount"] != row["hCount"]:
                print("Reset Data Structure for new trading session")
                cy = Cylinder(n_tick, min_tick, row["FallLimit"], row["RiseLimit"])
                transaction_record.append(dict())
            else:
                transaction_record.append(
                    analyze_between_tick(last_row, row, min_tick)
                )

            last_row = row
            cy.update(row)

            #ax.annotate(int(txt), (i, value), size=12, color=color)
            #if index in my_series.index:
            #    ax.plot(index - ts + plot_tick - 1, my_series.loc[index], marker='o', markersize=3, color="red")
            #    print(index)
            #    print(my_series.loc[index])
                #plt.plot([x], [y], marker='o', markersize=3, color="red")
        if my_series is not None:
            ax.plot(list(range(plot_tick)), my_series.loc[:ts].tail(plot_tick).values, marker='o', markersize=3,
                color="red", linestyle = ":")
        if my_series2 is not None:
            ax.plot(list(range(plot_tick)), my_series2.loc[:ts].tail(plot_tick).values, marker='o', markersize=3,
                color="green", linestyle=":")
        #print(my_series_time.loc[ts])

        vwap_list = df["vwap"].values[ts - plot_tick + 1: ts + 1]
        index = ts
        print(row["volume"] )
        vol_residual = round((row["turnover"] - (row["volume"] * last_row["BidPrice1"])) / min_tick)
        print(vol_residual)
        # After complete updating the data structure, we can plot
        bidPrice5, askPrice5 = last_row["BidPrice5"], last_row["AskPrice5"]
        if index > plot_tick:
            bidPrice5 = np.min(df.loc[index - plot_tick + 1:index, ["BidPrice5"]].values)
            askPrice5 = np.max(df.loc[index - plot_tick + 1:index, ["AskPrice5"]].values)


        min_tick_size = int(round((askPrice5 - bidPrice5) / min_tick)) + 1

        for value in [x * min_tick + bidPrice5 for x in range(min_tick_size)]:
            ax.scatter(list(range(plot_tick)), [value] * plot_tick,
                       marker = "o", alpha = 0)
            #print(cy.retrieve(value))
            for i, txt in enumerate(list(cy.retrieve(value))[-1 * plot_tick:]):
                if txt is not None:
                    color = "crimson"
                    if float(txt) > 0:
                        color = "mediumblue"
                    elif float(txt) == 0:
                        color = "silver"
                    ax.annotate(int(txt), (i, value) , size = 12, color = color)

        for i in range(plot_tick):
            ax.axvline(i, color='black', linestyle = ':', alpha = 0.3)

        if index > plot_tick:
            # Plot VWAP Price
            if vol_side != 1:
                ax.plot(list(range(plot_tick)), vwap_list[-plot_tick:],
                        marker = "^", linestyle = "-.")
            #ax.plot(list(range(plot_tick)), my_series.values[ts - plot_tick + 1: ts + 1],
            #        marker="*", linestyle="None")

            # Plot transaction record:
            for i in range(1, plot_tick + 1):
                 # start from -1
                if vol_side == 2:
                    for key, value in transaction_record[i * -1].items():
                        ax.arrow(plot_tick - i - 1 + 0.25, key, 0.5, 0, head_width = 0.1)
                        ax.annotate(str(value), (plot_tick - i - 0.5, key), size = 15, color = "forestgreen")
                elif vol_side == 1:
                    ax.arrow(plot_tick - i - 1 + 0.25, vwap_list[-1 * i], 0.5, 0, head_width=0.1)
                    ax.annotate(int(df.iloc[index - i + 1]["volume"]), (plot_tick - i - 0.5, vwap_list[-1 * i]), size=15, color="forestgreen")
        current_time = datetime.fromtimestamp(int(str(int(row["TimeStamp"]))[:10]))
        current_time = current_time + timedelta(
            milliseconds=int(str(int(row["TimeStamp"]))[10:13]))
        ax.set_title(str(current_time))
        ax.set_xticks(np.arange(plot_tick))
        ax.tick_params(axis='x', rotation=70)
        ax.set_xticklabels(np.arange(index - plot_tick + 1, index + 1))
        ax.set_yticks([x * min_tick + bidPrice5 for x in range(min_tick_size)])
        ax.set_xlabel("Tick")
        ax.set_ylabel("Price")
        #ax.grid(True)
        plt.draw()
        plt.pause(0.001)

        try:
            ts = int(input("Enter Index: "))
            if ts == 88:
                return
        except:
            print("Next tick")
            ts = ts + 1
        fig.clf()

def visualize_ob_history_arb_sub(fig, df, ts, plot_tick, min_tick, vol_side, contr, contract_id):

    ts = int(float(ts))
    # Setup Cylinder OB structure here
    ax = fig.add_subplot(2, 1, contr)
    cy = None
    n_tick = plot_tick

    transaction_record = []
    last_row = None

    for index in range(ts - plot_tick + 1, ts + 1):
        row = df.iloc[index]

        # Check for new trading sessions and Define VWAP price here
        if index == 0 or last_row is None:
            cy = Cylinder(n_tick, min_tick, row["FallLimit"], row["RiseLimit"])
            transaction_record.append(dict())
        elif last_row is not None and last_row["hCount"] != row["hCount"]:
            print("Reset Data Structure for new trading session")
            cy = Cylinder(n_tick, min_tick, row["FallLimit"], row["RiseLimit"])
            transaction_record.append(dict())
        else:
            transaction_record.append(
                analyze_between_tick(last_row, row, min_tick)
            )

        last_row = row
        cy.update(row)

    vwap_list = df["vwap"].values[ts - plot_tick + 1: ts + 1]

    index = ts
    print(row["volume"] )
    vol_residual = round((row["turnover"] - (row["volume"] * last_row["BidPrice1"])) / min_tick)
    print(vol_residual)
    # After complete updating the data structure, we can plot
    bidPrice5, askPrice5 = last_row["BidPrice5"], last_row["AskPrice5"]
    if index > plot_tick:
        bidPrice5 = np.min(df.loc[index - plot_tick + 1:index, ["BidPrice5"]].values)
        askPrice5 = np.max(df.loc[index - plot_tick + 1:index, ["AskPrice5"]].values)


    min_tick_size = int(round((askPrice5 - bidPrice5) / min_tick)) + 1

    for value in [x * min_tick + bidPrice5 for x in range(min_tick_size)]:
        ax.scatter(list(range(plot_tick)), [value] * plot_tick,
                   marker = "o", alpha = 0)
        #print(cy.retrieve(value))
        for i, txt in enumerate(list(cy.retrieve(value))[-1 * plot_tick:]):
            if txt is not None:
                color = "crimson"
                if float(txt) > 0:
                    color = "mediumblue"
                elif float(txt) == 0:
                    color = "silver"
                ax.annotate(int(txt), (i, value) , size = 10, color = color)

    for i in range(plot_tick):
        ax.axvline(i, color='black', linestyle = ':', alpha = 0.3)

    if index > plot_tick:
        # Plot VWAP Price
        if vol_side != 1:
            ax.plot(list(range(plot_tick)), vwap_list[-plot_tick:],
                    marker = "^", linestyle = "-.")

        # Plot transaction record:
        for i in range(1, plot_tick + 1):
             # start from -1
            if vol_side == 2:
                for key, value in transaction_record[i * -1].items():
                    ax.arrow(plot_tick - i - 1 + 0.25, key, 0.5, 0, head_width = 0.1)
                    ax.annotate(str(value), (plot_tick - i - 0.5, key), size = 10, color = "forestgreen")
            elif vol_side == 1:
                ax.arrow(plot_tick - i - 1 + 0.25, vwap_list[-1 * i], 0.5, 0, head_width=0.1)
                ax.annotate(int(df.iloc[index - i + 1]["volume"]), (plot_tick - i - 0.5, vwap_list[-1 * i]), size=11, color="forestgreen")
    current_time = datetime.fromtimestamp(int(str(int(row["TimeStamp"]))[:10]))
    current_time = current_time + timedelta(
        milliseconds=int(str(int(row["TimeStamp"]))[10:13]))
    ax.set_title(str(current_time))
    ax.set_xticks(np.arange(plot_tick))
    ax.set_xticklabels(np.arange(index - plot_tick + 1, index + 1))
    ax.tick_params(axis='x', rotation=40)
    ax.set_yticks([x * min_tick + bidPrice5 for x in range(min_tick_size)])
    ax.set_xlabel(contract_id)
    ax.set_ylabel("Price")
    #ax.grid(True)
    plt.draw()
    plt.pause(0.001)

def visualize_ob_history_arb(contract_id, contract_id_2, contract_month, contract_month_2,
                             start_time, end_time, source, filled):
    #mpl.style.use("seaborn")
    if source[:2] != 'l2' and source[-2:] != 'l2':
        print("Please use L2 data source")
        return

    column_names = ["BidPrice5", "BidSize5", "BidPrice4", "BidSize4",
                    "BidPrice3", "BidSize3", "BidPrice2", "BidSize2",
                    "BidPrice1", "BidSize1", "AskPrice1", "AskSize1",
                    "AskPrice2", "AskSize2", "AskPrice3", "AskSize3",
                    "AskPrice4", "AskSize4", "AskPrice5", "AskSize5",
                    "volume", "turnover", "TimeStamp", "hCount",
                    "FallLimit", "RiseLimit", "vwap"]

    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, source, filled, True)
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    df = hrb.get_hft_data()
    #df["volume"] = df["TotalVolume"].diff()/vol_side
    #df["turnover"] = df["Turnover"].diff()/vol_side/multiplier
    df["hCount"] = hrb.get_hCount()
    get_vwap(df, multiplier, vol_side)

    df = df.loc[:, column_names]
    # Select only tick with Volume != 0
    df.reset_index(inplace=True)
    # Setup plot here

    hrb2 = HRB.HRB(start_time, end_time, contract_id_2, contract_month_2, source, filled, True)
    contract_info2 = hrb2.get_contract_data()
    multiplier2 = contract_info2.multiplier
    min_tick2 = contract_info2.step
    vol_side2 = 2 if hrb2.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    df2 = hrb2.get_hft_data()
    #df2["volume"] = df2["TotalVolume"].diff()/vol_side2
    #df2["turnover"] = df2["Turnover"].diff()/vol_side2/multiplier2
    df2["hCount"] = hrb2.get_hCount()
    get_vwap(df2, multiplier2, vol_side2)

    df2 = df2.loc[:, column_names]
    # Select only tick with Volume != 0
    df2.reset_index(inplace=True)


    ts = input("Enter index: ")  # Index selected
    fig = plt.figure(figsize=(17, 10))
    fig.tight_layout()
    plot_tick = 30  # number of tick to be plotted


    while ts != "exit":

        visualize_ob_history_arb_sub(fig, df, ts, plot_tick, min_tick, vol_side, 1, contract_id + contract_month)
        visualize_ob_history_arb_sub(fig, df2, ts, plot_tick, min_tick2, vol_side2, 2, contract_id_2 + contract_month)

        try:
            ts = int(input("Enter Index: "))
            if ts == 88:
                return
        except:
            print("Next tick")
            ts = int(ts) + 1
        fig.clf()



if __name__ == '__main__':
    df = pd.read_csv("ni1911_20190826_trade_price.csv", index_col=1, header=0)
    #print(df["Open_Long"])
    starttime = datetime(2019, 8, 23, 21, 0, 0)
    endtime = datetime(2019, 8, 26, 15, 0, 0)

    """
    Custom dataframe, you can put two series as parameters into the plot along with index, 
    it will plot corresponding values based on index - df["Time"]
    
    In console, input index you want to observe
    input "88" to exit
    """
    #visualize_ob_history_2("ni", "1911", starttime, endtime, "xele_l2", True, df["Open_Long"], df["Close_Short"], df["Time"])
    visualize_ob_history_arb("ni", "ni", "1911", "1910", starttime, endtime, "xele_l2", True)
    #visualize_ob_history("jm"H "1909", starttime, endtime, "l2_dce")
    #visualize_order_book("eg", "1909", starttime, endtime, "l2_dce")