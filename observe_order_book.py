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
from Helper import analyze_between_tick, get_vwap
import tkinter
import matplotlib
matplotlib.use('TkAgg')

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
    print(df)
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
    #visualize_ob_history_2("ni", "1911", starttime, endtime, "xele_l2", True, df["Open_Long"], df["Close_Short"], df["Time"])
    visualize_ob_history_arb("ni", "ni", "1911", "1910", starttime, endtime, "xele_l2", True)
    #visualize_ob_history("jm"H "1909", starttime, endtime, "l2_dce")
    #visualize_order_book("eg", "1909", starttime, endtime, "l2_dce")