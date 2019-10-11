import numpy as np
import pandas as pd
import datetime as dt
import sys
import math
from hft_rsys import *
import matplotlib.pyplot as plt
import copy

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import probplot

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

from Helper import filter_volume, get_alternative_clock

pd.set_option('display.max_columns', 500)

def get_order_dir_sum(hrb, period):
    priceData = hrb.get_hft_data().loc[:, ["AskPrice1", "BidPrice1", "TotalVolume", "Turnover", "LastPrice"]]
    priceDataV = priceData.values
    contract_info = hrb.get_contract_data()

    multiplier = contract_info.multiplier

    vol_dir = [[0, 0]]
    last_row = []
    for index, row in enumerate(priceDataV):
        if index == 0:
            last_row = row
            continue
        volume = (row[2] - last_row[2])/2
        cashVolume = (row[3] - last_row[3])/2/multiplier

        # if only 2 non-zero volume prices
        # non_zero_key[0] * x1 + non_zero_key[1] * x2 = Turnover
        # x1 + x2 = volume
        a = np.array([[last_row[0], last_row[1]], [1, 1]])
        b = np.array([cashVolume, volume])
        x= np.linalg.solve(a, b)


        if x[0] < 0:
            x = [0, volume]
        elif x[1] < 0:
            x = [volume, 0]

        vol_dir.append([x[1], x[0]])
        last_row = row

    for i in range(priceDataV.shape[0]):
        if priceDataV[i, 0] == 0 and priceDataV[i, 1] != 0:
            priceDataV[i, 0] = priceDataV[i, 1]
        elif priceDataV[i, 0] != 0 and priceDataV[i, 1] == 0:
            priceDataV[i, 1] = priceDataV[i, 0]
        elif priceDataV[i, 0] == 0:
            priceDataV[i, 0] = priceDataV[i - 1, 0]
            priceDataV[i, 1] = priceDataV[i - 1, 1]

    df = pd.DataFrame(vol_dir, priceData.index, columns=["BidVol", "AskVol"])
    df["BidVolSum"] = df["BidVol"].rolling(period).sum()
    df["AskVolSum"] = df["AskVol"].rolling(period).sum()
    df["vol_dir_ratio"] = (df["AskVolSum"] - df["BidVolSum"])/(df["AskVolSum"] + df["BidVolSum"])

    df["BidSampled"] = generate_sample_list(df["BidVolSum"], period)
    df["AskSampled"] = generate_sample_list(df["AskVolSum"], period)
    df["BidSampled"] = df["BidSampled"].clip(0)
    df["AskSampled"] = df["AskSampled"].clip(0)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    plt.plot(priceData["LastPrice"].values)
    ax2 = fig.add_subplot(212)
    plt.plot((-1 * df["BidSampled"]).values)
    plt.plot(df["AskSampled"].values)
    plt.show()

    return df.loc[:, ["vol_dir_ratio"]]

def generate_sample(df, n):
    val = df.values
    col = []
    for index in range(len(val)):
        col.append(val[math.floor(index/n) * n][0])

    return pd.DataFrame(col, df.index)

def generate_sample_list(df, n):
    val = df.values
    col = []
    for index in range(len(val)):
        col.append(val[math.floor(index/n) * n])

    return col

# Count number of up tick/down tick in the period
def get_order_dir_count(hrb, period):
    priceData = hrb.get_hft_data().loc[:, ["AskPrice1", "BidPrice1", "TotalVolume", "Turnover",
                                           "LastPrice", "MidPrice", "AskSize1", "BidSize1"]]
    priceDataV = priceData.values
    contract_info = hrb.get_contract_data()

    multiplier = contract_info.multiplier

    vol_dir = [[0, 0]]
    abs_vol_dir = [0]
    last_row = []
    average_price = [0]

    for i in range(priceDataV.shape[0]):
        if priceDataV[i, 0] == 0 and priceDataV[i, 1] != 0:
            priceDataV[i, 0] = priceDataV[i, 1]
        elif priceDataV[i, 0] != 0 and priceDataV[i, 1] == 0:
            priceDataV[i, 1] = priceDataV[i, 0]
        elif priceDataV[i, 0] == 0:
            priceDataV[i, 0] = priceDataV[i - 1, 0]
            priceDataV[i, 1] = priceDataV[i - 1, 1]

    priceData = pd.DataFrame(priceDataV, columns=["AskPrice1", "BidPrice1", "TotalVolume", "Turnover",
                                           "LastPrice", "MidPrice", "AskSize1", "BidSize1"], dtype="float")

    for index, row in enumerate(priceDataV):
        if index == 0:
            last_row = row
            continue
        volume = (row[2] - last_row[2])/2
        cashVolume = (row[3] - last_row[3])/2/multiplier

        # if only 2 non-zero volume prices
        # non_zero_key[0] * x1 + non_zero_key[1] * x2 = Turnover
        # x1 + x2 = volume
        a = np.array([[last_row[0], last_row[1]], [1, 1]])
        b = np.array([cashVolume, volume])
        x= np.linalg.solve(a, b)


        if x[0] < 0:
            x = [0, volume]
        elif x[1] < 0:
            x = [volume, 0]
        #Bid, Ask
        vol_dir.append([x[1], x[0]])
        if x[1] < x[0]:
            abs_vol_dir.append(-1)
        elif x[1] > x[0]:
            abs_vol_dir.append(1)
        else:
            abs_vol_dir.append(0)

        if volume == 0:
            average_price.append(row[5])
        else:
            average_price.append(cashVolume/volume)
        last_row = row
    """
    fig, axs = plt.subplots(2, 1,  sharex="all")
    axs[0].plot(np.array(vol_dir)[:, 0])
    axs[0].plot(np.array(vol_dir)[:, 1] * -1)

    #axs[1].subplots(212, sharex="all")
    axs[1].plot(abs_vol_dir)
    plt.show()
    return
    """
    priceData["average_price"] = average_price
    priceData["weight_price"] = (priceData["BidPrice1"] * priceData["AskSize1"] +
                                 priceData["AskPrice1"] * priceData["BidSize1"])/(priceData["BidSize1"] + priceData["AskSize1"])
    #priceData["weight_price"].astype("float64")
    print(priceData.dtypes)
    #print(priceData["weight_price"].min())
    ###########################################
    priceData["abs_vol_sign"] = abs_vol_dir
    priceData["average_price_sign"] = np.sign(priceData["average_price"].diff())
    priceData["weight_price_sign"] = np.sign(priceData["weight_price"].diff())
    priceData["mid_price_sign"] = np.sign(priceData["MidPrice"].diff())
    priceData[["abs_vol_sign", "average_price_sign", "weight_price_sign", "mid_price_sign"]] = \
        priceData[["abs_vol_sign", "average_price_sign", "weight_price_sign", "mid_price_sign"]].fillna(value = 0)

    priceData["abs_vol_sum"] = priceData["abs_vol_sign"].rolling(period).sum()/period
    priceData["average_price_sum"] = priceData["average_price_sign"].rolling(period).sum()/period
    priceData["weight_price_sum"] = priceData["weight_price_sign"].rolling(period).sum()/period
    priceData["mid_price_sum"] = priceData["mid_price_sign"].rolling(period).sum()/period

    indicator_list = ["abs_vol_sum", "average_price_sum", "weight_price_sum", "mid_price_sum"]
    for ind in indicator_list:
        print(ind)
        qt = priceData[ind].quantile(0.9)
        print(qt)
        hrb.input_indicator(priceData.loc[:, [ind]], ind)
        hrb.indicator_observation(ind)
        hrb.indicator_distribution_observation(ind)
        hrb.generate_signal(ind, ind, qt, 'gap', 120, 'continuous')
        hrb.signal_show_response(ind, period=1000, direct="both")
        """
        print(ind)
        hrb.input_indicator(priceData.loc[:, [ind]], ind)
        hrb.indicator_observation(ind)
        hrb.indicator_distribution_observation(ind)
        hrb.indicator_linear_regression(ind, period=500)
        plt.close()
        """

"""
1. divide volume into buy vol and sell vol
2. calculate EMA/MA for each
3. Use these vol(s) to calculate the impacted price(s) in the current order book
4. Calculate late the "mid price" of the impacted bid/ask, compare that to the mid price of the OB

vol_side = 1 for index futures
"""
def impacted_price_simple_vol(hrb, period, vol_side = 2):
    priceData = hrb.get_hft_data().loc[:, ["BidPrice5", "BidPrice4", "BidPrice3", "BidPrice2", "BidPrice1",
             "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5", "BidSize5", "BidSize4",
                                           "BidSize3", "BidSize2", "BidSize1",
               "AskSize1", "AskSize2", "AskSize3", "AskSize4", "AskSize5", "TotalVolume", "Turnover",
                                           "LastPrice", "MidPrice"]]
    priceData.to_csv("Test.csv")
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step

    priceData.reset_index(inplace=True)
    # Volume and Turnover on tick
    priceData["vol"] = priceData["TotalVolume"].diff()/vol_side
    priceData["to"] = priceData["Turnover"].diff()/multiplier/vol_side
    priceData["vol"] = priceData["vol"].clip(0)
    priceData["to"] = priceData["to"].clip(0)

    # Need to process bad data?
    # df['colC'] = np.where(df['colA'] == 'a', df['colC'],0)

    weights = list(range(1, period + 1))

    bidVol, askVol = np.zeros(len(priceData.index)), np.zeros(len(priceData.index))
    impactedMidDiff = np.zeros(len(priceData.index))
    last_row = None
    for row in priceData.itertuples():
        if row.Index == 0:
            print("This is the first row")
            last_row = copy.deepcopy(row)
            continue

        if row.vol == 0:
            continue
        elif last_row.AskPrice1 == last_row.BidPrice1 == 0:
            continue
        elif last_row.AskPrice1 == 0:
            bidVol[row.Index] = row.vol
        elif last_row.BidPrice1 == 0:
            askVol[row.Index] = row.vol
        else:
            # Separate Volume here:
            tickBidVol = (row.vol * last_row.AskPrice1 - row.to) / (last_row.AskPrice1 - last_row.BidPrice1)
            tickAskVol = row.vol - tickBidVol


            if tickBidVol < 0:
                tickBidVol, tickAskVol = 0, row.vol
            elif tickAskVol < 0:
                tickBidVol, tickAskVol = row.vol, 0

            bidVol[row.Index], askVol[row.Index] = tickBidVol, tickAskVol


        if row.Index > period: # skip first (period + 1) rows for calculation
            # WMA to predict Volume on either side
            nextBidVol = np.average(bidVol[row.Index - period + 1:row.Index + 1], weights=weights) * 20
            nextAskVol = np.average(askVol[row.Index - period + 1:row.Index + 1], weights=weights) * 20

            bidVolCumSum = np.cumsum([row.BidSize1, row.BidSize2, row.BidSize3, row.BidSize4, row.BidSize5])
            askVolCumSum = np.cumsum([row.AskSize1, row.AskSize2, row.AskSize3, row.AskSize4, row.AskSize5])

            impactedBid, impactedAsk = 0, 0
            if round(nextBidVol) >= bidVolCumSum[4]:
                impactedBid = row.BidPrice5 - min_tick
            if round(nextAskVol) >= askVolCumSum[4]:
                impactedAsk = row.AskPrice5 + min_tick

            for index in range(5):
                if not impactedBid and round(nextBidVol) < bidVolCumSum[index]:
                    impactedBid = getattr(row, "BidPrice" + str(index + 1))

                if not impactedAsk and round(nextAskVol) < askVolCumSum[index]:
                    impactedAsk = getattr(row, "AskPrice" + str(index + 1))
            if nextBidVol > 2000:
                print([row.BidSize1, row.BidSize2, row.BidSize3, row.BidSize4, row.BidSize5])
                print(bidVolCumSum)
                print(nextBidVol)
                print(impactedBid)
                print(row.BidPrice1)
                print(row.BidPrice2)


            impactedMid = np.average([impactedBid, impactedAsk])
            impactedMidDiff[row.Index] = (impactedMid - row.MidPrice)/min_tick
        last_row = copy.deepcopy(row)

    signal_name = "impactedMid"
    pd_signal = pd.DataFrame(impactedMidDiff, priceData.index, [signal_name])
    print(pd_signal)
    hrb.input_indicator(pd_signal, signal_name)
    hrb.get_status()
    hrb.indicator_observation(signal_name)
    hrb.indicator_distribution_observation(signal_name)

# Evaluate predicted value with actual value
def evaluate_prediction(actual, pred, plot_size):
    print(len(actual))
    error = np.subtract(actual, pred)
    error = error[0:plot_size].astype(float)
    pred = np.array(pred[0:plot_size]).astype(float)
    vol = np.array(actual[0:plot_size]).astype(float)

    # only use the last 240 samples for the purpose of diagnostic plots
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(221)
    ax1.plot(pred, error, marker="o", ls="", markersize=6, alpha=0.5)
    ax1.set_title("Residuals vs Fitted")
    ax1.set_xlabel("Fitted values")
    ax1.set_ylabel("Residuals")
    ax1.axhline(y=0, linestyle="--", color="black")
    sns.regplot(pred, error,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    ax2 = fig.add_subplot(222)
    probplot(error, dist="norm", plot=ax2)
    ax2.set_title("Normal QQ plot for residuals")
    textstr = "\n".join((
        "MSE: " + str("%.4f" % mean_squared_error(vol, pred)),
        "R_squared: " + str("%.4f" % r2_score(vol, pred)))
    )
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ax3 = fig.add_subplot(223)
    sqrt_error = np.sqrt(np.abs(error))
    ax3.plot(pred, sqrt_error, marker="o", ls="", markersize=6, alpha=0.5)
    sns.regplot(pred, sqrt_error,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    ax3.set_title("Scale Location (heteroscedasticity test)")
    ax3.set_xlabel("Fitted values")
    ax3.set_ylabel("Residuals Sqrt")

    ax4 = fig.add_subplot(224)
    # ax4.plot(error)
    plot_acf(error, ax=ax4, title="Residuals Autocorrelation")
    ax4.set_xlabel("Lags")

    fig.suptitle("Diagnostic plots prediction", fontsize=16)
    plt.show()

    return 0

# Difference methods to predict volume
def predict_volume(hrb, interval, period, method, vol_side = 2):
    """
    :param hrb:
    :param interval: # of ticks's volume we want to predict
    :param period: period is use to compute ma, wma, ema (alpha)
    :param method:
                ar: auto-regressive model
                ma: moving average
                wma: weighted moving average
                ema: exponential weighted moving average
    :return:
    """
    priceData = hrb.get_hft_data().loc[:, ["TotalVolume",
                                           "Turnover"]]
    priceData.to_csv("Test.csv")
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step

    # Sub-sample data here based on interval
    priceData = priceData.iloc[::interval, :]

    priceData.reset_index(inplace=True)
    # Volume and Turnover on tick
    priceData["vol"] = priceData["TotalVolume"].diff() / vol_side
    priceData["to"] = priceData["Turnover"].diff() / multiplier / vol_side
    priceData["vol"] = priceData["vol"].clip(0)
    priceData["to"] = priceData["to"].clip(0)

    # Need to process bad data?
    # df['colC'] = np.where(df['colA'] == 'a', df['colC'],0)

    pred, vol = [], []
    if method == "ma":
        priceData["predicted"] = priceData["vol"].rolling(period).mean()
        priceData["predicted"].fillna(0, inplace=True)
        pred = priceData["predicted"].values[int(len(priceData.index)/2):].astype(float)
        vol = priceData["vol"].values[int(len(priceData.index)/2):].astype(float)

    elif method == "wma":
        weights = list(range(1, period + 1))
        priceData["vol"]
        priceDataV = priceData.loc[:, ["vol"]].values.astype(float)
        priceDataV = np.squeeze(priceDataV)
        wma_vol = []
        for index, row in enumerate(priceDataV):
            if index < period:
                wma_vol.append(0)
            else:
                wma_vol.append(np.average(priceDataV[index - period: index], weights=weights))
        priceData["predicted"] = np.array(wma_vol)
        pred = priceData["predicted"].values[int(len(priceData.index)/2):].astype(float)
        vol = priceData["vol"].values[int(len(priceData.index)/2):].astype(float)
    elif method == "ema":
        priceData["predicted"] = priceData["vol"].ewm(alpha=period).mean()
        priceData["predicted"].fillna(0, inplace = True)
        pred = priceData["predicted"].values[int(len(priceData.index)/2):].astype(float)
        vol = priceData["vol"].values[int(len(priceData.index)/2):].astype(float)
    elif method == "ar":
        plt.scatter(priceData["vol"], priceData["vol"].shift(1))
        plt.show()
        priceData["vol"].fillna(0, inplace = True)
        sm.graphics.tsa.plot_pacf(priceData["vol"].values[:int(len(priceData.index)/2)], lags=30)
        plt.show()
        print("Please input AR coeff: ")
        fit_val = int(input())
        fit_val = 50
        model = AR(priceData["vol"].values)
        model_fit = model.fit(fit_val)

        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)

        predictions = model_fit.predict(start=int(len(priceData.index)/2) + 1, end=len(priceData.index), dynamic=False)
        error = mean_squared_error(priceData["vol"].values[int(len(priceData.index)/2):], predictions)
        #plt.scatter(priceData["vol"].values[int(len(priceData.index)/2):], predictions)
        #plt.show()
        print('Test MSE: %.3f' % error)

        pred = predictions
        vol = priceData["vol"].values[int(len(priceData.index)/2):]

    evaluate_prediction(vol, pred, 2000)


"""
Starting here are my own thoughts
"""


"""
Here use volume residual a factor:
The idea is the number of ticks the market is willing to pay to execute whatever volume it desires,
as opposed to place orders.

For average price > ask, calculate vol residual with respect to bid
For average price < bid, vice verse
For average price in the middle, we ignore
"""
def volume_residual(hrb, period):
    """
    priceData = hrb.get_hft_data().loc[:, ["BidPrice5", "BidPrice4", "BidPrice3", "BidPrice2", "BidPrice1",
             "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5", "BidSize5", "BidSize4",
                                           "BidSize3", "BidSize2", "BidSize1",
               "AskSize1", "AskSize2", "AskSize3", "AskSize4", "AskSize5", "TotalVolume", "Turnover",
                                           "LastPrice", "MidPrice"]]
    """
    priceData = hrb.get_hft_data().loc[:, ["BidPrice1", "AskPrice1", "BidSize1", "AskSize1",
                                           "TotalVolume", "Turnover", "LastPrice", "MidPrice"]]
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step

    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1

    priceData.reset_index(inplace=True)
    # Volume and Turnover on tick
    priceData["vol"] = priceData["TotalVolume"].diff() / vol_side
    priceData["to"] = priceData["Turnover"].diff() / multiplier / vol_side
    priceData["vol"] = priceData["vol"].clip(0)
    priceData["to"] = priceData["to"].clip(0)

    priceData["LastBidPrice1"] = priceData["BidPrice1"].shift()
    priceData["LastAskPrice1"] = priceData["AskPrice1"].shift()

    def get_mean_price(row):
        if row["vol"] == 0:
            return 0
        else:
            return row["to"]/row["vol"]

    priceData["MeanPrice"] = priceData.apply(get_mean_price, axis = 1)

    def get_vol_resid(row):
        if row["vol"] == 0:
            return 0
        elif row["LastBidPrice1"] <= row["MeanPrice"] <= row["LastAskPrice1"]:
            return 0
        elif row["MeanPrice"] > row["LastAskPrice1"]:
            return (row["MeanPrice"] - row["LastAskPrice1"]) / min_tick * row["vol"]
        else:
            return -1 * (row["LastBidPrice1"] - row["MeanPrice"]) / min_tick * row["vol"]

    priceData["VolResid"] = priceData.apply(get_vol_resid, axis = 1)
    priceData["VolResid"] = priceData["VolResid"].rolling(20).mean()
    #print(priceData.head(10))

    hrb.input_indicator(priceData[["VolResid"]], "VolResid")
    hrb.indicator_observation('VolResid')
    hrb.indicator_distribution_observation('VolResid')

# Basic price_volume setup
def volume_price_diversion_1(hrb):
    priceData = hrb.get_hft_data().loc[:, ["BidPrice1", "AskPrice1", "BidSize1", "AskSize1",
                                           "TotalVolume", "Turnover", "LastPrice", "MidPrice"]]
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    priceData["hCount"] = hrb.get_hCount()
    # Whether volume and turnover are calculated single sided or double sided
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1

    #priceData = get_alternative_clock(priceData, multiplier, "volume", vol_side, 40)
    priceData = filter_volume(priceData, multiplier, vol_side)
    priceData["DeltaPrice"] = priceData["MidPrice"].diff()
    priceData["DeltaVol"] = priceData["vol"].diff()

    option = "1"
    if option == "1":
        # 1.
        priceDataV = priceData.loc[:, ["MidPrice", "vol", "to", "hCount"]].values
    elif option == "4b":
        # 4b
        priceDataV = priceData.loc[:, ["DeltaPrice", "vol", "to", "hCount"]].values


    priceDataV = priceDataV.astype(float)

    corr_ticks = 120
    last_hCount = -1
    hCount_counter = 1
    corr = []

    for index, row in enumerate(priceDataV):
        # Skip first n rows of the beginning of every hCount
        if row[3] != last_hCount and hCount_counter < corr_ticks:

            hCount_counter += 1
            corr.append(None)
        else:
            if row[3] != last_hCount and hCount_counter >= corr_ticks:
                hCount_counter = 1
                last_hCount = row[3]
            corr.append(np.corrcoef(
                priceDataV[index - 1 * corr_ticks + 1: index + 1, 0],
                priceDataV[index - 1 * corr_ticks + 1: index + 1, 1])[0, 1])

    priceData["MidVolumeCorr"] = np.array(corr, dtype = np.float)
    hrb.input_indicator(priceData[["MidVolumeCorr"]], "MidVolumeCorr")
    hrb.indicator_observation('MidVolumeCorr')
    hrb.indicator_distribution_observation('MidVolumeCorr')
    #hrb.indicator_linear_regression('MidVolumeCorr', period=100)

    p03 = priceData["MidVolumeCorr"].quantile(0.1)
    p97 = priceData["MidVolumeCorr"].quantile(0.9)
    hrb.generate_signal('MidVolumeCorr', 'MidVolumeCorrB', p97, 'gap', 200, 'intraday')
    hrb.generate_signal('MidVolumeCorr', 'MidVolumeCorrS', p03, 'gap', 200, 'intraday')
    hrb.signal_show_response('MidVolumeCorrB', period=50, direct="buy")
    hrb.signal_show_response('MidVolumeCorrS', period=50, direct="sell")

"""
 A more sophisticated improvement
 1. Signed Volume Correlationg with return
"""
def volume_price_diversion_4c(hrb):
    priceData = hrb.get_hft_data().loc[:, ["BidPrice1", "AskPrice1", "BidSize1", "AskSize1",
                                           "TotalVolume", "Turnover", "LastPrice", "MidPrice"]]
    contract_info = hrb.get_contract_data()
    multiplier = contract_info.multiplier
    min_tick = contract_info.step
    priceData["hCount"] = hrb.get_hCount()
    # Whether volume and turnover are calculated single sided or double sided
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1

    #priceData = get_alternative_clock(priceData, multiplier, "volume", vol_side, 40)
    priceData = filter_volume(priceData, multiplier, vol_side)
    signedVolume = get_signed_volume(hrb)
    priceData = pd.concat([priceData, signedVolume], axis = 1)

    priceData["deltaPrice"] = priceData["MidPrice"].diff()/min_tick # in jump
    priceData["deltaVol"] = priceData["vol"].diff()
    priceData["netVol"] = priceData["AskVol"] - priceData["BidVol"]


    #print(priceData[["deltaPrice", "netVol"]])
    print(priceData)

    priceDataV = priceData.loc[:, ["deltaPrice", "netVol", "to", "hCount"]].values
    priceDataV = priceDataV.astype(float)

    corr_ticks = 120
    last_hCount = -1
    hCount_counter = 1
    corr = []

    for index, row in enumerate(priceDataV):
        # Skip first n rows of the beginning of every hCount
        if row[3] != last_hCount and hCount_counter < corr_ticks:

            hCount_counter += 1
            corr.append(None)
        else:
            if row[3] != last_hCount and hCount_counter >= corr_ticks:
                hCount_counter = 1
                last_hCount = row[3]
            corr.append(np.corrcoef(
                priceDataV[index - 1 * corr_ticks + 1: index + 1, 0],
                priceDataV[index - 1 * corr_ticks + 1: index + 1, 1])[0, 1])

    priceData["MidVolumeCorr"] = np.array(corr, dtype = np.float)
    hrb.input_indicator(priceData[["MidVolumeCorr"]], "MidVolumeCorr")
    hrb.indicator_observation('MidVolumeCorr')
    hrb.indicator_distribution_observation('MidVolumeCorr')
    #hrb.indicator_linear_regression('MidVolumeCorr', period=100)

    p03 = priceData["MidVolumeCorr"].quantile(0.1)
    p97 = priceData["MidVolumeCorr"].quantile(0.9)
    hrb.generate_signal('MidVolumeCorr', 'MidVolumeCorrB', p97, 'gap', 200, 'intraday')
    hrb.generate_signal('MidVolumeCorr', 'MidVolumeCorrS', p03, 'gap', 200, 'intraday')
    hrb.signal_show_response('MidVolumeCorrB', period=200, direct="buy")
    hrb.signal_show_response('MidVolumeCorrS', period=200, direct="sell")


"""
Return reindexed DataFrame with Bid Volume and Ask Volume by solving volume and turnover
["BidVol", "AskVol"]
"""
def get_signed_volume(hrb):
    priceData = hrb.get_hft_data().loc[:, ["AskPrice1", "BidPrice1", "TotalVolume", "Turnover",
                                           "LastPrice", "MidPrice", "AskSize1", "BidSize1"]]

    priceDataV = priceData.values
    contract_info = hrb.get_contract_data()

    multiplier = contract_info.multiplier
    vol_side = 2 if hrb.tInfo.fSymbol not in ["IF", "IC", "IH"] else 1
    vol_dir = [[0, 0]]
    last_row = []

    for i in range(priceDataV.shape[0]):
        if priceDataV[i, 0] == 0 and priceDataV[i, 1] != 0:
            priceDataV[i, 0] = priceDataV[i, 1]
        elif priceDataV[i, 0] != 0 and priceDataV[i, 1] == 0:
            priceDataV[i, 1] = priceDataV[i, 0]
        elif priceDataV[i, 0] == 0:
            priceDataV[i, 0] = priceDataV[i - 1, 0]
            priceDataV[i, 1] = priceDataV[i - 1, 1]


    for index, row in enumerate(priceDataV):
        if index == 0:
            last_row = row
            continue
        volume = (row[2] - last_row[2]) / vol_side
        cashVolume = (row[3] - last_row[3]) / vol_side / multiplier

        # if only 2 non-zero volume prices
        # non_zero_key[0] * x1 + non_zero_key[1] * x2 = Turnover
        # x1 + x2 = volume
        a = np.array([[last_row[0], last_row[1]], [1, 1]])
        b = np.array([cashVolume, volume])
        x = np.linalg.solve(a, b)

        if x[0] < 0:
            x = [0, volume]
        elif x[1] < 0:
            x = [volume, 0]
        # Bid, Ask
        vol_dir.append([x[1], x[0]])

        last_row = row

    priceData.reset_index(inplace=True)

    return pd.DataFrame(vol_dir, priceData.index, ["BidVol", "AskVol"])

if __name__ == '__main__':
    st = dt.datetime(2019, 3, 15)
    et = dt.datetime(2019, 3, 30)
    #hrb = HRB.HRB(st, et, 'IF', '1904', 'l2_cffex', 0)
    hrb = HRB.HRB(st, et, 'rb', '1905', 'l1_ctp', 0)
    volume_price_diversion_4c(hrb)
    #volume_residual(hrb, 120)