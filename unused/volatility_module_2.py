import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from datetime import datetime, date, time, timedelta
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import probplot

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import math
from hft_rsys import *

def plot_ret_dist(contract_id, contract_month, start_time, end_time, 
    interval=4, log=False, mid=True, rolling=False, num_tick=5):
    
    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, 'l1_ctp')
    contract_info = hrb.get_contract_data()
    min_tick = contract_info.step
    df = hrb.get_hft_data()

    """
    :param df: dataframe to be passed in
    :param interval: interval to calculate return
    :param log: if use log return instead of tick jump
    :param mid: if use mid price instead of last price
    :param rolling: if use rolling return for interval > 1
    :param num_tick: number of tick to see on the plot (one side)
    :param min_tick: minimum tick size of the contract
    :return:
    """
    #min_tick = 0.5 # Need to update here
    if interval + 1 > len(df.index):
        print("Chosen interval too large.")
        return

    if mid:
        df["fair"] = (df["BidPrice1"] + df["AskPrice1"])/2
        order = np.arange(-1* num_tick, num_tick + 1/2, 1/2)
    else:
        df["fair"] = df["LastPrice"]
        order = np.arange(-1* num_tick, num_tick + 1, 1)

    if not log:
        df["ret"] = np.rint(df["fair"].diff(periods=interval)*2/min_tick)/2
    else:
        df["ret"] = np.log(df['fair']).diff(periods=interval)
    df = df[np.isfinite(df["ret"])]

    series = df["ret"].tolist()
    if not rolling:
        series = series[::interval]
    ncount = len(series)
    print(series)
    if not log:
        plt.figure(figsize=(12, 8))
        plt.title('Distribution of Tick Jump')
        plt.xlabel('Tick')
        ax = sns.countplot(series, order=order, palette="Blues_d")
        ax2 = ax.twinx()
        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')
        ax2.set_ylabel('Frequency [%]')
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                        ha='center', va='bottom')  # set the alignment of the text
        # Use a LinearLocator to ensure the correct number of ticks
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        # Fix the frequency range to 0-100
        ax2.set_ylim(0, 100)
        ax.set_ylim(0, ncount)
        # And use a MultipleLocator to ensure a tick spacing of 10
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
        st = ""
        if math.isnan(kurtosis(series)):
            st = "NaN"
        else:
            st = str(round(kurtosis((series))))
        textstr = "\n".join((
            "Mean: " + str(round(np.mean(series),2)),
            "Variance: " + str(round(np.var(series),2)),
            "Skewness " + str(round(skew(series),2)),
        "Kurtosis: " + st,)
        )
        props = dict(boxstyle='square', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    else:
        plt.title('Distribution of Tick return')
        plt.xlabel('Return')
        ax = sns.distplot(series, 50, kde=False)
        ax.set_ylabel('Count')
        textstr = "\n".join((
            "Mean: " + str(round(np.mean(series), 2)),
            "Variance: " + str(round(np.var(series), 2)),
            "Skewness " + str(round(skew(series), 2)),
            "Kurtosis: " + str(round(kurtosis((series)))),)
        )
        props = dict(boxstyle='square', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

    plt.show()

def evaluate_vol_prediction(contract_id, contract_month, start_time, end_time, pred_list, 
    bucket_size=240, sample_size = 120, fair="mid", plot_size=240):
    """
    This method compute the realized volatility with given frequency and method.
    The key is to eliminate microstructure noise i.e. bid ask bounce and discrete quote.
    This module doesn't deal with missing tick
    :param df:
    :param bucket_size: choose higher to avoid microstructure noise, usually 5 mins
    :param fair: choices of fair value
                mid:
                last:
                weighted: price weighted by first level bidsize and asksize
    :param plot_size: last n samples to be used in plots, as too many samples cause plot unrecognizable
    :param pred_list: predicted volatility as a list, should be very careful to line up with start and end time
    :return: Realized Volatility
    """
    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, 'l1_ctp')
    contract_info = hrb.get_contract_data()
    min_tick = contract_info.step
    df = hrb.get_hft_data()

    if fair == "mid":
        df["fair"] = (df["BidPrice1"] + df["AskPrice1"])/2
    elif fair == "last":
        df["fair"] = df["LastPrice"]
    elif fair == "weighted":
        df["fair"] = (df["BidPrice1"] * df["AskSize1"] + df["AskPrice1"] * df["BidSize1"])\
                     / (df["BidSize1"] + df["AskSize1"])
    df = df.replace({'fair': {0: np.nan}}).ffill()
    df["ret"] = df["fair"].pct_change(periods=bucket_size)    
    df = df[np.isfinite(df["ret"])]
    series = df["ret"].tolist()
    nrow = len(series)
    index_list = np.arange(0, bucket_size * sample_size, bucket_size)

    var_list = []
    set_break = False
    for index, val in enumerate(series):
        sq_sum = 0
        for idx in index_list:
            if index + idx >= nrow:
                set_break = True
                break

            sq_sum += np.square(series[index + idx])
        if set_break:
            break
        var_list.append(sq_sum)
        
    #error_list = np.subtract(np.random.rand(len(var_list)), 0.5)
    #pred_list = np.multiply(var_list, (error_list/100 + 1))
    error_list = np.subtract(var_list, pred_list)

    error = error_list[0:plot_size]
    pred = pred_list[0:plot_size]
    vol = var_list[0:plot_size]
    
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
    #ax4.plot(error)
    plot_acf(error, ax=ax4, title="Residuals Autocorrelation")
    ax4.set_xlabel("Lags")

    fig.suptitle("Diagnostic plots for Volatility prediction", fontsize=16)
    plt.show()

    return 0

def compute_realized_vol(contract_id, contract_month, start_time, end_time, bucket_size=40, 
    sample_size = 30, method="return", fair="last", intraday_fee=False, plot = True):
    """
    This method compute the realized volatility with given frequency and method.
    The key is to eliminate microstructure noise i.e. bid ask bounce and discrete quote.
    This module doesn't deal with missing tick
    :param df:
    :param bucket_size: choose higher to avoid microstructure noise, usually 5 mins
    :param method:
                return: percentage return between two timestamps
                tick: change in steps
                fee_adjusted: return adjusted with trading cost
    :param fair: choices of fair value
                mid:
                last:
                weighted: price weighted by first level bidsize and asksize
    :return: Realized Volatility

    """
    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, 'l1_ctp')
    contract_info = hrb.get_contract_data()
    min_tick = contract_info.step
    df = hrb.get_hft_data()
    trading_hr_id = hrb.get_hCount()
    feetype = contract_info.feetype 
    fee = contract_info.fee
    fee_intraday = 2#contract_info.fee_intraday

    if fair == "mid":
        df["fair"] = (df["BidPrice1"] + df["AskPrice1"])/2
    elif fair == "last":
        df["fair"] = df["LastPrice"]
    elif fair == "weighted":
        df["fair"] = (df["BidPrice1"] * df["AskSize1"] + df["AskPrice1"] * df["BidSize1"])\
                     / (df["BidSize1"] + df["AskSize1"])

    df = df.replace({'fair': {0: np.nan}}).ffill()
    df["hCount"] = trading_hr_id

    mat = df.loc[:, ["fair", "hCount"]].values

    pct_change_list = []
    abs_change_list = []
    for index, row in enumerate(mat):

        if index < bucket_size or row[1] != mat[index-bucket_size][1]:
            pct_change_list.append(float("nan"))
            abs_change_list.append(float("nan"))
        else:
            pct_change_list.append(row[0] / mat[index-bucket_size][0] - 1)
            abs_change_list.append(row[0] - mat[index-bucket_size][0])

    pct_change_list = np.asarray(pct_change_list)
    abs_change_list = np.asarray(abs_change_list)

    if method == "return":
        ret_list = pct_change_list 
    elif method == "tick":
        ret_list = abs_change_list  / min_tick
    elif method == "fee_adjusted":
        if intraday_fee and feetype == "hand": 
            ret_list = abs_change_list / (fee + fee_intraday)
        elif not intraday_fee and feetype == "hand": 
            ret_list = abs_change_list  / (fee * 2)
        elif  intraday_fee and feetype == "ratio": 
            ret_list = pct_change_list  / (fee * 2 / 10000)
        elif not intraday_fee and feetype == "ratio": 
            ret_list = pct_change_list  / ((fee + fee_intraday) / 10000)

    og_count = len(trading_hr_id)

    index_list = np.arange(0, bucket_size * sample_size, bucket_size)
    nrow = len(ret_list)
    var_list = []
    if method == "return":
        i = 0
        while i < len(ret_list):
            if math.isnan(ret_list[i]):
                var_list.append(float('nan'))
                i += 1
            elif math.isnan(ret_list[index - 1]): # when return on last spot was nan
                i += bucket_size * (sample_size - 1)
                var_list = var_list + [float('nan')] * (bucket_size * (sample_size - 1))
            else:
                sq_sum = 0
                for idx in index_list:
                    sq_sum += np.square(ret_list[i - idx])
                var_list.append(sq_sum)
                i += 1

    else:
        i = 0
        while i < len(ret_list):
            if math.isnan(ret_list[i]):
                var_list.append(float('nan'))
                i += 1
            elif math.isnan(ret_list[index - 1]): # when return on last spot was nan
                i += bucket_size * (sample_size - 1)
                var_list = var_list + [float('nan')] * (bucket_size * (sample_size - 1))
            else:
                samples = []
                for idx in index_list:
                    samples.append(ret_list[i - idx])
                var_list.append(np.std(samples))
                i += 1

    
    pp = np.nanpercentile(var_list, 99) * 20
    #print("99.9 Percentile: " + str(pp))
    for index, item in enumerate(var_list):
        if item > pp:
            var_list[index] = var_list[index - 1]
    if plot:
        plt.plot(var_list)
        plt.title(str(int(bucket_size * sample_size / 4)) + " seconds Realized Volatility")
        plt.show()

    return var_list

def compute_price_stdev(contract_id, contract_month, start_time, end_time,  
    bucket_size=240, sample_size=60, method="median", fair="mid"):

    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, 'l1_ctp')
    contract_info = hrb.get_contract_data()
    min_tick = contract_info.step
    df = hrb.get_hft_data()
    """
    The method allows user to first choose methods to sample within chosen bucket and calculate corresponding
    price stdev
    :param df: dataframe to be passed
    :param bucket_size: choose higher to avoid microstructure noise, usually 5 mins
    :param method: ways to sample within bucket, one of the following
                open:
                close:
                average: default
                median:
    :param fair: choices of fair value
                mid: default
                last:
                weighted: price weighted by first level bidsize and asksize
    :return: 
    """
    if fair == "mid":
        df["fair"] = (df["BidPrice1"] + df["AskPrice1"]) / 2
    elif fair == "last":
        df["fair"] = df["LastPrice"]
    elif fair == "weighted":
        df["fair"] = (df["BidPrice1"] * df["AskSize1"] + df["AskPrice1"] * df["BidSize1"]) \
                     / (df["BidSize1"] + df["AskSize1"])

    df = df.replace({'fair': {0: np.nan}}).ffill()

    if method == "open":
        df["sample"] = df["fair"].shift(periods=bucket_size)
    elif method == "close":
        df["sample"] = df["fair"]
    elif method == "average":
        df["sample"] = df["fair"].rolling(bucket_size).mean()
    elif method == "median":
        df["sample"] = df["fair"].rolling(bucket_size).median()
    # Deal with when bid or ask is 0
    
    df = df[np.isfinite(df["sample"])]
    series = df["sample"].tolist()
    nrow = len(series)
    offset_list = np.arange(0, bucket_size * sample_size, bucket_size)

    stdev_list = []
    set_break = False
    for index, val in enumerate(series):
        sample_list = []
        for idx in offset_list:
            if index + idx >= nrow:
                set_break = True
                break
            sample_list.append(series[index + idx])
        if set_break:
            break
        stdev_list.append(np.std(sample_list))

    stdev_list = [math.nan] * (df.shape[0] - len(stdev_list)) + stdev_list
    df["stdev"] = np.array(stdev_list)
    plt.plot(df["stdev"].tolist())
    plt.title("Standard Deviation of Price")
    plt.grid(True)
    plt.show()
    return(stdev_list)

def assess_trade_opportunity(contract_id, contract_month, start_time, end_time, 
    window_size=240, direction="buy", order_type=("p", "a"), num_tick = 10):
    """
    :param df:
    :param window_size: number of tick to look into future
    :param direction: buy or sell order
    :param order_type: a tuple, indicting type of order when enter position (first element) and exit position (second)
                    "p": passive order
                    "a": aggressive order
    :param min_tick: minimum tick size for the contract
    :param num_tick: number of tick to see in the plot
    :return:
    """

    hrb = HRB.HRB(start_time, end_time, contract_id, contract_month, 'l1_ctp')
    contract_info = hrb.get_contract_data()
    min_tick = contract_info.step
    df = hrb.get_hft_data()

    if direction == "buy":
        if order_type[0] == "a": #Aggressive buy
            df["w_start"] = df["AskPrice1"].shift(window_size-1)
        else:
            df["w_start"] = df["BidPrice1"].shift(window_size-1)

        if order_type[1] == "a": #Aggressive sell
            df["w_max"] = df["BidPrice1"].rolling(window_size).max()
        else:
            df["w_max"] = df["AskPrice1"].rolling(window_size).max()

        df["profit"] = df["w_max"] - df["w_start"]
    elif direction == "sell":
        if order_type[0] == "a":  # Aggressive sell
            df["w_start"] = df["BidPrice1"].shift(window_size-1)
        else:
            df["w_start"] = df["AskPrice1"].shift(window_size-1)

        if order_type[1] == "a":  # Aggressive buy
            df["w_min"] = df["AskPrice1"].rolling(window_size).min()
        else:
            df["w_min"] = df["BidPrice1"].rolling(window_size).min()

        df["profit"] = df["w_start"] - df["w_min"]

    df = df[np.isfinite(df["profit"])]
    df["profit"] = np.rint(df["profit"]*2/min_tick)/2
    series = df["profit"].tolist()
    order = np.arange(-1 , num_tick + 1, 1)

    plt.figure(figsize=(12, 8))
    plt.title('Distribution of Max Profit Tick Jump')
    plt.xlabel('Tick')
    ax = sns.countplot(series, order=order, palette="Blues_d")
    ax2 = ax.twinx()
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('Frequency [%]')
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.1f}%'.format(100. * y / len(series)), (x.mean(), y),
                    ha='center', va='bottom')  # set the alignment of the text
    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))
    # Fix the frequency range to 0-100
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, len(series))
    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.show()

def predict_volatility(contract_id, contract_month, start_time, end_time, model="EWMA",
    bucket_size=40, sample_size = 30, method="tick", fair="last", intraday_fee=False):
    """
    This method compute the realized volatility with given frequency and method.
    The key is to eliminate microstructure noise i.e. bid ask bounce and discrete quote.
    This module doesn't deal with missing tick
    :param df:
    :param bucket_size: choose higher to avoid microstructure noise, usually 5 mins
    :param method:
                return: percentage return between two timestamps
                tick: change in steps
                fee_adjusted: return adjusted with trading cost
    :param model:
                arima: time series model
                MA: moving average
                EWMA: exponential weighted moving average
    :param fair: choices of fair value
                mid:
                last:
                weighted: price weighted by first level bidsize and asksize
    :return: Realized Volatility

    """
    print("Computing Training data volatility")
    # Here change the time to previous period as training data
    train_period = compute_realized_vol(contract_id, contract_month, start_time,
        end_time, bucket_size, sample_size, method, fair, intraday_fee, plot = False)
    train_period = pd.DataFrame(train_period, columns = ["Vol"])
    # First create a df without NAs, use it to train and predict and put it back
    train_period_nna_df = train_period.dropna()
    train_period_nna = train_period_nna_df.reset_index()
    del train_period_nna["index"]

    print(model)
    if model == "arima":
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(train_period_nna.values.squeeze(), lags=80, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(train_period_nna, lags=80, ax=ax2)
        plt.show()
        
        param_p = int(input("Param p for ARMA: "))
        param_q = int(input("Param q for ARMA: "))
        arma_mod = sm.tsa.ARMA(train_period_nna, (param_p, param_q)).fit(disp=False)
        resid = arma_mod.resid
        print(arma_mod.params)
        sm.stats.durbin_watson(arma_mod.resid.values)
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax = arma_mod.resid.plot(ax=ax)
        plt.title("Residuals")
        plt.show()

        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=20, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(resid, lags=20, ax=ax2)
        plt.title("Residuals ACF and PACF")
        plt.show()

        train_period_nna["predicted"] = arma_mod.predict(0, 52335)
        train_period_nna_df["predicted"] = train_period_nna["predicted"].values
        train_period_nna_df["predicted"].plot(alpha = 0.4)
        train_period_nna_df["Vol"].plot(alpha=0.4)
        plt.title("Predicted vs Realized")
        plt.show()
        train_period["predicted"] = train_period_nna_df["predicted"]
        return train_period["predicted"]
    elif model == "MA":
        coeff = int(input("Choose window size: "))
        train_period_nna["predicted"] = train_period_nna["Vol"].rolling(coeff).mean()
        train_period_nna["Vol"].shift(-1).plot(alpha = 0.5)
        train_period_nna["predicted"].plot(alpha = 0.5)
        plt.legend(["actual", "predict"])
        plt.title("Actual vs. Predicted")
        plt.show()
        train_period_nna_df["predicted"] = train_period_nna["predicted"].values
        train_period["predicted"] = train_period_nna_df["predicted"]
        return train_period["predicted"]
    elif model == "EWMA":
        coeff = int(input("Choose coeffcient for EWMA: "))
        train_period_nna["predicted"] = train_period_nna["Vol"].ewm(com=coeff).mean()
        train_period_nna["Vol"].shift(-1).plot(alpha = 0.5)
        train_period_nna["predicted"].plot(alpha = 0.5)
        plt.legend(["actual", "predict"])
        plt.title("Actual vs. Predicted")
        plt.show()
        train_period_nna_df["predicted"] = train_period_nna["predicted"].values
        train_period["predicted"] = train_period_nna_df["predicted"]
        return train_period["predicted"]
        


    

if __name__ == '__main__':

    starttime = datetime(2018, 11, 6, 9, 0, 0)
    endtime = datetime(2018, 11, 6, 10, 0, 0)
    compute_realized_vol('ni', '1901', starttime, endtime)
    #plot_ret_dist('ni', '1901', starttime, endtime)
    #compute_price_stdev('ni', '1901', starttime, endtime)
    #assess_trade_opportunity('ni', '1901', starttime, endtime)
    #evaluate_vol_prediction(b'ni', b'1901', starttime, endtime)
    #predict_volatility('ni', '1901', starttime, endtime)
