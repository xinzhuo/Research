from hft_rsys import *
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from scipy import stats

def run_ar_model(hrb):
    hftData = hrb.get_hft_data().loc[:, ['AskPrice1', 'BidPrice1', 'TotalVolume', 'Turnover', 'MidPrice']]
    contract_info = hrb.get_contract_data()
    # Needed to be update
    min_step = contract_info.step
    print(len(hftData.index.values))
    hftData["tick_ret"] = (hftData["MidPrice"] - hftData["MidPrice"].shift(1))/min_step
    hftData.dropna(subset=["tick_ret"], inplace=True)
    #hftData["tick_ret"].plot()
    #plt.scatter(hftData["tick_ret"], hftData["tick_ret"].shift(1))
    #hftData["tick_ret"].plot()
    #plt.plot(hftData["tick_ret"].values)
    print(hftData["tick_ret"].values)
    #sm.graphics.tsa.plot_pacf(hftData["tick_ret"].values.squeeze(), lags = 30)

    X = hftData["tick_ret"].values
    train, test = X[:len(X) - 20], X[len(X) - 20:]

    model = AR(train)
    model_fit = model.fit(10)

    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)

    #predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=True)
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=True)
    #for i in range(len(predictions)):
    #    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot results
    plt.plot(np.array(test).cumsum())
    plt.plot(np.array(predictions).cumsum(), color='red')
    plt.show()

if __name__ == '__main__':
    st = dt.datetime(2019, 3, 19, 10, 0, 0)
    et = dt.datetime(2019, 3, 20, 8, 0, 0)
    hrb = HRB.HRB(st, et, 'IF', '1905', 'l1_ctp', 0)
    run_ar_model(hrb)

