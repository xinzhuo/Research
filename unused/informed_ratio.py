import numpy as np
import pandas as pd
import datetime as dt 
import sys
from base.bsvolume import get_bsvolume_by_mid
from hft_rsys import *

def get_informed_ratio_volume(hrb, period):
    """
    this function is used to get informed ratio.
    hrb:instance.
    period: time window to get informed ratio.
    """
    sv, bv = get_bsvolume_by_mid(hrb)
    svV = sv.values
    bvV = bv.values
    ir = np.zeros([svV.shape[0], 1])
    #hrb.input_indicator(sv, 'svolume')
    #hrb.input_indicator(bv, 'bvolume')

    ssum = 0
    bsum = 0

    for i in range(period):
        ssum += (svV[i, 0] if not np.isnan(svV[i, 0]) else 0)
        bsum += (bvV[i, 0] if not np.isnan(bvV[i, 0]) else 0)
    for i in range(period, svV.shape[0]):
        ssum += (svV[i, 0] if not np.isnan(svV[i, 0]) else 0)
        bsum += (bvV[i, 0] if not np.isnan(bvV[i, 0]) else 0)
        ssum -= (svV[i - period, 0] if not np.isnan(svV[i - period, 0]) else 0)
        bsum -= (bvV[i - period, 0] if not np.isnan(bvV[i - period, 0]) else 0)
        ir[i, 0] = (ssum + bsum)/(bsum - ssum) if (bsum - ssum) > 0.001 else 0
        assert(not np.isnan(bsum))

    return pd.DataFrame(ir, sv.index)


if __name__ == '__main__':
    st = dt.datetime(2018, 9, 1)
    et = dt.datetime(2018, 10, 2)
    hrb = HRB.HRB(st, et, 'rb', '1901', 'l1_ctp')
    ir100 = get_informed_ratio_volume(hrb, 100)
    #ir20 = get_informed_ratio_volume(hrb, 20)
    hrb.input_indicator(ir100, 'ir100')
    hrb.get_status()
    #hrb.observation(price_names=['AskPrice1', 'BidPrice1'], show_list=[['svolume', 'bvolume'], ['ir20'], ['ir100']])
    hrb.indicator_observation('ir100')
    hrb.indicator_distribution_observation('ir100')
    hrb.indicator_distribution_observation('ir100', plot_type='hist', hist_range=[-1,1], hist_precision=0.1)
    hrb.indicator_linear_regression('ir100',period=100)
    hrb.indicator_nonpara_regression('ir100',period=100)
    hrb.generate_signal('ir100', 'ir100s', 0.8, 'independent', 200, 'intraday')
    hrb.signal_show_response('ir100s', period=200)
