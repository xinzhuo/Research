import numpy as np
import pandas as pd
import datetime as dt 
from hft_rsys import *

"""
this lib is used to calculate tick average price
"""

def get_tick_vavp(hrb):
    """
    this function is used to get average price by volume at every tick, if no volume exists, the values is nan
    """ 
    hftData = hrb.get_hft_data().loc[:, ['TotalVolume', 'Turnover']]
    hftDataV = hftData.values
    avp=np.full([hftDataV.shape[0], 1], np.nan)
    contract_info = hrb.get_contract_data()
 
    if hftDataV[0, 0] > 0.001:
        avp[0, 0] = hftDataV[0, 1] / hftDataV[0, 0] / contract_info.multiplier
    for i in range(1, hftDataV.shape[0]):
        if hftDataV[i, 0] - hftDataV[i - 1, 0] > 0.001:
            avp[i, 0] = (hftDataV[i, 1] - hftDataV[i - 1, 1]) / (hftDataV[i, 0] - hftDataV[i - 1, 0]) / contract_info.multiplier

    return pd.DataFrame(avp, hftData.index, ['avp'])


def get_tick_vavp_complete(hrb):
    """
    this function is used to get average price by volume at every tick, if no volume exists, the values is equal to last value
    """ 
    hftData = hrb.get_hft_data().loc[:, ['TotalVolume', 'Turnover']]
    hftDataV = hftData.values
    avp=np.full([hftDataV.shape[0], 1], np.nan)
    contract_info = hrb.get_contract_data()
 
    if hftDataV[0, 0] > 0.001:
        avp[0, 0] = hftDataV[0, 1] / hftDataV[0, 0] / contract_info.multiplier
    for i in range(1, hftDataV.shape[0]):
        if hftDataV[i, 0] - hftDataV[i - 1, 0] > 0.001:
            avp[i, 0] = (hftDataV[i, 1] - hftDataV[i - 1, 1]) / (hftDataV[i, 0] - hftDataV[i - 1, 0]) / contract_info.multiplier
        else:
            avp[i, 0] = avp[i - 1, 0]
 
    return pd.DataFrame(avp, hftData.index, ['avp'])
