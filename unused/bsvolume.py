from hft_rsys import *
import datetime as dt
import pandas as pd
import numpy as np
"""
this file is used to seperate volume to buy volume and sell volume

usually the total volume of buy and sell volume should be the tick volume

but sometimes bs volume could include new best size when best price has changed
"""


def get_bsvolume_by_mid(hrb):
    """
    get buy as sell volume on every tick by last mid price.
    assuming deals are all happenning on the two price beside mid price.
    cannot be used at zce.
    """
    hftData = hrb.get_hft_data().loc[:, ['AskPrice1', 'BidPrice1', 'TotalVolume', 'Turnover']]
    hftDataV = hftData.values
    #midPrice = hrb.get_hft_data().loc[:, ['MidPrice']]
    midPrice = hrb.get_hft_data().loc[:,['MidPrice']]
    midPriceV = midPrice.values
    bnp = np.full([hftDataV.shape[0], 1], np.nan)
    snp = np.full([hftDataV.shape[0], 1], np.nan)
    contract_info = hrb.get_contract_data()
    
    for i in range(1, bnp.shape[0]):
        lastMid = midPriceV[i - 1, 0]
        intertickVolume = hftDataV[i, 2] - hftDataV[i - 1, 2]
        intertickTurnover = hftDataV[i, 3] - hftDataV[i - 1, 3]
        if intertickVolume == 0:
            bnp[i, 0] = snp[i, 0] = 0
        elif intertickVolume > 0:
            bs = (intertickTurnover / intertickVolume / contract_info.multiplier - lastMid) / contract_info.step
            bs = 0.5 if bs > 0.5 else bs
            bs = -0.5 if bs < -0.5 else bs
            bnp[i, 0] = intertickVolume * (0.5 + bs)
            snp[i, 0] = bnp[i, 0] - intertickVolume 

    return pd.DataFrame(snp, hftData.index, columns=['sell volume']), pd.DataFrame(bnp, hftData.index, columns=['buy volume'])

def get_bsvolume_by_best(hrb):
    """
    get buy as sell volume on every tick by last ask and bid price.
    assuming deals are all happenning on last tick's ask and bid price
    cannot be used at zce.
    """
    hftData = hrb.get_hft_data().loc[:, ['AskPrice1', 'BidPrice1', 'TotalVolume', 'Turnover']]
    hftDataV = hftData.values
    #midPrice = hrb.get_hft_data().loc[:, ['MidPrice']]
    midPrice = hrb.get_hft_data().loc[:,['MidPrice']]
    midPriceV = midPrice.values
    bnp = np.full([hftDataV.shape[0], 1], np.nan)
    snp = np.full([hftDataV.shape[0], 1], np.nan)
    contract_info = hrb.get_contract_data()
    
    for i in range(1, bnp.shape[0]):
        lastMid = midPriceV[i - 1, 0]
        intertickVolume = hftDataV[i, 2] - hftDataV[i - 1, 2]
        intertickTurnover = hftDataV[i, 3] - hftDataV[i - 1, 3]
        spread=hftDataV[i,0]-hftDataV[i,1]
        if intertickVolume == 0:
            bnp[i, 0] = snp[i, 0] = 0
        elif intertickVolume > 0:
            bs = (intertickTurnover / intertickVolume / contract_info.multiplier - lastMid) / spread 
            bs = 0.5 if bs > 0.5 else bs
            bs = -0.5 if bs < -0.5 else bs
            bnp[i, 0] = intertickVolume * (0.5 + bs)
            snp[i, 0] = bnp[i, 0] - intertickVolume 

    return pd.DataFrame(snp, hftData.index, columns=['sell volume']), pd.DataFrame(bnp, hftData.index, columns=['buy volume'])

if __name__ == '__main__':
    st = dt.datetime(2018, 11, 1)
    et = dt.datetime(2018, 11, 2)
    hrb = HRB.HRB(st, et, 'i', '1901', 'l2_dce')
    sv, bv  = get_bsvolume_by_mid(hrb)
    hrb.input_indicator(bv, 'bvolume')
    hrb.input_indicator(sv, 'svolume')
    hrb.get_status()
    hrb.observation(price_names=['AskPrice1', 'BidPrice1'], show_list=[['bvolume', 'svolume']])
