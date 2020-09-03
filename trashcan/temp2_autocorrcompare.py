import pandas as pd
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.stattools import acf, pacf
from scipy import io
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from funcs import *


mat = io.loadmat('../GRI/CP_use_h_15~18.mat')
agg = mat[f'CP_use_h_2015'][:, :, 0]
agg = agg.transpose().reshape((agg.shape[0]*agg.shape[1],))

for i in range(2016, 2019):
    temp = mat[f'CP_use_h_{i}'][:, :, 0]
    a = temp.transpose().reshape((temp.shape[0]*temp.shape[1],))
    agg = np.append(agg, a)

ticker_data = pd.Series(agg)

ticker_data_acf_1 = acf(ticker_data)[1:32]
ticker_data_acf_2 = [ticker_data.autocorr(i) for i in range(1,32)]

test_df = pd.DataFrame([ticker_data_acf_1, ticker_data_acf_2]).T
test_df.columns = ['Pandas Autocorr', 'Statsmodels Autocorr']
test_df.index += 1
test_df.plot(kind='bar')
