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


# plt.figure(figsize=(7.5, 5))
# plt.acorr(agg, maxlags=24)
# plt.title('GIST CAMPUS')
# plt.xlabel('Lag')
# plt.xticks(ticks=[i for i in range(-24, 25, 12)])
# plt.show()
# plt.savefig(f'D:/2020_ETRI/200804_monthly/autocorr_lag24_GIST.png')


plt.rcParams.update({'font.size': 13})
plt.figure()

plt.plot(acf(agg, nlags=24, fft=False))
plt.ylim([0, 1.1])


test_house_list = ['68181c16', '0098d3ee', '1bce71e8']

legends = ['GIST campus']
for test_house in test_house_list:
    data_raw = load_labeled()
    data, nan_data = clear_head(data_raw)
    data_col = data[test_house]
    legends.append(test_house)

    # plt.figure(figsize=(7.5, 5))
    # plt.acorr(data_col.fillna(0), maxlags=24, normed=False)
    # plt.title(f'Autocorrelation - {test_house}')
    # plt.xlabel('Lag')
    # plt.show()
    # plt.savefig(f'D:/2020_ETRI/200804_monthly/autocorr_lag24_{test_house}.png')

    plt.plot(acf(data_col.fillna(0), nlags=24, fft=False))

plt.legend(legends, loc='upper right', fontsize=10)
plt.ylim([0, 1.1])
plt.title('Autocorrelation')
plt.xticks(ticks=[i for i in range(0, 25, 6)])
plt.xlabel('Lags')
plt.tight_layout()
