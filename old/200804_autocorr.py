from funcs import *
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mape(A, F):
    return 100 / len(A) * np.sum((np.abs(A - F))/A)


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def inject_nan_imputation(d_col, n_mask, n_len=3):
    # d_col에서 n_len=3인 window를 moving해서 모든 데이터를 출력해주자. output: 2D data, 2D mask
    # forward, backward 각 4일씩 넣을 거니까... 앞 뒤 5주~-5주까지 범위로. -> 이건 그냥 후처리 해주자...
    data_injected = []
    mask_injected = []
    d_temp = d_col.copy()
    n_temp = n_mask.copy()
    i = 0
    while i <= len(d_col)-n_len-1:
        if sum(n_mask.values[i:i+n_len+1]) == 0:      # inject할 범위+1 중 nan이 없음 (+1 안 하면 nan 길이가 늘어나니까)
            d_temp[i:i+n_len] = np.nan
            n_temp[i:i+n_len] = 2
            d = d_temp.values
            n = n_temp.values.reshape(len(n_temp),)
            data_injected.append(d)
            mask_injected.append(n)

            d_temp = d_col.copy()
            n_temp = n_mask.copy()
            i += 1
        else:
            i += 1
    data_injected_nd = np.array(data_injected).transpose()
    mask_injected_nd = np.array(mask_injected).transpose()
    return data_injected_nd, mask_injected_nd


def make_bidirectional_input(d_col, n_mask):
    # forward 4주, backward 4주
    # 너무앞 or 너무뒤 or nan 많아서 4주 안 채워지면 그 만큼을 뒤 or 앞에서 더 채워넣음
    d = 48

    idx_inj = np.array(np.where(n_mask == 2))[0]
    train_x_fwd, train_x_bwd, train_y_temp, test_x_temp = [], [], [], []

    # 앞 4주, 뒤 4주 -> train_x에 포함시킬 24point에 nan이 절반 이상이면 pass시킴
    i = 168
    while len(train_x_fwd) < 8:
        # before
        idx_temp = idx_inj - i
        if (idx_temp[0]-(d+1) < 0) | (idx_temp[-1]+(d+1) > len(d_col)):
            pass
        elif (sum(n_mask[idx_temp[0]-d:idx_temp[0]]) > 10*(d/24)) | (sum(n_mask[idx_temp[-1]+1:idx_temp[-1]+(d+1)]) > 10*(d/24)):
            pass
        else:
            train_y_temp.append(d_col[idx_temp])
            train_x_fwd.append(d_col[idx_temp[0]-d:idx_temp[0]])
            train_x_bwd.append(d_col[idx_temp[-1]+1:idx_temp[-1]+(d+1)])

        # after
        idx_temp = idx_inj + i
        if (idx_temp[0]-(d+1) < 0) | (idx_temp[-1]+(d+1) > len(d_col)):
            pass
        elif (sum(n_mask[idx_temp[0]-d:idx_temp[0]]) > 10*(d/24)) | (sum(n_mask[idx_temp[-1]+1:idx_temp[-1]+(d+1)]) > 10*(d/24)):
            pass
        else:
            train_y_temp.append(d_col[idx_temp])
            train_x_fwd.append(d_col[idx_temp[0]-d:idx_temp[0]])
            train_x_bwd.append(d_col[idx_temp[-1]+1:idx_temp[-1]+(d+1)])
        i += 168
    test_x_temp = np.append(d_col[idx_inj[0]-d:idx_inj[0]], d_col[idx_inj[-1]+1:idx_inj[-1]+(d+1)])

    train_x = np.append(np.array(train_x_fwd), np.array(train_x_bwd), axis=1)
    # train_y, test_x = np.array(train_y_temp), test_x_temp.reshape((1, len(test_x_temp)))
    train_y, test_x = np.array(train_y_temp), test_x_temp.copy()

    train_x, train_y, test_x = np.nan_to_num(train_x), np.nan_to_num(train_y), np.nan_to_num(test_x)
    return train_x, train_y, test_x


def linear_prediction(train_x, train_y, test_x, f_len_fwd, f_len_bwd, n_len=3):
    d = 48

    len_tr = len(train_x[0, :])  # 시간 포인트 수
    day_t = len(train_x)
    prediction = np.empty((len(train_x), n_len))
    # fcst = np.empty((len(train_x), len_tr))

    for j in range(0, day_t):
        if day_t > 1:
            x_ar = np.delete(train_x[:, d-f_len_fwd:d+f_len_bwd], j, axis=0)
            y = np.delete(train_y, j, axis=0)
        else:
            x_ar = train_x[:, d-f_len_fwd:d+f_len_bwd]
            y = train_y

        pi_x_ar = np.linalg.pinv(x_ar)
        # lpc_c = np.empty((len(x_ar), f_len))

        lpc_c = np.matmul(pi_x_ar, y)

        test_e = train_x[j, :]
        test_ex = test_e[d-f_len_fwd:d+f_len_bwd]
        prediction[j, :] = np.matmul(test_ex, lpc_c)

    x_ar = train_x[:, d-f_len_fwd:d+f_len_bwd]
    y = train_y
    pi_x_ar = np.linalg.pinv(x_ar)
    # lpc_c = np.empty((len(x_ar), f_len))

    lpc_c = np.matmul(pi_x_ar, y)

    test_ar = train_y[0:len(train_y), :]

    # average_smape = []
    # smape_list = np.zeros((len(prediction), 1))
    # mse_list = np.zeros((len(prediction), 1))

    # for i in range(0, len(prediction)):
    #     smape_list[i] = smape(prediction[i, :], test_ar[i, :])
    #     average_smape = np.mean(smape_list)
    #     mse_list[i] = mean_squared_error(prediction[i, :], test_ar[i, :])

    test_e = test_x
    test_ex = test_e[d-f_len_fwd:d+f_len_bwd]
    forecast = np.matmul(test_ex, lpc_c)

    return forecast, prediction


##################################################
# set variables
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96', '4250dcd0']
# test_house = '1dcb5feb'

test_house_list = ['68181c16', '0098d3ee', '1bce71e8']
# test_house = '68181c16'
# test_house = '0098d3ee'


# test_house = '8d3d1b3a'   # bad case
# test_house = '18d9ac24'   # bad case
# test_house = '19ba2aab'   # bad case
# test_house = '9bf6e207'   # good case


##################################################
# load data
from statsmodels.tsa.stattools import acf
legends = []
for test_house in test_house_list:
    data_raw = load_labeled()
    data, nan_data = clear_head(data_raw)
    #
    # for i in range(data.shape[1]):
    #     test_house = data.columns[0]
    data_col = data[test_house]
    legends.append(test_house)

    # plt.figure(figsize=(7.5, 5))
    # plt.acorr(data_col.fillna(0), maxlags=24, normed=False)
    # plt.title(f'Autocorrelation - {test_house}')
    # plt.xlabel('Lag')
    # plt.show()
    # plt.savefig(f'D:/2020_ETRI/200804_monthly/autocorr_lag24_{test_house}.png')

    plt.plot(acf(data_col.fillna(0), nlags=24))
plt.legend(legends)
plt.ylim([0, 1.1])
plt.title('Autocorrelation')
plt.xticks(ticks=[i for i in range(0, 25, 6)])
plt.xlabel('Lags')

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data_col.fillna(0), lags=24)