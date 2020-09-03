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
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
# test_house = '1dcb5feb'
# test_house = '68181c16'
test_house = '0098d3ee'

# f_fwd, f_bwd = 24, 24
n_len = 3

##################################################
# load data
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
nan_mask = chk_nan_bfaf(data_col)

data_inj, mask_inj = inject_nan_imputation(data_col, nan_mask, n_len=3)
data_inj = data_inj[:, 48:data_inj.shape[1]-48]
mask_inj = mask_inj[:, 48:mask_inj.shape[1]-48]


##################################################
# injected nan 위치를 linear prediction
result = np.empty([24*24, 5])

list_fcst_temp = []
list_true_temp = []
for c in range(data_inj.shape[1]):
    col = data_inj[:, c]
    mask = mask_inj[:, c]
    trn_x, trn_y, tst_x = make_bidirectional_input(col, mask)
    fcst, pred = linear_prediction(trn_x, trn_y, tst_x, f_fwd, f_bwd)
    list_fcst_temp.append(fcst)
    list_true_temp.append(data_col[mask == 2].values)

list_fcst = np.array(list_fcst_temp).transpose()
list_true = np.array(list_true_temp).transpose()

list_mse = []
list_rmse = []
list_smape = []
for i in range(list_true.shape[1]):
    temp = mean_squared_error(list_true[:, i], list_fcst[:, i])
    list_mse.append(temp)
    list_rmse.append(temp**(1/2))
    list_smape.append(smape(list_true[:, i], list_fcst[:, i]))

nd_mse = np.array(list_mse)
nd_rmse = np.array(list_rmse)
nd_smape = np.array(list_smape)

# idx = (f_fwd-1)*24 + (f_bwd-1)
# result[idx, :] = [f_fwd, f_bwd, nd_mse, nd_rmse, nd_smape]


##################################################
# 결과 좋은 곳들 뽑아서 forecast값 plotting
test_idx = 7200     # 1147, 3222, 4771, 7326, 14363
test_col, test_mask = data_inj[:, test_idx], mask_inj[:, test_idx]
trn_x, trn_y, tst_x = make_bidirectional_input(test_col, test_mask)
fcst_bidirec, _ = linear_prediction(trn_x, trn_y, tst_x, f_fwd, f_bwd)
fcst_forward, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=48, f_len_bwd=0)
fcst_backward, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=0, f_len_bwd=48)

idx_inj = np.where(test_mask == 2)[0]
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(7.5, 5))
plt.plot(data_col.values[idx_inj[0]-48:idx_inj[-1]+24], '.-')
plt.plot(range(48, 48+n_len), fcst_forward, 'x-', linewidth=1, alpha=0.7)
plt.plot(range(48, 48+n_len), fcst_backward, '*-', linewidth=1, alpha=0.7)
plt.plot(range(48, 48+n_len), fcst_bidirec, 'o-', linewidth=1)
plt.legend(['observation', 'forward AR', 'backward AR', 'bidirectional AR'])
plt.xticks(ticks=[i for i in range(0, 48+24+n_len, 12)],
           labels=[data_col.index[i][2:16] for i in range(idx_inj[0]-48, idx_inj[-1]+24, 12)],
           rotation=30)
plt.xlim([24, 72])
plt.ylim([-0.1, 2])
# plt.ylim([-0.025, 0.2])
plt.grid(alpha=0.3)
plt.tight_layout()


##################################################
# 모든 경우에 대해 forward, backward, bidirectional 시행, 정확도
list_fcst_bi, list_fcst_fw, list_fcst_bw, list_true = [], [], [], []
list_mse_bi, list_rmse_bi, list_smape_bi, list_mae_bi = [], [], [], []
list_mse_fw, list_rmse_fw, list_smape_fw, list_mae_fw = [], [], [], []
list_mse_bw, list_rmse_bw, list_smape_bw, list_mae_bw = [], [], [], []

list_mape_bi, list_mape_fw, list_mape_bw = [], [], []

for c in range(data_inj.shape[1]):
    col = data_inj[:, c]
    mask = mask_inj[:, c]
    trn_x, trn_y, tst_x = make_bidirectional_input(col, mask)
    fcst_bi, _ = linear_prediction(trn_x, trn_y, tst_x, f_fwd, f_bwd)
    fcst_fw, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=48, f_len_bwd=0)
    fcst_bw, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=0, f_len_bwd=48)
    true = data_col[mask == 2].values

    list_fcst_bi.append(fcst_bi)
    list_fcst_fw.append(fcst_fw)
    list_fcst_bw.append(fcst_bw)
    list_true.append(true)

    list_mse_bi.append(mean_squared_error(true, fcst_bi))
    list_rmse_bi.append(mean_squared_error(true, fcst_bi)**(1/2))
    list_smape_bi.append(smape(true, fcst_bi))
    list_mae_bi.append(mean_absolute_error(true, fcst_bi))
    list_mape_bi.append(mape(true, fcst_bi))

    list_mse_fw.append(mean_squared_error(true, fcst_fw))
    list_rmse_fw.append(mean_squared_error(true, fcst_fw)**(1/2))
    list_smape_fw.append(smape(true, fcst_fw))
    list_mae_fw.append(mean_absolute_error(true, fcst_fw))
    list_mape_fw.append(mape(true, fcst_fw))

    list_mse_bw.append(mean_squared_error(true, fcst_bw))
    list_rmse_bw.append(mean_squared_error(true, fcst_bw)**(1/2))
    list_smape_bw.append(smape(true, fcst_bw))
    list_mae_bw.append(mean_absolute_error(true, fcst_bw))
    list_mape_bw.append(mape(true, fcst_bw))

nd_fcst_bi = np.array(list_fcst_bi).transpose()
nd_fcst_fw = np.array(list_fcst_fw).transpose()
nd_fcst_bw = np.array(list_fcst_bw).transpose()
nd_true = np.array(list_true).transpose()
nd_mse = np.array([list_mse_bi, list_mse_fw, list_mse_bw]).transpose()
nd_rmse = np.array([list_rmse_bi, list_rmse_fw, list_rmse_bw]).transpose()
nd_smape = np.array([list_smape_bi, list_smape_fw, list_smape_bw]).transpose()
nd_mae = np.array([list_mae_bi, list_mae_fw, list_mae_bw]).transpose()
nd_mape = np.array([list_mape_bi, list_mape_fw, list_mape_bw]).transpose()


##################################################
# 정확도 계산 및 분석
df_mse = pd.DataFrame(nd_mse, columns=['bi', 'fw', 'bw'])
df_rmse = pd.DataFrame(nd_rmse, columns=['bi', 'fw', 'bw'])
df_smape = pd.DataFrame(nd_smape, columns=['bi', 'fw', 'bw'])
df_mae = pd.DataFrame(nd_mae, columns=['bi', 'fw', 'bw'])


comp_mse, comp_rmse, comp_smape, comp_mae = [], [], [], []
for i in range(df_mse.shape[0]):
    if (df_mse.iloc[i, 0] < df_mse.iloc[i, 1]) & (df_mse.iloc[i, 0] < df_mse.iloc[i, 2]):
        comp_mse.append(0)
    else:
        comp_mse.append(1)

    if (df_rmse.iloc[i, 0] < df_rmse.iloc[i, 1]) & (df_rmse.iloc[i, 0] < df_rmse.iloc[i, 2]):
        comp_rmse.append(0)
    else:
        comp_rmse.append(1)

    if (df_smape.iloc[i, 0] < df_smape.iloc[i, 1]) & (df_smape.iloc[i, 0] < df_smape.iloc[i, 2]):
        comp_smape.append(0)
    else:
        comp_smape.append(1)

    if (df_mae.iloc[i, 0] < df_mae.iloc[i, 1]) & (df_mae.iloc[i, 0] < df_mae.iloc[i, 2]):
        comp_mae.append(0)
    else:
        comp_mae.append(1)

df_mse['compare'] = comp_mse
df_rmse['compare'] = comp_rmse
df_smape['compare'] = comp_smape
df_mae['compare'] = comp_mae

print(f'mse:{sum(comp_mse)}, rmse:{sum(comp_rmse)}, smape:{sum(comp_smape)}, mae:{sum(comp_mae)}')

l = df_mse.shape[0]
print(f'  avg_mse: (bi, {sum(df_mse["bi"])/l:.4f}), (fw, {sum(df_mse["fw"])/l:.4f}), (bw, {sum(df_mse["bw"])/l:.4f})')
print(f' avg_rmse: (bi, {sum(df_rmse["bi"])/l:.4f}), (fw, {sum(df_rmse["fw"])/l:.4f}), (bw, {sum(df_rmse["bw"])/l:.4f})')
print(f'avg_smape: '
      f'(bi, {sum(df_smape["bi"].fillna(0))/l:.4f}), '
      f'(fw, {sum(df_smape["fw"].fillna(0))/l:.4f}), '
      f'(bw, {sum(df_smape["bw"].fillna(0))/l:.4f})')
print(f'  avg_mae: (bi, {sum(df_mae["bi"])/l:.4f}), (fw, {sum(df_mae["fw"])/l:.4f}), (bw, {sum(df_mae["bw"])/l:.4f})')


##################################################
# 에러 가장 작은 case 찾아서 bidirectional, forward, backward plot
result = pd.DataFrame(df_mse['bi'].values, columns=['mse'])
result['rmse'], result['smape'], result['mae'] = df_rmse['bi'], df_smape['bi'], df_mae['bi']
result_mse = result.copy().sort_values(by='mse')
result_rmse = result.copy().sort_values(by='rmse')
result_smape = result.copy().sort_values(by='smape')
result_mae = result.copy().sort_values(by='mae')
