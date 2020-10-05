from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from funcs import *

result = pd.read_csv('D:/PycharmProjects/ETRI_2020/AR_imputation_result_another.csv', header=0, index_col=0)
x_labels = [f'({f_fwd}, {f_bwd})' for f_fwd, f_bwd in zip(result['f_fwd'], result['f_bwd'])]


##############################
fig, ax1 = plt.subplots(figsize=(15, 5))
ax2 = ax1.twinx()
data_y10 = ax1.plot(result['avr_mse'], linewidth=0.5, color='blue', marker='x', markersize=4)
data_y11 = ax1.plot(result['avr_rmse'], linewidth=0.5, color='green', marker='o', markersize=4)
data_y2 = ax2.plot(result['avr_smape'], linewidth=0.5, color='orange', marker='*', markersize=4)

ax1.set_xlabel('filter length (forward, backward))')
ax1.set_xticks(ticks=[i+24 for i in range(0, 24*24, 24)])
ax1.set_xticklabels(labels=[x_labels[l+23] for l in range(0, 24*24, 24)], rotation=45)
ax1.set_ylabel('MSE, RMSE')
ax1.set_ylim([-0.1, 1])
ax2.set_ylabel('sMAPE')

plt.grid('both', alpha=0.3)
plt.legend(data_y10+data_y11+data_y2, ['MSE', 'RMSE', 'sMAPE'])
plt.tight_layout()
plt.show()


##############################
for j in range(0, 24, 6):
    plt.figure(figsize=(7.5, 5))
    plt.plot(result['avr_smape'].values.reshape((24, -1))[:, j:j+6], 'x-')
    # plt.ylim([-0.1, 1])
    plt.legend([i for i in range(j, j+6)])


sort_mse = result.copy().sort_values(by='avr_mse').drop(columns=['avr_rmse', 'avr_smape'])
sort_rmse = result.copy().sort_values(by='avr_rmse').drop(columns=['avr_mse', 'avr_smape'])
sort_smape = result.copy().sort_values(by='avr_smape').drop(columns=['avr_mse', 'avr_rmse'])

sort_mse.to_csv('flen_sort_mse-another.csv')
sort_rmse.to_csv('flen_sort_rmse-another.csv')
sort_smape.to_csv('flen_sort_smape-another.csv')

# (f_fwd, f_bwd) = (24, 24) is the best


##################################################
# set variables
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
# test_house = '1dcb5feb'
# test_house = '68181c16'
test_house = '0098d3ee'

f_fwd, f_bwd = 24, 24
nan_len = 3

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
    fcst_bi, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=24, f_len_bwd=24)
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
