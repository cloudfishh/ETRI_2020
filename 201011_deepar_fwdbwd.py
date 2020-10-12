"""
Accumulation detection with nearest neighbor,
imputation with DeepAR
 - input the holiday feature, fwdbwd average

2020. 10. 12. Mon.
Soyeong Park
"""
##############################
from funcs import *
import time
import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 3


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()
df['nan'] = chk_nan_bfaf(data_col)
df['holiday'] = calendar.loc[pd.Timestamp(df.index[0]):pd.Timestamp(df.index[-1])]


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. get the sample with nearest neighbor method
idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
nan_mask = df['nan'].copy()

sample_list, mean_list, std_list = list(), list(), list()
for i in range(len(idx_list)):
    idx_target = idx_list[i]
    sample, m, s = nearest_neighbor(data_col, df['nan'].copy(), idx_target, calendar)
    sample_list.append(sample)
    mean_list.append(m)
    std_list.append(s)
smlr_sample = pd.DataFrame(sample_list)
# smlr_sample.to_csv('result_nearest.csv')


# 3-2. z-score
# smlr_sample = pd.read_csv('result_nearest.csv', index_col=0)
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
z_score = (cand['injected'].values-smlr_sample.mean(axis=1))/smlr_sample.std(axis=1)
cand['z_score'] = z_score.values

# df_z = df.copy()
df['z_score'] = np.nan
df['z_score'][(df['mask_inj'] == 3) | (df['mask_inj'] == 4)] = z_score.values

threshold = 3.4
idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
detected = np.zeros(len(data_col))
detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
detected[idx_detected_acc.astype('int')] = 4
df['mask_detected'] = detected


##############################
# 4. imputation
# 4-0. load deepar samples + append original idx
sample_fwd = pd.concat([pd.read_csv('201011_deepar_fwd_1.csv', index_col=0),
                        pd.read_csv('201011_deepar_fwd_2.csv', index_col=0)])
sample_bwd = pd.concat([pd.read_csv('201011_deepar_bwd_1.csv', index_col=0),
                        pd.read_csv('201011_deepar_bwd_2.csv', index_col=0)])
idx_cand = np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]

idx_list_temp = np.empty([sum(df['mask_detected']==3)*nan_len + sum(df['mask_detected']==4)*(nan_len+1), 1])
i = 0
for idx in idx_cand:
    pred_len = nan_len if df['mask_detected'][idx]==3 else nan_len+1
    idx_list_temp[i:i+pred_len] = np.ones((pred_len,1))*idx
    i += pred_len
sample_fwd = np.append(idx_list_temp, sample_fwd.values, axis=1)
sample_bwd = np.append(idx_list_temp, sample_bwd.values, axis=1)


df['imp_const'] = df['injected'].copy()
df['imp_no-const'] = df['injected'].copy()

i = 0
for idx in idx_cand:
    if df['mask_detected'][idx] == 3:       # detected normal
        pred_len = nan_len
        df['imp_const'][idx+1:idx+4] = (sample_fwd[i:i+pred_len, 1:].mean(axis=1) + sample_bwd[i:i+pred_len, 1:].mean(axis=1))/2
        df['imp_no-const'][idx+1:idx+4] = (sample_fwd[i:i+pred_len, 1:].mean(axis=1) + sample_bwd[i:i+pred_len, 1:].mean(axis=1))/2
        i += pred_len
    else:           # detected acc.
        # 4-1. w/ const.
        pred_len = nan_len + 1
        r = df['injected'][idx] / sum((sample_fwd[i:i+pred_len,1:].mean(axis=1) + sample_bwd[i:i+pred_len,1:].mean(axis=1)) / 2)
        df['imp_const'][idx:idx+4] = r*((sample_fwd[i:i+pred_len,1:].mean(axis=1) + sample_bwd[i:i+pred_len,1:].mean(axis=1)) / 2)
        # 4-2. w/o const.
        pred_len = nan_len
        df['imp_no-const'][idx+1:idx+4] = (sample_fwd[i:i+pred_len,1:].mean(axis=1) + sample_bwd[i:i+pred_len,1:].mean(axis=1)) / 2
        i += (nan_len+1)



##############################
# 5-0. result - prob forecast accuracy
# y_pred_prob = prob_sample.mean(axis=1)
# y_true = cand['values']
# print('Probabilistic forecast mean accuracy')
# print(f'RMSE: {mean_squared_error(y_true, y_pred_prob)**(1/2)}')
# print(f' MSE: {mean_squared_error(y_true, y_pred_prob)}')
# print(f' MAE: {mean_absolute_error(y_true, y_pred_prob)}')
# print(f'MAPE: {mape(y_true, y_pred_prob)}\n')
#
# y_pred_smlr = smlr_sample.mean(axis=1)
# print('Nearest neighbor accuracy')
# print(f'RMSE: {mean_squared_error(y_true, y_pred_smlr)**(1/2)}')
# print(f' MSE: {mean_squared_error(y_true, y_pred_smlr)}')
# print(f' MAE: {mean_absolute_error(y_true, y_pred_smlr)}')
# print(f'MAPE: {mape(y_true, y_pred_prob)}\n')


##############################
# 5-1. result - detection confusion matrix
# idx_injected = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
# idx_real_nor = np.where(df['mask_inj'] == 3)[0]
# idx_real_acc = np.where(df['mask_inj'] == 4)[0]
#
# idx_detected = np.isin(idx_injected, idx_detected_acc)
# idx_real = np.isin(idx_injected, idx_real_acc)
# cm = confusion_matrix(idx_real, idx_detected)
#
# plt.rcParams.update({'font.size': 14})
# plt.figure()
# sns.heatmap(cm, annot=True, fmt='d', annot_kws={'size': 20}, square=True, cmap='Greys',     # 'gist_gray': reverse
#             xticklabels=['normal', 'accumulation'], yticklabels=['normal', 'accumulation'])
# # plt.title(f'{test_house}, nan_length=3, {sigma}'+r'$\sigma$', fontsize=14)
# plt.title(f'{test_house}, nan_length=3, threshold={threshold}', fontsize=14)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')


##############################
# 5-2. result -  imputation accuracy
# (1) true normal / predicted normal
result_true, result_impt = [], []
for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==3))[0]:
    result_true.append(df['values'][idx:idx+4])
    result_impt.append(df['imp_const'][idx:idx+4])
result_true = np.array(result_true)
result_impt = np.array(result_impt)
print('1. true normal / predicted normal')
print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
print(f'  MAPE {mape(result_true, result_impt)}\n')


# (2) true normal / predicted acc - 2 cases: w or w/o const.
result_true, result_impt = [], []
for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==4))[0]:
    result_true.append(df['values'][idx:idx+4])
    result_impt.append(df['imp_const'][idx:idx+4])
result_true = np.array(result_true)
result_impt = np.array(result_impt)
print('2-1. true normal / predicted acc - with const.')
print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
print(f'  MAPE {mape(result_true, result_impt)}\n')

result_true, result_impt = [], []
for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==4))[0]:
    result_true.append(df['values'][idx:idx+4])
    result_impt.append(df['imp_no-const'][idx:idx+4])
result_true = np.array(result_true)
result_impt = np.array(result_impt)
print('2-2. true normal / predicted acc - w/o const.')
print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
print(f'  MAPE {mape(result_true, result_impt)}\n')


# (3) true acc / predicted normal
result_true, result_impt = [], []
for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==3))[0]:
    result_true.append(df['values'][idx:idx+4])
    result_impt.append(df['imp_const'][idx:idx+4])
result_true = np.array(result_true)
result_impt = np.array(result_impt)
print('3. true acc / predicted normal')
print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
print(f'  MAPE {mape(result_true, result_impt)}\n')


# (4) true acc / predicted acc - 2 cases: w or w/o const.
result_true, result_impt = [], []
for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==4))[0]:
    result_true.append(df['values'][idx:idx+4])
    result_impt.append(df['imp_const'][idx:idx+4])
result_true = np.array(result_true)
result_impt = np.array(result_impt)
print('4-1. true acc / predicted acc - with const.')
print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
print(f'  MAPE {mape(result_true, result_impt)}\n')

result_true, result_impt = [], []
for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==4))[0]:
    result_true.append(df['values'][idx:idx+4])
    result_impt.append(df['imp_no-const'][idx:idx+4])
result_true = np.array(result_true)
result_impt = np.array(result_impt)
print('4-2. true acc / predicted acc - w/o const.')
print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
print(f'  MAPE {mape(result_true, result_impt)}\n')
