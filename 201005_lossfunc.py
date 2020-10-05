"""
optimize the z-score threshold by loss function (average MAE)

2020. 10. 05. Mon.
Soyeong Park
"""

##############################
from funcs import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

##############################
'''
z-score threshold에 따른 imputation accuracy를 구해서 최적의 threshold를 찾는다
1. 데이터 로드, injection
2. forecast 결과 로드 (샘플)
3. 각 candidate points에 대한 z-score 구하기
4. threshold에 대한 imputation accuracy 구해서 plot 
'''

##############################
# 1. 데이터 로드, injection, forecast 결과 로드, z-score 구하기
# data 로드
test_house = '68181c16'
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()
df['nan'] = chk_nan_bfaf(data_col)

# injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)

# forecast result sample 로드
method = 'deepar'
forecast = pd.read_csv(f'result_{method}.csv', index_col=0)

# z-score 구하기
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
z_score = (cand['injected'].values - forecast.mean(axis=1)) / forecast.std(axis=1)
cand['z_score'] = z_score.values
df['z_score'] = np.nan
df['z_score'][(df['mask_inj'] == 3) | (df['mask_inj'] == 4)] = z_score.values


##############################
# 2. threshold에 따라 detection, imputation, accuracy
result = pd.DataFrame([], columns=['thld', 'MAE', 'MAE_no'])
i = 0
for thld in np.arange(0, 40, 0.1):
    # detection
    idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < thld))[0]
    idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > thld))[0]
    detected = np.zeros(len(data_col))
    detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
    detected[idx_detected_acc.astype('int')] = 4

    # imputation
    imp_const = df['injected'].copy()
    imp_no = df['injected'].copy()
    # normal imputation - idx_detected_nor
    for idx in idx_detected_nor:
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx + 4] = df['injected'][idx:idx + 4]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx + 4] = df['mask_inj'][idx:idx + 4]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1)
        imp_const[idx + 1:idx + 4] = fcst_bidirec1
        imp_no[idx + 1:idx + 4] = fcst_bidirec1

    # acc. imputation - idx_detected_acc
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx + 4] = df['injected'][idx:idx + 4]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx + 4] = 2
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=4)
        acc = data_inj_temp[idx]
        fcst_bidirec1 = fcst_bidirec1 * (acc / sum(fcst_bidirec1))
        imp_const[idx:idx + 4] = fcst_bidirec1
    # acc. imputation - no constraints
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx + 4] = df['injected'][idx:idx + 4]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx + 4] = df['mask_inj'][idx:idx + 4]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=3)
        imp_no[idx + 1:idx + 4] = fcst_bidirec1

    # accuracy
    temp = pd.DataFrame({'values': data_col, 'imp_const': imp_const, 'imp_no': imp_no}).dropna()
    result.loc[i] = [thld,
                     mean_absolute_error(temp['values'], temp['imp_const']),
                     mean_absolute_error(temp['values'], temp['imp_no'])]

plt.figure()
plt.plot(result['thld'], result['MAE'])
plt.plot(result['thld'], result['MAE_no'])
plt.legend(['w/ const.', 'w/o const.'])
plt.title(f'{method}')
plt.tight_layout()
