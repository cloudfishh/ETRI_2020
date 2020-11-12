"""
Anomaly detection
 and fwd-bwd joint imputation
 ~ check sMAPE

2020. 11. 06. Fri.
Soyeong Park
"""
from funcs_ETRI import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import statistics


def mape(A, F):
    return 100 / len(A) * np.sum((np.abs(A - F))/A)


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def accuracy_by_cases(df):
    ##############################
    # 5-2. result -  imputation accuracy
    print('* IMPUTATION ACCURACY - FWD-BWD JOINT IMPUTATION *')
    # (1) true normal / predicted normal
    if np.where((df['mask_inj']==3)&(df['mask_detected']==3))[0].shape[0] != 0:
        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==3))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])

        print('1. true normal / predicted normal')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')


    # (2) true normal / predicted acc - 2 cases: w or w/o const.
    if np.where((df['mask_inj']==3)&(df['mask_detected']==4))[0].shape[0] != 0:
        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==4))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('2-1. true normal / predicted acc - with const.')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')

        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==4))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_no-const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('2-2. true normal / predicted acc - w/o const.')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')


    # (3) true acc / predicted normal
    if np.where((df['mask_inj']==4)&(df['mask_detected']==3))[0].shape[0] != 0:
        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==3))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('3. true acc / predicted normal')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')


    # (4) true acc / predicted acc - 2 cases: w or w/o const.
    if np.where((df['mask_inj']==4)&(df['mask_detected']==4))[0].shape[0] != 0:
        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==4))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('4-1. true acc / predicted acc - with const.')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')

        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==4))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_no-const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('4-2. true acc / predicted acc - w/o const.')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')


    # (5) total accuracy
    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
    result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
    result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
    result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
    print('5-1. total accuracy - w/ const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
    print(f'  MAPE {mape(result_true, result_impt)}')
    print(f' sMAPE {smape(result_true, result_impt)}\n')

    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_no-const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
    result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
    result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
    result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
    print('5-2. total accuracy - w/o const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
    print(f'  MAPE {mape(result_true, result_impt)}')
    print(f' sMAPE {smape(result_true, result_impt)}\n')


def accuracy_by_cases_li(df):
    ##############################
    # 5-2. result -  imputation accuracy
    print('* IMPUTATION ACCURACY - LINEAR INTERPOLATION *')
    # (1) true normal / predicted normal
    if np.where((df['mask_inj'] == 3) & (df['mask_detected'] == 3))[0].shape[0] != 0:
        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==3))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_linear_const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('1. true normal / predicted normal')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')


    # (2) true normal / predicted acc - 2 cases: w or w/o const.
    if np.where((df['mask_inj'] == 3) & (df['mask_detected'] == 4))[0].shape[0] != 0:
        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==4))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_linear_const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('2-1. true normal / predicted acc - with const.')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')

        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==4))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_linear_no-const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('2-2. true normal / predicted acc - w/o const.')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')


    # (3) true acc / predicted normal
    if np.where((df['mask_inj'] == 4) & (df['mask_detected'] == 3))[0].shape[0] != 0:
        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==3))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_linear_const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('3. true acc / predicted normal')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')


    # (4) true acc / predicted acc - 2 cases: w or w/o const.
    if np.where((df['mask_inj'] == 4) & (df['mask_detected'] == 4))[0].shape[0] != 0:
        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==4))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_linear_const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('4-1. true acc / predicted acc - with const.')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')

        result_true, result_impt = [], []
        for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==4))[0]:
            result_true.append(df['values'][idx:idx+4])
            result_impt.append(df['imp_linear_no-const'][idx:idx+4])
        result_true = np.array(result_true)
        result_impt = np.array(result_impt)
        result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
        result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
        result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
        result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
        print('4-2. true acc / predicted acc - w/o const.')
        print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
        print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
        print(f'  MAPE {mape(result_true, result_impt)}')
        print(f' sMAPE {smape(result_true, result_impt)}\n')


    # (5) total accuracy
    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_linear_const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
    result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
    result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
    result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
    print('5-1. total accuracy - w/ const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
    print(f'  MAPE {mape(result_true, result_impt)}')
    print(f' sMAPE {smape(result_true, result_impt)}\n')

    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_linear_no-const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    result_true = np.array(pd.DataFrame(result_true).replace(0, 0.001))
    result_impt = np.array(pd.DataFrame(result_impt).replace(0, 0.001))
    result_true = result_true.reshape([result_true.shape[0]*result_true.shape[1],])
    result_impt = result_impt.reshape([result_impt.shape[0]*result_impt.shape[1],])
    print('5-2. total accuracy - w/o const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
    print(f'  MAPE {mape(result_true, result_impt)}')
    print(f' sMAPE {smape(result_true, result_impt)}\n')


##############################
# 0. parameter setting
dir_data = 'D:/2020_ETRI/data'    # SG_data, label data, calendar 디렉토리를 포함하고 있는 경로
location = 'label'                 # 광주, 나주, 대전, 서울, 인천, label
test_apt = '1120011200'
test_house = '1dcb5feb'
f_fwd, f_bwd = 24, 24
nan_len = 3


##############################
# 1. load dataset
data_raw = load_apt(dir_data, location, test_apt)
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()
df['nan'] = chk_nan_bfaf(data_col)
df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
df['org_idx'] = np.arange(0, len(data_col))


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. accumulation detection
idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
nan_mask = df['nan'].copy()

# 3-2. z-score
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
method = 'nearest'
print(f'********** 1. detection : {method.upper()}')

detect_sample = pd.read_csv(f'201017_detection_{method}_{test_house}.csv', index_col=0)
z_score = (cand['injected'].values - detect_sample.mean(axis=1)) / detect_sample.std(axis=1)
df['z_score'] = pd.Series(z_score.values, index=df.index[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]])

# 3-3. threshold test ~ load the result
detection_result = pd.read_csv(f'201017_lossfunc_{method}_{test_house}.csv', index_col=0)

# plt.rcParams.update({'font.size': 14})
# plt.figure(figsize=(6,4), dpi=100)
# plt.plot(detection_result['thld'], detection_result['MAE'])
# plt.plot(detection_result['thld'], detection_result['MAE_no'])
# plt.axvline(x=detection_result['thld'][detection_result['MAE']==detection_result['MAE'].min()].values[0], color='r',
#             linewidth=1, linestyle='--')
# plt.legend(['w/ const.', 'w/o const.', 'threshold'], loc='lower right')
# plt.xlabel('z-score')
# plt.ylabel('total MAE')
# plt.xlim([0, 40])
# plt.ylim([0.005, 0.03])
# # plt.ylim([0.006, 0.0225])
# # plt.title(f'{test_house}')
# plt.tight_layout()

threshold = detection_result['thld'][detection_result['MAE']==detection_result['MAE'].min()].values[0]
print(f'** SELECTED THRESHOLD: {threshold}')

idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
detected = np.zeros(len(data_col))
detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
detected[idx_detected_acc.astype('int')] = 4
df['mask_detected'] = detected

# 3-4.  CONFUSION MATRIX ~ plot the detection result
# idx_injected = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
# idx_real_nor = np.where(df['mask_inj'] == 3)[0]
# idx_real_acc = np.where(df['mask_inj'] == 4)[0]

# idx_detected = np.isin(idx_injected, idx_detected_acc)
# idx_real = np.isin(idx_injected, idx_real_acc)
# cm = confusion_matrix(idx_real, idx_detected)
#
# group_names = ['TN', 'FP', 'FN', 'TP']
# group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
# cm_label = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
# cm_label = np.asarray(cm_label).reshape(2, 2)
#
# plt.rcParams.update({'font.size': 16})
# plt.figure(figsize=(4, 4), dpi=100)
# sns.heatmap(cm, annot=cm_label, fmt='', square=True, cmap='Greys', annot_kws={'size': 15}, # 'gist_gray': reverse
#             xticklabels=['normal', 'anomaly'], yticklabels=['normal', 'anomaly'], cbar=False)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.tight_layout()


##############################
# 4. imputation
print(f'***** 2. imputation : JOINT')

df['imp_const'] = df['injected'].copy()
df['imp_no-const'] = df['injected'].copy()

# 4-1. normal imputation - idx_detected_nor
for idx in idx_detected_nor:
    # idx 있는 곳만 injection 남겨서 imputation
    data_inj_temp = data_col.copy()
    data_inj_temp[idx:idx+4] = df['injected'][idx:idx+4]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+4] = df['mask_inj'][idx:idx+4]
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1)
    df['imp_const'][idx+1:idx+4] = fcst_bidirec1
    df['imp_no-const'][idx+1:idx+4] = fcst_bidirec1

# 4-2. acc. imputation - idx_detected_acc
for idx in idx_detected_acc:
    data_inj_temp = data_col.copy()
    # data_inj_temp[idx:idx+4] = data_inj[idx:idx+4]
    data_inj_temp[idx:idx+4] = df['injected'][idx:idx+4]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+4] = 2
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=4)
    acc = data_inj_temp[idx]
    fcst_bidirec1 = fcst_bidirec1*(acc/sum(fcst_bidirec1))
    df['imp_const'][idx:idx+4] = fcst_bidirec1

# 4-2-2. acc. imputation - no constraints
for idx in idx_detected_acc:
    data_inj_temp = data_col.copy()
    data_inj_temp[idx:idx+4] = df['injected'][idx:idx+4]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+4] = df['mask_inj'][idx:idx+4]
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=3)
    df['imp_no-const'][idx+1:idx+4] = fcst_bidirec1

accuracy_by_cases(df)


print(f'***** 2. imputation : LINEAR INTERPOLATION')
nan_len = 3

linear = df[{'values', 'injected', 'mask_detected'}].copy()
linear['imp_linear_const'] = linear['injected'].copy()
linear['imp_linear_no-const'] = linear['injected'].copy()

for idx in np.where((linear['mask_detected']==3)|(linear['mask_detected']==4))[0]:
    temp_col = linear['values'].copy()
    temp_col[idx:idx+nan_len+2] = linear['injected'][idx:idx+nan_len+2]

    linear['imp_linear_no-const'][idx:idx+nan_len+2] = temp_col[idx:idx+nan_len+2].interpolate(method='linear')
    if linear['mask_detected'][idx] == 3:
        p = 0
        while pd.isna(temp_col[idx-p]):
            p += 1
        q = 0
        while pd.isna(temp_col[idx+nan_len+2+q]):
            q += 1
        linear['imp_linear_const'][idx:idx+nan_len+2] = temp_col[idx:idx+nan_len+2].interpolate(method='linear')

    else:   # 4
        p = 0
        while pd.isna(temp_col[idx-1-p]):
            p += 1
        q = 0
        while pd.isna(temp_col[idx+nan_len+2+q]):
            q += 1
        s = temp_col[idx]
        temp_col[idx] = np.nan
        li_temp = temp_col[idx-1-p:idx+nan_len+2+q].interpolate(method='linear')
        linear['imp_linear_const'][idx-1-p:idx+nan_len+2+q] = li_temp*(s/sum(li_temp.values))

df['imp_linear_const'] = linear['imp_linear_const'].copy()
df['imp_linear_no-const'] = linear['imp_linear_no-const'].copy()

accuracy_by_cases_li(df)
