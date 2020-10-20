"""
Anomaly detection
 and fwd-bwd joint imputation

2020. 10. 20. Tue.
Soyeong Park
"""
from funcs_ETRI import *
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


##############################
# 0. parameter setting
dir_data = 'D:/2020_ETRI/data'    # SG_data, label data, calendar 디렉토리를 포함하고 있는 경로
location = '서울'                 # 광주, 나주, 대전, 서울, 인천, label
test_apt = '1120011200'
test_house = 'fa7caf27'

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
df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
df['nan'] = chk_nan_bfaf(data_col)

##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. accumulation detection
idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
sample_list = list()
for i in range(len(idx_list)):
    idx_target = idx_list[i]
    sample, _, _ = nearest_neighbor(data_col, df['nan'].copy(), idx_target, calendar)
    sample_list.append(sample)
detect_sample = pd.DataFrame(sample_list)

# 3-2. z-score
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
z_score = (cand['injected'].values - detect_sample.mean(axis=1)) / detect_sample.std(axis=1)
df['z_score'] = pd.Series(z_score.values, index=df.index[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]])

detection_result = pd.DataFrame([], columns=['thld', 'MAE', 'MAE_no'])
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
        data_inj_temp[idx:idx+4] = df['injected'][idx:idx+4]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+4] = df['mask_inj'][idx:idx+4]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=f_fwd, f_len_bwd=f_bwd)
        imp_const[idx+1:idx+4] = fcst_bidirec1
        imp_no[idx+1:idx+4] = fcst_bidirec1

    # acc. imputation - idx_detected_acc
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx+4] = df['injected'][idx:idx+4]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+4] = 2
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=f_fwd, f_len_bwd=f_bwd, n_len=4)
        acc = data_inj_temp[idx]
        fcst_bidirec1 = fcst_bidirec1*(acc/sum(fcst_bidirec1))
        imp_const[idx:idx+4] = fcst_bidirec1
    # acc. imputation - no constraints
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx+4] = df['injected'][idx:idx+4]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+4] = df['mask_inj'][idx:idx+4]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=f_fwd, f_len_bwd=f_bwd, n_len=3)
        imp_no[idx+1:idx+4] = fcst_bidirec1

    # accuracy
    temp = pd.DataFrame({'values': data_col, 'imp_const': imp_const, 'imp_no': imp_no}).dropna()
    detection_result.loc[i] = [thld,
                               mean_absolute_error(temp['values'], temp['imp_const']),
                               mean_absolute_error(temp['values'], temp['imp_no'])]
    i += 1

plt.figure()
plt.plot(detection_result['thld'], detection_result['MAE'])
plt.plot(detection_result['thld'], detection_result['MAE_no'])
plt.legend(['w/ const.', 'w/o const.'], loc='lower right')
plt.xlabel('z-score threshold')
plt.ylabel('total MAE')
plt.title(f'{test_house}')
plt.tight_layout()
plt.show()

threshold = detection_result['thld'][detection_result['MAE']==detection_result['MAE'].min()].values[0]
print(f'** SELECTED THRESHOLD: {threshold}')


# 3-3. detection
idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
detected = np.zeros(len(data_col))
detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
detected[idx_detected_acc.astype('int')] = 4
df['mask_detected'] = detected

idx_injected = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
idx_real_nor = np.where(df['mask_inj'] == 3)[0]
idx_real_acc = np.where(df['mask_inj'] == 4)[0]

idx_detected = np.isin(idx_injected, idx_detected_acc)
idx_real = np.isin(idx_injected, idx_real_acc)
cm = confusion_matrix(idx_real, idx_detected)

plt.rcParams.update({'font.size': 14})
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', annot_kws={'size': 20}, square=True, cmap='Greys',  # 'gist_gray': reverse
            xticklabels=['normal', 'accumulation'], yticklabels=['normal', 'accumulation'])
plt.title(f'{location}, {test_apt}, {test_house}, threshold={threshold}', fontsize=14)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig(f'result/cm_{location}_{test_apt}_{test_house}.png')
plt.show()


##############################
# 4. imputation
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