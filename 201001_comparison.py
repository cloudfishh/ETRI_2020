"""
Comparison between probabilistic forecast & nearest neighbor

2020. 10. 01. Thu.
Soyeong Park
"""
##############################
from funcs import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


##############################
'''
detection 할 때 기준을 세우는 방법은 2가지.
1. nearest neighbor 했을 때처럼 sigma 기준으로 자르기.
2. prob forecast 처럼 z-score 기준으로 자르기
원론적으로 둘이 같은 방식임. 그냥 계산 어케 하냐 차이죠.
z-score threshold로 통일하도록 합시다. 
nearest neighbor method를 z-score 구하는 방식으로 만드는거죠.
'''

##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 3
sigma = 4
# imputation_acc = True


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()
df['nan'] = chk_nan_bfaf(data_col)


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)

inj_mask = df['mask_inj'].copy()


##############################
# 3. get the sample with nearest neighbor method
# idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
# nan_mask = df['nan'].copy()
#
# sample_list, mean_list, std_list = list(), list(), list()
# for i in range(len(idx_list)):
#     idx_target = idx_list[i]
#     sample, m, s = nearest_neighbor(data_col, df['nan'].copy(), idx_target, calendar)
#     sample_list.append(sample)
#     mean_list.append(m)
#     std_list.append(s)
# smlr_sample = pd.DataFrame(sample_list)
# smlr_sample.to_csv('201005_smlrdy-result.csv')


# 3-2. z-score
prob_sample = pd.read_csv('200928_deepar-result.csv', index_col=0)
smlr_sample = pd.read_csv('201005_smlrdy-result.csv', index_col=0)
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
z_score = (cand['injected'].values-smlr_sample.mean(axis=1))/smlr_sample.std(axis=1)
cand['z_score'] = z_score.values

df_z = df.copy()
df_z['z_score'] = np.nan
df_z['z_score'][(df['mask_inj'] == 3) | (df['mask_inj'] == 4)] = z_score.values

# 3-3. determine threshold for z-score
# x-axis) z-score threshold [0, 10], y-axis) # of detected acc.
detection = list()
for z in np.arange(0, 10, 0.1):
    # detected_acc = sum(candidate.values > z)
    # detection.append([z, sum(cand['z_score'] > z),
    detection.append([z,
                      sum((cand['mask_inj'] == 4) & (cand['z_score'] > z)),
                      sum((cand['mask_inj'] == 3) & (cand['z_score'] > z)),     # false positive (true nor, detect acc)
                      sum((cand['mask_inj'] == 4) & (cand['z_score'] < z))])    # false negative (true acc, detect nor)
detection = pd.DataFrame(detection, columns=['z-score', 'detected_acc', 'false_positive', 'false_negative'])

# plt.figure()
# plt.plot(detection['z-score'], detection['detected_acc'])
# plt.xlabel('z-score threshold')
# plt.ylabel('# of detected acc.')
# plt.tight_layout()
#
# plt.figure()
# plt.plot(detection['z-score'], detection['false_positive'], color='tomato')
# plt.plot(detection['z-score'], detection['false_negative'], color='seagreen')
# plt.legend(['false positive', 'false negative'])
# plt.xlabel('z-score threshold')
# plt.ylabel('# of detection')
# plt.tight_layout()


threshold = 3.4
idx_detected_nor = np.where(((df_z['mask_inj'] == 3) | (df_z['mask_inj'] == 4)) & (df_z['z_score'] < threshold))[0]
idx_detected_acc = np.where(((df_z['mask_inj'] == 3) | (df_z['mask_inj'] == 4)) & (df_z['z_score'] > threshold))[0]
detected = np.zeros(len(data_col))
detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
detected[idx_detected_acc.astype('int')] = 4
df['mask_detected'] = detected


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


##############################
# 5-0. result - prob forecast accuracy
y_pred_prob = prob_sample.mean(axis=1)
y_true = cand['values']
print('Probabilistic forecast mean accuracy')
print(f'RMSE: {mean_squared_error(y_true, y_pred_prob)**(1/2)}')
print(f' MSE: {mean_squared_error(y_true, y_pred_prob)}')
print(f' MAE: {mean_absolute_error(y_true, y_pred_prob)}')
print(f'MAPE: {mape(y_true, y_pred_prob)}\n')

y_pred_smlr = smlr_sample.mean(axis=1)
print('Nearest neighbor accuracy')
print(f'RMSE: {mean_squared_error(y_true, y_pred_smlr)**(1/2)}')
print(f' MSE: {mean_squared_error(y_true, y_pred_smlr)}')
print(f' MAE: {mean_absolute_error(y_true, y_pred_smlr)}')
print(f'MAPE: {mape(y_true, y_pred_prob)}\n')


##############################
# 5-1. result - detection confusion matrix
idx_injected = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
idx_real_nor = np.where(df['mask_inj'] == 3)[0]
idx_real_acc = np.where(df['mask_inj'] == 4)[0]

idx_detected = np.isin(idx_injected, idx_detected_acc)
idx_real = np.isin(idx_injected, idx_real_acc)
cm = confusion_matrix(idx_real, idx_detected)

plt.rcParams.update({'font.size': 14})
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', annot_kws={'size': 20}, square=True, cmap='Greys',     # 'gist_gray': reverse
            xticklabels=['normal', 'accumulation'], yticklabels=['normal', 'accumulation'])
# plt.title(f'{test_house}, nan_length=3, {sigma}'+r'$\sigma$', fontsize=14)
plt.title(f'{test_house}, nan_length=3, threshold={threshold}', fontsize=14)
plt.xlabel('Predicted label')
plt.ylabel('True label')


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


##############################
# 5-3. compare the sample distribution
# for i in range(len(idx_list)):
prob_stdz = list()
smlr_stdz = list()
for i in range(317):
    prob_stdz.append((prob_sample.iloc[i, :]-prob_sample.iloc[i, :].mean())/prob_sample.iloc[i, :].std())
    smlr_stdz.append((smlr_sample.iloc[i, :]-smlr_sample.iloc[i, :].mean())/smlr_sample.iloc[i, :].std())
prob_stdz = pd.DataFrame(prob_stdz)
smlr_stdz = pd.DataFrame(smlr_stdz)

plt.figure()
for i in range(50, 150):
    sns.distplot(prob_stdz.iloc[i, :])
plt.xlabel('')
plt.xlim([-5, 5])
plt.ylim([0, 1.7])

plt.figure()
for i in range(50, 150):
    sns.distplot(smlr_stdz.iloc[i, :])
plt.xlabel('')
plt.xlim([-5, 5])
plt.ylim([0, 1.7])


plt.figure()
sns.distplot((prob_sample.iloc[0, :]-prob_sample.iloc[0, :].mean())/prob_sample.iloc[0, :].std())
sns.distplot(prob_sample.iloc[0, :])

plt.figure()
sns.distplot((smlr_sample.iloc[0, :]-smlr_sample.iloc[0, :].mean())/smlr_sample.iloc[0, :].std())
sns.distplot(smlr_sample.iloc[0, :])