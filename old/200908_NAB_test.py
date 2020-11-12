"""
다 합쳐야 됨
1. household 1개 데이터 불러오기
2. injection 하기
3. accumulation detection 하기 -> 결과 저장
4. detection 결과에 따라서 normal / acc 나눠서 imputation
4-1. normal인 경우는 그냥 imputation
4-2. acc인 경우는 imputation 후 constraint를 걸어서 합이 acc가 나오도록
5. 결과 출력  - normal part 정확도, acc part 정확도
"""

from funcs import *
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error


##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 3
sigma = 4
# imputation_acc = True


##############################
# 1. 데이터 불러오기
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. accumulation detection
idx_detected_acc, _ = check_accumulation_injected(df['injected'], df['mask_inj'], calendar, sigma=sigma)
idx_detected_acc = idx_detected_acc.astype('int')
detected = np.zeros(len(data_col))
detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
detected[idx_detected_acc.astype('int')] = 4
df['mask_detected'] = detected


##############################
# 4. separate cases
idx_candidates = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
idx_detected_nor = np.delete(idx_candidates, np.searchsorted(idx_candidates, idx_detected_acc))
df['imp_const'] = df['injected'].copy()
df['imp_no-const'] = df['injected'].copy()


##############################
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
plt.title(f'{test_house}, nan_length=3, {sigma}'+r'$\sigma$', fontsize=14)
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
