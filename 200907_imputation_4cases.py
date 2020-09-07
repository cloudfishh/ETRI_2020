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
# data_inj, mask_inj = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)
# df['injected'], df['mask_inj'] = data_inj, mask_inj
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. accumulation detection
# idx_detected_acc, _ = check_accumulation_injected(data_inj, mask_inj, calendar, sigma=sigma)
idx_detected_acc, _ = check_accumulation_injected(df['injected'], df['mask_inj'], calendar, sigma=sigma)
detected = np.zeros(len(data_col))
detected[idx_detected_acc.astype('int')] = 1
df['mask_detected'] = detected


##############################
# 4. seperate cases
idx_candidates = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
idx_detected_nor = np.delete(idx_candidates, np.searchsorted(idx_candidates, idx_detected_acc))


##############################
# 4-1. normal imputation - idx_detected_nor
result_nor = []
for idx in idx_detected_nor:
    data_inj_temp = data_col.copy()
    data_inj_temp[idx:idx+4] = data_inj[idx:idx+4]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+4] = mask_inj[idx:idx+4]
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1)
    result_nor.append(fcst_bidirec1)
result_nor = np.array(result_nor)


##############################
# 4-2. acc. imputation - idx_detected_acc
result_acc = []
for idx in idx_detected_acc:
    data_inj_temp = data_col.copy()
    data_inj_temp[idx:idx+4] = data_inj[idx:idx+4]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+4] = 2
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=4)
    acc = data_inj_temp[idx]
    fcst_bidirec1 = fcst_bidirec1*(acc/sum(fcst_bidirec1))
    result_acc.append(fcst_bidirec1)
result_acc = np.array(result_acc)

# 4-2-2. acc. imputation - no constraints
result_acc_no = []
for idx in idx_detected_acc:
    data_inj_temp = data_col.copy()
    data_inj_temp[idx:idx+4] = data_inj[idx:idx+4]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+4] = 2
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=4)
    result_acc_no.append(fcst_bidirec1)
result_acc_no = np.array(result_acc_no)


##############################
# 5-1. result - detection confusion matrix
idx_injected = np.where((mask_inj == 3) | (mask_inj == 4))[0]
idx_real_nor = np.where(mask_inj == 3)[0]
idx_real_acc = np.where(mask_inj == 4)[0]

idx_detected = np.isin(idx_injected, idx_detected_acc)
idx_real = np.isin(idx_injected, idx_real_acc)
cm = confusion_matrix(idx_real, idx_detected)

plt.rcParams.update({'font.size': 14})
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', annot_kws={'size': 20}, square=True, cmap='Greys',     # 'gist_gray': reverse
            xticklabels=['normal', 'accumulation'], yticklabels=['normal', 'accumulation'])
plt.title(f'{test_house}, nan_length=3, {sigma}$\sigma$', fontsize=14)
plt.xlabel('Predicted label')
plt.ylabel('True label')


##############################
# 5-2. result -  imputation accuracy
# (1) detected normal
true_nor = []
for idx in idx_detected_nor:
    true_nor.append(data_col[idx+1:idx+4])
true_nor = np.array(true_nor)
nor_mae, nor_mse = [], []
for i in range(true_nor.shape[0]):
    nor_mae.append(mean_absolute_error(true_nor[i, :], result_nor[i, :]))
    nor_mse.append(mean_squared_error(true_nor[i, :], result_nor[i, :]))
    # print(mean_absolute_error(true_nor[i, :], result_nor[i, :]))
    # print(f'{mean_squared_error(true_nor[i, :], result_nor[i, :])}\n')

result_nor_cal = np.delete(result_nor, np.where(true_nor == 0)[0], axis=0)
true_nor_cal = np.delete(true_nor, np.where(true_nor == 0)[0], axis=0)
print('** RESULT - detected normal **')
print(f' MAE {mean_absolute_error(true_nor_cal, result_nor_cal)}')
print(f'RMSE {mean_squared_error(true_nor_cal, result_nor_cal)**(1/2)}')
print(f'MAPE {mape(true_nor_cal, result_nor_cal)}\n')


# (2)-1. detected accumulation
true_acc = []
for idx in idx_detected_acc:
    true_acc.append(data_col[idx:idx+4])
true_acc = np.array(true_acc)
acc_mae, acc_mse = [], []
for i in range(true_acc.shape[0]):
    acc_mae.append(mean_absolute_error(true_acc[i, :], result_acc[i, :]))
    acc_mse.append(mean_squared_error(true_acc[i, :], result_acc[i, :]))
    # print(mean_absolute_error(true_acc[i, :], result_acc[i, :]))
    # print(f'{mean_squared_error(true_acc[i, :], result_acc[i, :])}\n')

result_acc_cal = np.delete(result_acc, np.where(true_acc == 0)[0], axis=0)
true_acc_cal = np.delete(true_acc, np.where(true_acc == 0)[0], axis=0)
print('** RESULT - detected acc. ~ apply constraints **')
print(f' MAE {mean_absolute_error(true_acc_cal, result_acc_cal)}')
print(f'RMSE {mean_squared_error(true_acc_cal, result_acc_cal)**(1/2)}')
print(f'MAPE {mape(true_acc_cal, result_acc_cal)}\n')


# (2)-2. detected accumulation - no constraints
acc_no_mae, acc_no_mse = [], []
for i in range(true_acc.shape[0]):
    acc_no_mae.append(mean_absolute_error(true_acc[i, :], result_acc_no[i, :]))
    acc_no_mse.append(mean_squared_error(true_acc[i, :], result_acc_no[i, :]))
    # print(mean_absolute_error(true_acc[i, :], result_acc_no[i, :]))
    # print(f'{mean_squared_error(true_acc[i, :], result_acc_no[i, :])}\n')

result_acc_no_cal = np.delete(result_acc_no, np.where(true_acc == 0)[0], axis=0)
print('** RESULT - detected acc. ~ no constraints **')
print(f' MAE {mean_absolute_error(true_acc_cal, result_acc_no_cal)}')
print(f'RMSE {mean_squared_error(true_acc_cal, result_acc_no_cal)**(1/2)}')
print(f'MAPE {mape(true_acc_cal, result_acc_no_cal)}\n')


##############################
# 6. result - plot imputation
# raw data / injected data / injected acc (도넛) / detected acc (도넛 안 채우기) / imputed data
# imputed도 acc에 대해 constraint를 준 경우와 안 준 경우가 있음.
plot_df = pd.DataFrame([], index=data_col.index,
                       columns=['obs', 'injected', 'imputed', 'imputed_no', 'injected_acc', 'detected_acc'])
plot_df['obs'] = data_col.values
plot_df['injected'] = data_inj.values

plt.figure(figsize=(15, 5))
plt.plot(data_col.values)
plt.axvspan(6,9)
plt.legend(['obs.', ])
plt.tight_layout()



p = np.array([[-2,1,2],[3,6,1],[1,16,-1]])
a = np.array([[-2,-4,2],[-2,1,2],[4,2,5]])

d = np.linalg.inv(p) * a * p


q = np.array([[2,-2,1],[1,3,6],[-1,1,16]])
q * np.array([[-5,0,0],[0,3,0],[0,0,6]]) * np.linalg.inv(q)

np.linalg.inv(np.array([[-2,-2,1/16],[-1,3,3/8],[1,1,1]])) * np.array([[-5,0,0],[0,3,0],[0,0,6]]) * np.array([[-2,-2,1/16],[-1,3,3/8],[1,1,1]])