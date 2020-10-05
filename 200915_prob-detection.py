"""
Accumulation detection with probabilistic forecast

2020. 09. 15. Tue.
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
'''
1. household 1개 데이터 불러오기
2. injection 하기
3. accumulation detection
    - probabilistic forecast로 판단하기?
    - candidate(before value)에 대해 prob. forecast를 하고 그 candidate가 위치한 interval에 따라 (z-score) 판단.
    - criteria? hmm?
    - 그럼 일단 다음주까진 candidate의 z-score 분석
4. imputation
5. result
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


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. accumulation detection
# 3-1. candidates probabilistic forecast
# input으로 test point 이전 23 points, output으로 test point 1 point
# training set으로는? 이전 한 달...? 너무 긴가 이전 2주로 일단.
#
# idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
# sample_list = list()
#
# len_input = 23+24*6
# len_train = 24*7*2
#
# total_time = time.time()
# for i in range(len(idx_list)):
#     idx_target = idx_list[i]
#     if idx_target > len_input + len_train + 1:
#         idx_trn, idx_tst = idx_target-len_input-len_train, idx_target-len_input
#         time_trn, time_tst = pd.Timestamp(df.index[idx_trn], freq='1H'), pd.Timestamp(df.index[idx_tst], freq='1H')
#         trn = ListDataset([{'start': time_trn, 'target': df['injected'][idx_trn:idx_trn+len_train]}], freq='1H')
#         tst = ListDataset([{'start': time_tst, 'target': df['injected'][idx_tst:idx_tst+len_input+1]}], freq='1H')
#     else:
#         # 데이터는 리버스로 넣고, timestamp는 정상적으로 넣고.
#         idx_trn, idx_tst = idx_target+len_input+len_train, idx_target+len_input
#         time_trn = pd.Timestamp(df.index[idx_target], freq='1H') - pd.Timedelta(value=len_input+len_train, unit='hours')
#         time_tst = pd.Timestamp(df.index[idx_target], freq='1H') - pd.Timedelta(value=len_input, unit='hours')
#         # trn = ListDataset([{'start': time_trn, 'target': df['injected'][idx_trn:idx_trn+len_train]}], freq='1H')
#         # tst = ListDataset([{'start': time_tst, 'target': df['injected'][idx_tst:idx_tst+len_input+1]}], freq='1H')
#         trn = ListDataset([{'start': time_trn,
#                             'target': df['injected'][idx_target+len_input+1:idx_target+len_input+len_train+1][::-1]}], freq='1H')
#         tst = ListDataset([{'start': time_tst,
#                             'target': df['injected'][idx_target:idx_target+len_input+1][::-1]}], freq='1H')
#
#     estimator = DeepAREstimator(
#         freq='1H',
#         prediction_length=1,
#         context_length=len_input,
#         num_layers=2,
#         num_cells=40,
#         cell_type='lstm',
#         dropout_rate=0.1,
#         # use_feat_dynamic_real=True,
#         # embedding_dimension=20,
#         scaling=True,
#         lags_seq=None,
#         time_features=None,
#         trainer=Trainer(ctx=mx.cpu(0), epochs=10, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=200, ),
#     )
#     print(f'*** {i}th candidate training start ***')
#     start_time = time.time()
#     predictor = estimator.train(trn)
#     print(f'*** {i}th candidate training end ***')
#     print(f'*** elapsed time {time.time() - start_time} secs ***')
#     forecast_it, ts_it = make_evaluation_predictions(
#         dataset=tst,  # test dataset
#         predictor=predictor,  # predictor
#         num_samples=1000,  # number of sample paths we want for evaluation
#     )
#     forecasts = list(forecast_it)
#     tss = list(ts_it)
#     sample_list.append(forecasts[0].samples.reshape(1000,))
#
# print(f'***** Total elapsed time {time.time()-total_time} secs')
# pd.DataFrame(sample_list).to_csv('200928_deepar-result.csv')


# 3-2. z-score
prob_sample = pd.read_csv('200928_deepar-result.csv', index_col=0)
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
z_score = (cand['injected'].values-prob_sample.mean(axis=1))/prob_sample.std(axis=1)
cand['z_score'] = z_score.values

df_z = df.copy()
df_z['z_score'] = np.nan
df_z['z_score'][(df['mask_inj'] == 3) | (df['mask_inj'] == 4)] = z_score.values

# 3-3. determine threshold for z-score
# x-axis) z-score threshold [0, 10], y-axis) # of detected acc.
detection = list()
for z in np.arange(0, 40, 0.1):
    # detected_acc = sum(candidate.values > z)
    # detection.append([z, sum(cand['z_score'] > z),
    detection.append([z,
                      sum((cand['mask_inj'] == 4) & (cand['z_score'] > z)),
                      sum((cand['mask_inj'] == 3) & (cand['z_score'] > z)),     # false positive (true nor, detect acc)
                      sum((cand['mask_inj'] == 4) & (cand['z_score'] < z))])    # false negative (true acc, detect nor)
detection = pd.DataFrame(detection, columns=['z-score', 'detected_acc', 'false_positive', 'false_negative'])

plt.figure()
plt.plot(detection['z-score'], detection['detected_acc'])
plt.xlabel('z-score threshold')
plt.ylabel('# of detected acc.')
plt.tight_layout()

plt.figure()
plt.plot(detection['z-score'], detection['false_positive'], color='tomato')
plt.plot(detection['z-score'], detection['false_negative'], color='seagreen')
plt.legend(['false positive', 'false negative'])
plt.xlabel('z-score threshold')
plt.ylabel('# of detection')
plt.tight_layout()

threshold = 19.6
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
y_pred = prob_sample.mean(axis=1)
y_true = cand['values']
print('Probabilistic forecast mean accuracy')
print(f'RMSE: {mean_squared_error(y_true, y_pred)**(1/2)}')
print(f' MSE: {mean_squared_error(y_true, y_pred)}')
print(f' MAE: {mean_absolute_error(y_true, y_pred)}')
print(f'MAPE: {mape(y_true, y_pred)}\n')

_, y_pred_m = check_accumulation_injected(df['injected'], df['mask_inj'], calendar, sigma=4)
print('Nearest neighbor accuracy')
print(f'RMSE: {mean_squared_error(y_true, y_pred_m)**(1/2)}')
print(f' MSE: {mean_squared_error(y_true, y_pred_m)}')
print(f' MAE: {mean_absolute_error(y_true, y_pred_m)}')
print(f'MAPE: {mape(y_true, y_pred)}\n')


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

