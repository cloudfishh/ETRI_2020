"""
Accumulation detection with nearest neighbor,
imputation with DeepAR
+) separate sequences by W/N days

2020. 10. 10. Sat.
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
idx_cand = np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]

len_unit = 24   # context_length + prediction length
len_train = 24*7*4

total_time = time.time()
sample_fwd, sample_bwd = list(), list()
for idx in idx_cand:
    if df['mask_detected'][idx] == 3:
        pred_len = nan_len
    else:
        pred_len = nan_len+1

    trn_fwd, tst_fwd, trn_bwd, tst_bwd = bidirec_dataset_deepar(df, idx, len_unit, len_train)

    # forward
    start_time = time.time()
    print(f'*** {idx} index forward forecast start')
    estimator = model_deepar(len_unit, df['mask_detected'][idx], epochs=10, feature=False)
    predictor = estimator.train(trn_fwd)
    forecast_it, _ = make_evaluation_predictions(
        dataset=tst_fwd,  # test dataset
        predictor=predictor,  # predictor
        num_samples=1000  # number of sample paths we want for evaluation
    )
    forecast_fwd = list(forecast_it)
    sample_fwd.append(np.append(np.ones((pred_len, 1))*df['original_idx'][idx],
                                forecast_fwd[0].samples.transpose(), axis=1))
    print(f'*** {idx} index forward forecast end - elapsed time {time.time()-start_time}')

    # backward
    print(f'*** {idx} index backward forecast start')
    estimator = model_deepar(len_unit, df['mask_detected'][idx])
    predictor = estimator.train(trn_bwd)
    forecast_it, _ = make_evaluation_predictions(
        dataset=tst_bwd,  # test dataset
        predictor=predictor,  # predictor
        num_samples=1000,  # number of sample paths we want for evaluation
    )
    forecast_bwd = list(forecast_it)
    sample_bwd.append(np.append(np.ones((pred_len, 1))*df['original_idx'][idx],
                                forecast_bwd[0].samples.transpose()[::-1], axis=1))    # backward는 거꾸로 다시 뒤집어줘야되죠

    print(f'*** {idx} index backward forecast end - elapsed time {time.time()-start_time}')
    pd.DataFrame(forecast_fwd[0].samples.transpose()).to_csv(f'201011_deear_separated_{df["original"][idx]}_fwd.csv')
    pd.DataFrame(forecast_bwd[0].samples.transpose()[::-1]).to_csv(f'201011_deear_separated_{df["original"][idx]}_bwd.csv')
print(f'***** COMPLETED - total elasped time {time.time() - total_time}')


sample_fwd_np, sample_bwd_np = np.array(sample_fwd[0]), np.array(sample_bwd[0])
for s in range(len(sample_fwd)-1):
    ss = s + 1
    sample_fwd_np = np.concatenate((sample_fwd_np, sample_fwd[ss]))
    sample_bwd_np = np.concatenate((sample_bwd_np, sample_bwd[ss]))

pd.DataFrame(sample_fwd_np).to_csv('201011_deepar_separated_fwd.csv')
pd.DataFrame(sample_bwd_np).to_csv('201011_deepar_separated_bwd.csv')
