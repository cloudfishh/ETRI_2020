"""
Accumulation detection with DeepAR
 and fwd-bwd joint imputation with DeepAR

2020. 10. 17. Sat.
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
test_house = '1dcb5feb'
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


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 2.5. sepearte cases


##############################
# 3. accumulation detection
# 3-1. candidates probabilistic forecast
# input으로 test point 이전 23 points, output으로 test point 1 point
# training set으로는? 이전 한 달...? 너무 긴가 이전 2주로 일단.
df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
df['org_idx'] = np.arange(0, len(data_col))
for holi in range(2):
    print(f'***** detection - holiday {holi} start *****')
    df_temp = df[df['holiday'] == holi]        # work
    # df = df[df['holiday'] == 1]   # non-work

    idx_list = np.where((df_temp['mask_inj'] == 3) | (df_temp['mask_inj'] == 4))[0]
    sample_list = list()

    len_input = 23
    len_train = 24*7*2

    total_time = time.time()
    for i in range(len(idx_list)):
        idx_target = idx_list[i]
        if idx_target > len_input + len_train + 1:
            idx_trn, idx_tst = idx_target-len_input-len_train, idx_target-len_input
            time_trn, time_tst = pd.Timestamp(df_temp.index[idx_trn], freq='1H'), pd.Timestamp(df_temp.index[idx_tst], freq='1H')
            trn = ListDataset([{'start': time_trn, 'target': df_temp['injected'][idx_trn:idx_trn+len_train]}], freq='1H')
            tst = ListDataset([{'start': time_tst, 'target': df_temp['injected'][idx_tst:idx_tst+len_input+1]}], freq='1H')
        else:
            # 데이터는 리버스로 넣고, timestamp는 정상적으로 넣고.
            idx_trn, idx_tst = idx_target+len_input+len_train, idx_target+len_input
            time_trn = pd.Timestamp(df_temp.index[idx_target], freq='1H') - pd.Timedelta(value=len_input+len_train, unit='hours')
            time_tst = pd.Timestamp(df_temp.index[idx_target], freq='1H') - pd.Timedelta(value=len_input, unit='hours')
            # trn = ListDataset([{'start': time_trn, 'target': df_temp['injected'][idx_trn:idx_trn+len_train]}], freq='1H')
            # tst = ListDataset([{'start': time_tst, 'target': df_temp['injected'][idx_tst:idx_tst+len_input+1]}], freq='1H')
            trn = ListDataset([{'start': time_trn,
                                'target': df_temp['injected'][idx_target+len_input+1:idx_target+len_input+len_train+1][::-1]}], freq='1H')
            tst = ListDataset([{'start': time_tst,
                                'target': df_temp['injected'][idx_target:idx_target+len_input+1][::-1]}], freq='1H')

        estimator = DeepAREstimator(
            freq='1H',
            prediction_length=1,
            context_length=len_input,
            num_layers=2,
            num_cells=40,
            cell_type='lstm',
            dropout_rate=0.1,
            # use_feat_dynamic_real=True,
            # embedding_dimension=20,
            scaling=True,
            lags_seq=None,
            time_features=None,
            trainer=Trainer(ctx=mx.cpu(0), epochs=10, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=200, ),
        )
        print(f'*** {i}th candidate training start ***')
        start_time = time.time()
        predictor = estimator.train(trn)
        print(f'*** {i}th candidate training end ***')
        print(f'*** elapsed time {time.time() - start_time} secs ***')
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=tst,  # test dataset
            predictor=predictor,  # predictor
            num_samples=1000,  # number of sample paths we want for evaluation
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        sample_list.append(forecasts[0].samples.reshape(1000,))
        # pd.DataFrame(forecasts[0].samples.reshape(1000,)).to_csv(f'201005_separate_idx{idx_target}.csv')
        print(f'***** detection - holiday {holi} end *****')

    print(f'***** Total detection elapsed time {time.time()-total_time} secs')

    pd.DataFrame(sample_list).to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/201017_detection_{test_house}_holiday{holi}.csv')


# 3-2. z-score
# detect_sample = pd.read_csv('result_deepar_separate_4weeks.csv', index_col=0)     # DEEPAR
# detect_sample = pd.DataFrame(sample_list)
df_holi0 = df[df['holiday'] == 0]  # work, holi0
df_holi1 = df[df['holiday'] == 1]  # non-work, holi1

idx_list_holi0 = df_holi0['org_idx'][(df_holi0['mask_inj'] == 3) | (df_holi0['mask_inj'] == 4)]
idx_list_holi1 = df_holi1['org_idx'][(df_holi1['mask_inj'] == 3) | (df_holi1['mask_inj'] == 4)]

holi0 = pd.read_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/201017_detection_{test_house}_holiday0.csv', index_col=0)
holi1 = pd.read_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/201017_detection_{test_house}_holiday1.csv', index_col=0)

a0, a1 = pd.DataFrame(), pd.DataFrame()
a0['idx'], a1['idx'] = idx_list_holi0, idx_list_holi1

b0 = pd.concat([a0.reset_index(drop=True), holi0], axis=1, ignore_index=True)
b1 = pd.concat([a1.reset_index(drop=True), holi1], axis=1, ignore_index=True)

c = pd.concat([b0, b1])
cc = c.sort_values(by=0)
idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]

cc.drop(columns=[0]).to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/201017_detection_{test_house}.csv')


##############################
# 4. imputation
idx_cand = np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]

len_unit = 24   # context_length + prediction length
len_train = 24*7*4

total_time = time.time()
sample_fwd, sample_bwd = list(), list()
print(f'***** imputation start *****')
for idx in idx_cand:
    trn_fwd, tst_fwd, trn_bwd, tst_bwd = bidirec_dataset_deepar_test(df, idx, len_unit, len_train)

    # forward
    start_time = time.time()
    print(f'*** {idx} index forward forecast start')
    estimator = model_deepar_test(len_unit, feature=True, epochs=10)
    predictor = estimator.train(trn_fwd)
    forecast_it, _ = make_evaluation_predictions(
        dataset=tst_fwd,  # test dataset
        predictor=predictor,  # predictor
        num_samples=1000  # number of sample paths we want for evaluation
    )
    forecast_fwd = list(forecast_it)
    sample_fwd.append(forecast_fwd[0].samples.transpose())
    print(f'*** {idx} index forward forecast end - elapsed time {time.time()-start_time}')

    # backward
    print(f'*** {idx} index backward forecast start')
    estimator = model_deepar(len_unit, feature=True, epochs=10)
    predictor = estimator.train(trn_bwd)
    forecast_it, _ = make_evaluation_predictions(
        dataset=tst_bwd,  # test dataset
        predictor=predictor,  # predictor
        num_samples=1000,  # number of sample paths we want for evaluation
    )
    forecast_bwd = list(forecast_it)
    sample_bwd.append(forecast_bwd[0].samples.transpose()[::-1])    # backward는 거꾸로 다시 뒤집어줘야되죠
    print(f'*** {idx} index backward forecast end - elapsed time {time.time()-start_time}')
    pd.DataFrame(forecast_fwd[0].samples.transpose()).to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/csv_temp/201017_impt_{idx}_fwd.csv')
    pd.DataFrame(forecast_bwd[0].samples.transpose()[::-1]).to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/csv_temp/201017_impt_{idx}_bwd.csv')

print(f'***** COMPLETED - total imputation elasped time {time.time() - total_time}')


sample_fwd_np, sample_bwd_np = np.array(sample_fwd[0]), np.array(sample_bwd[0])
for s in range(len(sample_fwd)-1):
    ss = s + 1
    sample_fwd_np = np.concatenate((sample_fwd_np, sample_fwd[ss]))
    sample_bwd_np = np.concatenate((sample_bwd_np, sample_bwd[ss]))

pd.DataFrame(sample_fwd_np).to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/201017_impt_fwd.csv')
pd.DataFrame(sample_bwd_np).to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/201017_impt_bwd.csv')

