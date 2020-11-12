"""
Accumulation detection with probabilistic forecast
 - separate sequences (work/non-work)

2020. 10. 05. Mon.
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
sigma = 4


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
df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
a = df[df['holiday'] == 1]   # non-work
df = df[df['holiday'] == 0]        # work
# df = df[df['holiday'] == 1]   # non-work


##############################
# 3. accumulation detection
# 3-1. candidates probabilistic forecast
# input으로 test point 이전 23 points, output으로 test point 1 point
# training set으로는? 이전 한 달...? 너무 긴가 이전 2주로 일단.

idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
sample_list = list()

len_input = 23
len_train = 24*7*2

total_time = time.time()
for i in range(len(idx_list)):
    idx_target = idx_list[i]
    if idx_target > len_input + len_train + 1:
        idx_trn, idx_tst = idx_target-len_input-len_train, idx_target-len_input
        time_trn, time_tst = pd.Timestamp(df.index[idx_trn], freq='1H'), pd.Timestamp(df.index[idx_tst], freq='1H')
        trn = ListDataset([{'start': time_trn, 'target': df['injected'][idx_trn:idx_trn+len_train]}], freq='1H')
        tst = ListDataset([{'start': time_tst, 'target': df['injected'][idx_tst:idx_tst+len_input+1]}], freq='1H')
    else:
        # 데이터는 리버스로 넣고, timestamp는 정상적으로 넣고.
        idx_trn, idx_tst = idx_target+len_input+len_train, idx_target+len_input
        time_trn = pd.Timestamp(df.index[idx_target], freq='1H') - pd.Timedelta(value=len_input+len_train, unit='hours')
        time_tst = pd.Timestamp(df.index[idx_target], freq='1H') - pd.Timedelta(value=len_input, unit='hours')
        # trn = ListDataset([{'start': time_trn, 'target': df['injected'][idx_trn:idx_trn+len_train]}], freq='1H')
        # tst = ListDataset([{'start': time_tst, 'target': df['injected'][idx_tst:idx_tst+len_input+1]}], freq='1H')
        trn = ListDataset([{'start': time_trn,
                            'target': df['injected'][idx_target+len_input+1:idx_target+len_input+len_train+1][::-1]}], freq='1H')
        tst = ListDataset([{'start': time_tst,
                            'target': df['injected'][idx_target:idx_target+len_input+1][::-1]}], freq='1H')

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
    pd.DataFrame(forecasts[0].samples.reshape(1000,)).to_csv(f'201005_separate_idx{idx_target}.csv')

print(f'***** Total elapsed time {time.time()-total_time} secs')
pd.DataFrame(sample_list).to_csv(f'201005_separate_holiday{int(df["holiday"][0])}.csv')
