"""
Accumulation detection with probabilistic forecast

2020. 09. 15. Tue.
Soyeong Park
"""
##############################
from funcs import *
import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator


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
idx_list = np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]

i = 99
idx, idx_trn = idx_list[i], idx_list[i]-24*7
time_trn, time_tst = pd.Timestamp(df.index[idx_trn], freq='1H'), pd.Timestamp(df.index[idx], freq='1H')
trn = ListDataset([{'start': time_trn, 'target': df['injected'][idx_trn:idx-24]}], freq='1H')
tst = ListDataset([{'start': time_tst, 'target': df['injected'][idx-24:idx]}], freq='1H')
estimator = DeepAREstimator(
    freq='1H',
    prediction_length=1,
    context_length=23,
    num_layers=2,
    num_cells=40,
    cell_type='lstm',
    dropout_rate=0.1,
    use_feat_dynamic_real=True,
    # embedding_dimension=20,
    scaling=True,
    lags_seq=None,
    time_features=None,
    trainer=Trainer(ctx=mx.cpu(0), epochs=5, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=200, ),
)

