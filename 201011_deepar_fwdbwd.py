"""
Accumulation detection with nearest neighbor,
imputation with DeepAR
 - input the holiday feature, fwdbwd average

2020. 10. 12. Mon.
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
# 4-0. load deepar samples + append original idx
sample_fwd = pd.concat([pd.read_csv('201011_deepar_fwd_1.csv', index_col=0),
                        pd.read_csv('201011_deepar_fwd_2.csv', index_col=0)])
sample_bwd = pd.concat([pd.read_csv('201011_deepar_bwd_1.csv', index_col=0),
                        pd.read_csv('201011_deepar_bwd_2.csv', index_col=0)])
idx_cand = np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]

idx_list_temp = np.empty([sum(df['mask_detected']==3)*nan_len + sum(df['mask_detected']==4)*(nan_len+1), 1])
i = 0
for idx in idx_cand:
    pred_len = nan_len if df['mask_detected'][idx]==3 else nan_len+1
    idx_list_temp[i:i+pred_len] = np.ones((pred_len,1))*idx
    i += pred_len
sample_fwd = np.append(idx_list_temp, sample_fwd.values, axis=1)
sample_bwd = np.append(idx_list_temp, sample_bwd.values, axis=1)


df['imp_const'] = df['injected'].copy()
df['imp_no-const'] = df['injected'].copy()

for idx in idx_cand:
    if df['mask_detected'][idx] == 3:       # detected normal
        df['imp_const'][idx+1:idx+4] =
        df['imp_no-const'][idx+1:idx+4] =
        pass
    else:           # detected acc.
        # 4-1. w/ const.
        # 4-2. w/o const.
        pass