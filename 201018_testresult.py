"""
Accumulation detection with DeepAR
 and fwd-bwd joint imputation with DeepAR
- load results and analyse

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


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. accumulation detection
# 3-1. candidates probabilistic forecast
df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
df['org_idx'] = np.arange(0, len(data_col))

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

smlr_sample.to_csv(f'result/201017_detection_nearest_{test_house}.csv')
prob_sample = pd.read_csv(f'result/201017_detection_{test_house}.csv')


# 3-2. z-score
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
method = '201017_detection_nearest'

for method in list(['201017_detection', '201017_detection_nearest']):
    detect_sample = pd.read_csv(f'result/{method}_{test_house}.csv', index_col=0)
    z_score = (cand['injected'].values - detect_sample.mean(axis=1)) / detect_sample.std(axis=1)
    # df['z_score'] = np.nan
    # df['z_score'][np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]] = z_score.values
    df['z_score'] = pd.Series(z_score.values, index=df.index[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]])

    # 3-3. determine threshold for z-score
    # x-axis) z-score threshold [0, 10], y-axis) # of detected acc.
    # cand['z_score'] = z_score.values
    # detection = list()
    # for z in np.arange(0, 10, 0.1):
    #     # detected_acc = sum(candidate.values > z)
    #     # detection.append([z, sum(cand['z_score'] > z),
    #     detection.append([z,
    #                       sum((cand['mask_inj'] == 4) & (cand['z_score'] > z)),
    #                       sum((cand['mask_inj'] == 3) & (cand['z_score'] > z)),     # false positive (true nor, detect acc)
    #                       sum((cand['mask_inj'] == 4) & (cand['z_score'] < z))])    # false negative (true acc, detect nor)
    # detection = pd.DataFrame(detection, columns=['z-score', 'detected_acc', 'false_positive', 'false_negative'])
    #
    # plt.figure()
    # plt.plot(detection['z-score'], detection['false_positive'], color='tomato')
    # plt.plot(detection['z-score'], detection['false_negative'], color='seagreen')
    # plt.legend(['false positive', 'false negative'])
    # plt.xlabel('z-score threshold')
    # plt.ylabel('# of detection')
    # plt.title(f'{test_house}')
    # plt.tight_layout()

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
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1)
            imp_const[idx+1:idx+4] = fcst_bidirec1
            imp_no[idx+1:idx+4] = fcst_bidirec1

        # acc. imputation - idx_detected_acc
        for idx in idx_detected_acc:
            data_inj_temp = data_col.copy()
            data_inj_temp[idx:idx+4] = df['injected'][idx:idx+4]
            mask_inj_temp = np.isnan(data_col).astype('float')
            mask_inj_temp[idx:idx+4] = 2
            trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=4)
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
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=3)
            imp_no[idx+1:idx+4] = fcst_bidirec1

        # accuracy
        temp = pd.DataFrame({'values': data_col, 'imp_const': imp_const, 'imp_no': imp_no}).dropna()
        detection_result.loc[i] = [thld,
                                   mean_absolute_error(temp['values'], temp['imp_const']),
                                   mean_absolute_error(temp['values'], temp['imp_no'])]
        i += 1
    detection_result.to_csv(f'result/201017_{method[7:]}_lossfunc_{test_house}.csv')

    plt.figure()
    plt.plot(detection_result['thld'], detection_result['MAE'])
    plt.plot(detection_result['thld'], detection_result['MAE_no'])
    plt.legend(['w/ const.', 'w/o const.'], loc='lower right')
    plt.title(f'{method}')
    plt.xlabel('z-score threshold')
    plt.ylabel('total MAE')
    plt.ylim([0.006, 0.0225])
    plt.title(f'{test_house}')
    plt.tight_layout()
    plt.savefig(f'result/201017_{method[7:]}_lossfunc_{test_house}.png')

    # threshold = 7.5     # DEEPAR
    # threshold = 3.4   # NEAREST
    threshold = detection_result['MAE'].min()
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

sample_fwd = pd.concat([pd.read_csv(f'result/201017_deepar_{test_house}_fwd_1.csv', index_col=0),
                        pd.read_csv(f'result/201017_deepar_{test_house}_fwd_2.csv', index_col=0)])
sample_bwd = pd.concat([pd.read_csv(f'result/201017_deepar_{test_house}_bwd_1.csv', index_col=0),
                        pd.read_csv(f'result/201017_deepar_{test_house}_bwd_2.csv', index_col=0)])


sample_fwd_np, sample_bwd_np = np.array(sample_fwd[0]), np.array(sample_bwd[0])
for s in range(len(sample_fwd)-1):
    ss = s + 1
    sample_fwd_np = np.concatenate((sample_fwd_np, sample_fwd[ss]))
    sample_bwd_np = np.concatenate((sample_bwd_np, sample_bwd[ss]))

pd.DataFrame(sample_fwd_np).to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/201017_impt_fwd.csv')
pd.DataFrame(sample_bwd_np).to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/{test_house}/201017_impt_bwd.csv')

