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
import seaborn as sns
from sklearn.metrics import confusion_matrix


def accuracy_by_cases(df):
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


    # (5) total accuracy
    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    print('5-1. total accuracy - w/ const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
    print(f'  MAPE {mape(result_true, result_impt)}\n')

    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_no-const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    print('5-2. total accuracy - w/o const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')
    print(f'  MAPE {mape(result_true, result_impt)}\n')


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
df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
df['org_idx'] = np.arange(0, len(data_col))

idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
nan_mask = df['nan'].copy()

# save the nearest neighbor samples
sample_list, mean_list, std_list = list(), list(), list()
for i in range(len(idx_list)):
    idx_target = idx_list[i]
    sample, m, s = nearest_neighbor(data_col, df['nan'].copy(), idx_target, calendar)
    sample_list.append(sample)
    mean_list.append(m)
    std_list.append(s)
smlr_sample = pd.DataFrame(sample_list)
smlr_sample.to_csv(f'result/{test_house}/201017_detection_nearest_{test_house}.csv')

# rename the  probabilistic sample file
pd.read_csv(f'result/{test_house}/201017_detection_{test_house}.csv').to_csv((f'result/{test_house}/201017_detection_deepar_{test_house}.csv'))


# 3-2. z-score
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
method = '201017_detection_nearest'
# method = '201017_detection_deepar'

# find the optimal threshold
for method in list(['201017_detection_deepar', '201017_detection_nearest']):
    print(f'********** 1. detection : {method[17:].upper()}')

    detect_sample = pd.read_csv(f'result/{test_house}/{method}_{test_house}.csv', index_col=0)
    z_score = (cand['injected'].values - detect_sample.mean(axis=1)) / detect_sample.std(axis=1)
    # df['z_score'] = np.nan
    # df['z_score'][np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]] = z_score.values
    df['z_score'] = pd.Series(z_score.values, index=df.index[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]])

    # 3-3. threshold test
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
        print(f'* threshold test: {thld}')

    detection_result.to_csv(f'result/{test_house}/201017_lossfunc_{method[17:]}_{test_house}.csv')
    detection_result = pd.read_csv(f'result/{test_house}/201017_lossfunc_{method[17:]}_{test_house}.csv', index_col=0)

    plt.figure()
    plt.plot(detection_result['thld'], detection_result['MAE'])
    plt.plot(detection_result['thld'], detection_result['MAE_no'])
    plt.legend(['w/ const.', 'w/o const.'], loc='lower right')
    plt.title(f'{method}')
    plt.xlabel('z-score threshold')
    plt.ylabel('total MAE')
    # plt.ylim([0.006, 0.0225])
    plt.title(f'{test_house}')
    plt.tight_layout()
    plt.savefig(f'result/{test_house}/201017_lossfunc_{method[17:]}_{test_house}.png')

    # threshold = 7.5     # DEEPAR
    # threshold = 3.4   # NEAREST
    threshold = detection_result['thld'][detection_result['MAE']==detection_result['MAE'].min()].values[0]
    print(f'** SELECTED THRESHOLD: {threshold}')

    idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
    idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
    detected = np.zeros(len(data_col))
    detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
    detected[idx_detected_acc.astype('int')] = 4
    df['mask_detected'] = detected

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
    plt.title(f'{test_house}, {method[17:]}, nan_length=3, threshold={threshold}', fontsize=14)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f'result/{test_house}/201017_confusion_{method[17:]}_{test_house}.png')


    ##############################
    # 4. imputation
    print(f'***** 2. imputation : LINEAR')

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

    accuracy_by_cases(df)


    print(f'***** 2. imputation : DEEPAR')
    idx_cand = np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]

    sample_fwd = pd.read_csv(f'result/{test_house}/201017_impt_fwd.csv', index_col=0).values
    sample_bwd = pd.read_csv(f'result/{test_house}/201017_impt_bwd.csv', index_col=0).values

    w_nor = np.array([[1, 0.5, 0], [0, 0.5, 1]])
    w_acc = np.array([[1, 2/3, 1/3, 0], [0, 1/3, 2/3, 1]])

    # imputation
    df['imp_const'] = df['injected'].copy()
    df['imp_no-const'] = df['injected'].copy()

    i = 0
    for idx in idx_cand:
        if df['mask_detected'][idx] == 3:  # detected normal
            pred_len = nan_len
            df['imp_const'][idx+1:idx+4] = (sample_fwd[i+1:i+1+pred_len, 1:].mean(axis=1)*w_nor[0, :]
                                            + sample_bwd[i+1:i+1+pred_len, 1:].mean(axis=1)*w_nor[1, :])
            df['imp_no-const'][idx+1:idx+4] = (sample_fwd[i+1:i+1+pred_len, 1:].mean(axis=1)*w_nor[0, :]
                                               + sample_bwd[i+1:i+1+pred_len, 1:].mean(axis=1)*w_nor[1, :])
            # i += pred_len
            i += 4
        else:  # detected acc.
            # 4-1. w/ const.
            pred_len = nan_len+1
            r = df['injected'][idx]/sum(sample_fwd[i:i+pred_len, 1:].mean(axis=1)*w_acc[0, :]
                                        + sample_bwd[i:i+pred_len, 1:].mean(axis=1)*w_acc[1, :])
            df['imp_const'][idx:idx+4] = r*(sample_fwd[i:i+pred_len, 1:].mean(axis=1)*w_acc[0, :]
                                            + sample_bwd[i:i+pred_len, 1:].mean(axis=1)*w_acc[1, :])
            # 4-2. w/o const.
            pred_len = nan_len
            df['imp_no-const'][idx+1:idx+4] = (sample_fwd[i+1:i+1+pred_len, 1:].mean(axis=1)*w_nor[0, :]
                                               + sample_bwd[i+1:i+1+pred_len, 1:].mean(axis=1)*w_nor[1, :])
            # i += (nan_len+1)
            i += 4

    accuracy_by_cases(df)

