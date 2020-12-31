"""
Accumulation detection with nearest neighbor (FIN)
 and fwd-bwd joint imputation with AR
  ~ considering WORKDAY TYPE

2020. 12. 29. Tue.
Soyeong Park
"""
##############################
from funcs import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
# test_house = '1dcb5feb'
f_fwd, f_bwd = 24, 24
nan_len = 5


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)

threshold_df = pd.DataFrame([], columns=['test_house', 'thld'])
for test_house in data.columns:
    print(f'********** TEST HOUSE {test_house} start - {np.where(data.columns == test_house)[0][0]}th')
    data_col = data[test_house]
    df = pd.DataFrame([], index=data_col.index)
    df['values'] = data_col.copy()
    df['nan'] = chk_nan_bfaf(data_col)
    df['holiday'] = load_calendar(2017, 2019)[data_col.index[0]:data_col.index[-1]]


    ##############################
    # 2. injection
    df['injected'], df['mask_inj'] = inject_nan_acc_nanlen(data_col, n_len=nan_len, p_nan=1, p_acc=0.25)


    ##############################
    # 3. accumulation detection
    idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
    idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
    detected = np.zeros(len(data_col))
    detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
    detected[idx_detected_acc.astype('int')] = 4
    df['mask_detected'] = detected


    ##############################
    # 4. imputation
    print(f'***** 2. imputation : LINEAR')

    df['imp_const'] = df['injected'].copy()
    df['imp_no-const'] = df['injected'].copy()

    # 4-1. normal imputation - idx_detected_nor
    for idx in idx_detected_nor:
        # idx 있는 곳만 injection 남겨서 imputation
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+nan_len+1] = df['mask_inj'][idx:idx+nan_len+1]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
        df['imp_const'][idx+1:idx+nan_len+1] = fcst_bidirec1
        df['imp_no-const'][idx+1:idx+nan_len+1] = fcst_bidirec1

    # 4-2. acc. imputation - idx_detected_acc
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        # data_inj_temp[idx:idx+nan_len+1] = data_inj[idx:idx+nan_len+1]
        data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+nan_len+1] = 2
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len+1)
        acc = data_inj_temp[idx]
        fcst_bidirec1 = fcst_bidirec1*(acc/sum(fcst_bidirec1))
        df['imp_const'][idx:idx+nan_len+1] = fcst_bidirec1

    # 4-2-2. acc. imputation - no constraints
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+nan_len+1] = df['mask_inj'][idx:idx+nan_len+1]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
        df['imp_no-const'][idx+1:idx+nan_len+1] = fcst_bidirec1


    df.to_csv(f'201027_{test_house}_nan{nan_len}_result.csv')
    print(f'********** TEST HOUSE {test_house} end, saved successfully\n\n')
