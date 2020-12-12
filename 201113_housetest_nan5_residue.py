"""
Accumulation detection with nearest neighbor
 and fwd-bwd joint imputation with AR
- length of NaN 5 test
- total 354 households, exist 234. have to run remaining 120 households.

2020. 11. 13. Fri.
Soyeong Park
"""
##############################
from funcs import *
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 5


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)

# get the remaining households list
list_files = np.array([l[7:15] for l in os.listdir('D:/2020_ETRI/result_201027') if l.endswith('.csv')])
list_counts = np.array(np.unique(list_files, return_counts=True)).transpose()
# list_weird = list_counts[:, 0][np.where(list_counts[:, 1] != '3')]

list_all = data.columns.values
list_exist = list_counts[:, 0][np.where(list_counts[:, 1] == '3')]
list_haveto = list_all[np.invert(np.isin(list_all, list_exist))]


threshold_df = pd.DataFrame([], columns=['test_house', 'thld'])
for test_house in list_haveto:
    print(f'********** TEST HOUSE {test_house} start - {np.where(data.columns == test_house)[0][0]}th')
    data_col = data[test_house]
    calendar = load_calendar(2017, 2019)
    df = pd.DataFrame([], index=data_col.index)
    df['values'] = data_col.copy()
    df['nan'] = chk_nan_bfaf(data_col)


    ##############################
    # 2. injection
    df['injected'], df['mask_inj'] = inject_nan_acc_nanlen(data_col, n_len=nan_len, p_nan=1, p_acc=0.25)


    ##############################
    # 3. accumulation detection
    print(f'***** 1. detection : NEAREST')
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
    smlr_sample.to_csv(f'result_201113/201027_{test_house}_nan{nan_len}_nearest.csv')


    # 3-2. z-score
    cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
    detect_sample = pd.read_csv(f'result_201113/201027_{test_house}_nan{nan_len}_nearest.csv', index_col=0)
    z_score = (cand['injected'].values - detect_sample.mean(axis=1)) / detect_sample.std(axis=1)
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
            data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
            mask_inj_temp = np.isnan(data_col).astype('float')
            mask_inj_temp[idx:idx+nan_len+1] = df['mask_inj'][idx:idx+nan_len+1]
            trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
            imp_const[idx+1:idx+nan_len+1] = fcst_bidirec1
            imp_no[idx+1:idx+nan_len+1] = fcst_bidirec1

        # acc. imputation - idx_detected_acc
        for idx in idx_detected_acc:
            data_inj_temp = data_col.copy()
            data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
            mask_inj_temp = np.isnan(data_col).astype('float')
            mask_inj_temp[idx:idx+nan_len+1] = 2
            trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len+1)
            acc = data_inj_temp[idx]
            fcst_bidirec1 = fcst_bidirec1*(acc/sum(fcst_bidirec1))
            imp_const[idx:idx+nan_len+1] = fcst_bidirec1
        # acc. imputation - no constraints
        for idx in idx_detected_acc:
            data_inj_temp = data_col.copy()
            data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
            mask_inj_temp = np.isnan(data_col).astype('float')
            mask_inj_temp[idx:idx+nan_len+1] = df['mask_inj'][idx:idx+nan_len+1]
            trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
            imp_no[idx+1:idx+nan_len+1] = fcst_bidirec1

        # accuracy
        temp = pd.DataFrame({'values': data_col, 'imp_const': imp_const, 'imp_no': imp_no}).dropna()
        detection_result.loc[i] = [thld,
                                   mean_absolute_error(temp['values'], temp['imp_const']),
                                   mean_absolute_error(temp['values'], temp['imp_no'])]
        i += 1
        print(f'     * threshold test: {thld}')

    detection_result.to_csv(f'result_201113/201027_{test_house}_nan{nan_len}_lossfunc.csv')
    # detection_result = pd.read_csv(f'D:/2020_ETRI/result_201113/201027_{test_house}_nan{nan_len}_lossfunc.csv', index_col=0)
    detection_result = pd.read_csv(f'D:/2020_ETRI/result_201115_total-nearest//201027_{test_house}_nan{nan_len}_lossfunc.csv', index_col=0)

    # test_house = data_raw.columns[35]
    # thld=7.8
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(6,4), dpi=400)
    plt.plot(detection_result['thld'], detection_result['MAE'])
    plt.plot(detection_result['thld'], detection_result['MAE_no'], linestyle='--')
    plt.axvline(x=detection_result['thld'][detection_result['MAE']==detection_result['MAE'].min()].values[0], color='r',
                linewidth=1, linestyle=':')
    plt.legend(['w/ AOD-AI', 'w/o AOD-AI', 'selected'], loc='lower right', fontsize=14)
    plt.xlabel('z-score threshold')
    plt.ylabel('total MAE')
    plt.xlim([0, 40])
    # plt.ylim([0.005, 0.03])
    # plt.ylim([0.006, 0.0225])
    # plt.title(f'{test_house}')
    plt.tight_layout()
    plt.savefig('Fig_loss.pdf')
    # plt.savefig(f'result/{test_house}/201114_lossfunc_{test_house}_nan{nan_len}.png')

    # threshold = 7.5     # DEEPAR
    # threshold = 3.4   # NEAREST
    threshold = detection_result['thld'][detection_result['MAE']==detection_result['MAE'].min()].values[0]
    threshold_df = threshold_df.append({'test_house': test_house, 'thld': threshold}, ignore_index=True)
    print(f'   ** SELECTED THRESHOLD: {threshold}')

    idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
    idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
    detected = np.zeros(len(data_col))
    detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
    detected[idx_detected_acc.astype('int')] = 4
    df['mask_detected'] = detected

    # idx_injected = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
    # idx_real_nor = np.where(df['mask_inj'] == 3)[0]
    # idx_real_acc = np.where(df['mask_inj'] == 4)[0]
    #
    # idx_detected = np.isin(idx_injected, idx_detected_acc)
    # idx_real = np.isin(idx_injected, idx_real_acc)
    # cm = confusion_matrix(idx_real, idx_detected)
    #
    # group_names = ['TN', 'FP', 'FN', 'TP']
    # group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    # cm_label = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    # cm_label = np.asarray(cm_label).reshape(2, 2)
    #
    # plt.rcParams.update({'font.size': 16})
    # plt.figure(figsize=(4, 4), dpi=400)
    # sns.heatmap(cm, annot=cm_label, fmt='', square=True, cmap='Greys', annot_kws={'size': 15}, # 'gist_gray': reverse
    #             xticklabels=['normal', 'anomaly'], yticklabels=['normal', 'anomaly'], cbar=False)
    # # plt.title(f'{test_house}, {method[17:]}, nan_length=3, threshold={threshold}', fontsize=14)
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.tight_layout()
    # plt.savefig('Fig_cm_(a).pdf')
    # # plt.savefig(f'result/{test_house}/201017_confusion_{method[17:]}_{test_house}.png')


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


    df.to_csv(f'result_201113/201027_{test_house}_nan{nan_len}_result.csv')
    print(f'********** TEST HOUSE {test_house} end, saved successfully\n\n')
