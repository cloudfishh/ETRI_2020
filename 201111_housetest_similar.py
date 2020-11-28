"""
Accumulation detection with similar days
 and fwd-bwd joint imputation with AR
- length of NaN = 5 test

2020. 11. 11. Wed.
Soyeong Park
"""
##############################
from funcs import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time


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

threshold_df = pd.DataFrame([], columns=['test_house', 'thld'])
iter = 0
for test_house in data.columns:
# for test_house in range(1):
    if iter != 0:
        list_file = len([l for l in os.listdir('/home/ubuntu/Documents/sypark/2020_ETRI/result_201114_similar') if l.startswith(f'201114_{test_house_before}')])
        if list_file != 3:
            test_house = test_house_before
    start_time = time.time()
    print(f'********** TEST HOUSE {test_house} start - {np.where(data.columns == test_house)[0][0]}th')
    data_col = data[test_house]
    calendar = load_calendar(2017, 2019)

    df = pd.DataFrame([], index=data_col.index)
    df['values'] = data_col.copy()
    df['nan'] = chk_nan_bfaf(data_col)
    df['holiday'] = load_calendar(2017, 2019)[data_col.index[0]:data_col.index[-1]]
    df['org_idx'] = np.arange(0, len(data_col))

    weather = load_weather('incheon', 2017, 2019)[df.index[0][:16]:df.index[-1][:16]]
    weather.index = df.index



    # # # # # # #
    # # # temporary codes
    # feature = weather.columns[0]
    # df['injected'], df['mask_inj'] = inject_nan_acc_nanlen(data_col, n_len=nan_len, p_nan=1, p_acc=0.25)
    # idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
    # idx_target = idx_list[0]    # holiday==0
    # # idx_target = idx_list[17]   # holiday==1
    #
    # # holiday==0 이랑 1이랑 샘플 개수 알아보기
    # for holi in [0, 1]:
    #     print(f'** holi={holi}')
    #     h = []
    #     for idx_temp in [t for t in idx_list if df['holiday'][t]==holi]:
    #         # sample, _, _ = nearest_neighbor(data_col, df['nan'].copy(), idx_temp, calendar)
    #         sample, _, _ = similar_days(df, idx_temp, weather, weather.columns[0])
    #         print(f'{sum(sample!=None)} ', end='')
    #         h.append(sum(sample!=None))
    #     print(f'** holi={holi}, # of samples mean={sum(h)/len(h)}\n')
    # # # # # # #




    ##############################
    # 2. injection
    df['injected'], df['mask_inj'] = inject_nan_acc_nanlen(data_col, n_len=nan_len, p_nan=1, p_acc=0.25)


    ##############################
    # 3. accumulation detection
    print(f'***** 1. detection : SIMILAR DAYS')

    idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
    nan_mask = df['nan'].copy()

    # save the nearest neighbor samples
    sample_list, mean_list, std_list = list(), list(), list()
    for i in range(len(idx_list)):
        idx_target = idx_list[i]
        sample, m, s = similar_days(df, idx_target, weather, weather.columns[0])
        sample_list.append(sample)
        mean_list.append(m)
        std_list.append(s)
    smlr_sample = pd.DataFrame(sample_list)
    smlr_sample.to_csv(f'result_201114_similar/201114_{test_house}_nan{nan_len}_samples.csv')


    # 3-2. z-score
    cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
    detect_sample = pd.read_csv(f'result_201114_similar/201114_{test_house}_nan{nan_len}_samples.csv', index_col=0)
    z_score = (cand['injected'].values - detect_sample.mean(axis=1)) / detect_sample.std(axis=1)
    df['z_score'] = pd.Series(z_score.values, index=df.index[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]])

    # 3-3. threshold test
    detection_result = pd.DataFrame([], columns=['thld', 'MAE', 'MAE_no'])
    i = 0
    for thld in np.arange(0, 40, 0.1):
    # for thld in np.arange(2, 6, 0.25):
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

    detection_result.to_csv(f'result_201114_similar/201114_{test_house}_nan{nan_len}_lossfunc.csv')
    detection_result = pd.read_csv(f'D:/2020_ETRI/result_201115_total-nearest/201027_{test_house}_nan{nan_len}_lossfunc.csv', index_col=0)

    # plt.rcParams.update({'font.size': 14})
    # plt.figure(figsize=(6,4), dpi=400)
    # plt.plot(detection_result['thld'], detection_result['MAE'])
    # plt.plot(detection_result['thld'], detection_result['MAE_no'])
    # plt.axvline(x=detection_result['thld'][detection_result['MAE']==detection_result['MAE'].min()].values[0], color='r',
    #             linewidth=1, linestyle='--')
    # plt.legend(['w/ AOD-AI', 'w/o AOD-AI', 'threshold'], loc='lower right')
    # plt.xlabel('z-score')
    # plt.ylabel('total MAE')
    # plt.xlim([0, 40])
    # plt.ylim([0.005, 0.03])
    # # plt.ylim([0.006, 0.0225])
    # # plt.title(f'{test_house}')
    # plt.tight_layout()
    # plt.savefig('Fig_loss.pdf')
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


    df.to_csv(f'result_201114_similar/201114_{test_house}_nan{nan_len}_result.csv')
    print(f'********** TEST HOUSE {test_house} end, saved successfully / elasped time={time.time()-start_time:.3f}secs\n\n')

    test_house_before = test_house
    iter += 1


