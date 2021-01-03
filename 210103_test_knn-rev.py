"""
k-NN rev

2021. 01. 03. Sun.
Soyeong Park
"""
from funcs import *
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pylab as plt
# from matplotlib import cm
# import seaborn as sns
# from sklearn.metrics import confusion_matrix


def nearest_neighbor_rev(data_col, nan_mask, idx_target, calendar):
    # have to check the calendar start point
    cal_idx = calendar.index.to_frame().astype('str').reset_index(drop=True)
    str_idx = cal_idx[cal_idx == data_col.index[0]].dropna().index[0]
    end_idx = cal_idx[cal_idx == data_col.index[-1]].dropna().index[0]
    cal_rev = calendar.iloc[str_idx:end_idx + 1]
    cal_rev.index = data_col.index

    # reshape data vector
    data_day = data_col.values.reshape([int(data_col.shape[0]/24), -1])
    data_day = data_day.transpose()

    # get the target day/hour/holiday
    target_timestamp = data_col.index[idx_target]
    # target_day = int(target_timestamp[8:10])
    # target_hour = int(target_timestamp[11:13])
    target_ts_delta = pd.Timestamp(target_timestamp) - pd.Timestamp(data_col.index[0])
    target_day = target_ts_delta.components.days
    target_hour = target_ts_delta.components.hours
    target_holi = calendar.loc[target_timestamp][0]

    if target_holi == 1:
        target_cal = np.invert(cal_rev.values.astype('bool')).astype('int')
        mask_temp = np.logical_or(nan_mask.values.reshape(len(data_col), 1), target_cal)
    else:
        mask_temp = np.logical_or(cal_rev.values.reshape(len(data_col), 1), nan_mask.values.reshape(len(data_col), 1))

    mask_day = mask_temp.reshape([int(mask_temp.shape[0] / 24), -1])
    mask_day = mask_day.transpose()
    mask_day[:, target_day] = True

    if target_day < 30:
        target_data = data_day[target_hour, :target_day+30]
        target_mask = mask_day[target_hour, :target_day+30]
    elif target_day > data_day.shape[1]-30:
        target_data = data_day[target_hour, target_day-30:]
        target_mask = mask_day[target_hour, target_day-30:]
    else:
        target_data = data_day[target_hour, target_day-30:target_day+30]
        target_mask = mask_day[target_hour, target_day-30:target_day+30]

    # target_mask_rev = np.invert(target_mask)
    # ma = np.ma.masked_array(target_data, mask=target_mask_rev)
    ma = np.ma.masked_array(target_data, mask=target_mask)
    return np.array(ma.tolist()), ma.mean(), ma.std()


nan_len = 5
calendar = load_calendar(2017, 2019)
df_all = pd.read_csv('D:/202010_energies/201207_result_aodsc+owa_spline-rev-again.csv', index_col=0)

df_all = df_all.astype({'house': 'str'})

h_list_raw = np.unique(df_all['house'].values.astype('str'), return_index=True)
house_list = h_list_raw[0].astype('str')[np.argsort(h_list_raw[1])]

# house = house_list[5]
# house = house_list[238]
# house = '68181c16'

for test_house in house_list:
    print(f'***** {test_house} start')
    starttime = time.time()
    df = df_all[df_all['house'] == test_house]
    df.index = df['Time']
    data_col = df['values']

    idx_list = np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]

    # k-NN samples - rev version
    sample_list = list()
    for i in range(len(idx_list)):
        idx_target = idx_list[i]
        sample, _, _ = nearest_neighbor_rev(df['values'], df['nan'].copy(), idx_target, calendar)
        sample_list.append(sample)
    sample_near = pd.DataFrame(sample_list)
    sample_near.to_csv(f'result_210103/210103_knn-rev_sample_{test_house}.csv')

    # 3-2. z-score
    cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()

    # find the optimal threshold
    print(f'  *** 1. detection')
    detect_sample = sample_near
    z_score = (cand['injected'].values - detect_sample.mean(axis=1)) / detect_sample.std(axis=1)
    df['z_score'] = pd.Series(z_score.values,
                              index=df.index[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]])

    # 3-3. threshold test
    print(f'    * threshold test: ', end='')
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
            data_inj_temp[idx:idx + 4] = df['injected'][idx:idx + 4]
            mask_inj_temp = np.isnan(data_col).astype('float')
            mask_inj_temp[idx:idx + 4] = df['mask_inj'][idx:idx + 4]
            trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1)
            imp_const[idx + 1:idx + 4] = fcst_bidirec1
            imp_no[idx + 1:idx + 4] = fcst_bidirec1

        # acc. imputation - idx_detected_acc
        for idx in idx_detected_acc:
            data_inj_temp = data_col.copy()
            data_inj_temp[idx:idx + 4] = df['injected'][idx:idx + 4]
            mask_inj_temp = np.isnan(data_col).astype('float')
            mask_inj_temp[idx:idx + 4] = 2
            trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=4)
            acc = data_inj_temp[idx]
            fcst_bidirec1 = fcst_bidirec1 * (acc / sum(fcst_bidirec1))
            imp_const[idx:idx + 4] = fcst_bidirec1
        # acc. imputation - no constraints
        for idx in idx_detected_acc:
            data_inj_temp = data_col.copy()
            data_inj_temp[idx:idx + 4] = df['injected'][idx:idx + 4]
            mask_inj_temp = np.isnan(data_col).astype('float')
            mask_inj_temp[idx:idx + 4] = df['mask_inj'][idx:idx + 4]
            trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
            fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=3)
            imp_no[idx + 1:idx + 4] = fcst_bidirec1

        # accuracy
        temp = pd.DataFrame({'values': data_col, 'imp_const': imp_const, 'imp_no': imp_no}).dropna()
        detection_result.loc[i] = [thld,
                                   mean_absolute_error(temp['values'], temp['imp_const']),
                                   mean_absolute_error(temp['values'], temp['imp_no'])]
        i += 1
        print(f'{thld}', end=' ')


    detection_result.to_csv(f'result_210103/210103_knn-rev_lossfunc_{test_house}.csv')
    # detection_result = pd.read_csv(f'result_210103/210103_knn-rev_sample_{test_house}.csv', index_col=0)

    threshold = detection_result['thld'][detection_result['MAE'] == detection_result['MAE'].min()].values[0]
    print(f'   ** SELECTED THRESHOLD: {threshold}')

    idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
    idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
    detected = np.zeros(len(data_col))
    detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
    detected[idx_detected_acc.astype('int')] = 4
    df['mask_detected'] = detected

    ##############################
    # 4. imputation
    print(f'  *** 2. imputation : JOINT')

    df['imp_const'] = df['injected'].copy()
    df['imp_no-const'] = df['injected'].copy()

    ##############################
    # 4-1. JOINT
    print('    * - JOINT start')

    df['joint'] = df['injected'].copy()
    df['joint_aod'] = df['injected'].copy()
    df['joint_aod_sc'] = df['injected'].copy()

    # 4-1. normal imputation - idx_detected_nor
    print('    * - JOINT detected nor cases')
    for idx in idx_detected_nor:
        # idx 있는 곳만 injection 남겨서 imputation
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx + nan_len + 1] = df['injected'][idx:idx + nan_len + 1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx + nan_len + 1] = df['mask_inj'][idx:idx + nan_len + 1]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
        df['joint_aod_sc'][idx + 1:idx + nan_len + 1] = fcst_bidirec1
        df['joint_aod'][idx + 1:idx + nan_len + 1] = fcst_bidirec1
        df['joint'][idx + 1:idx + nan_len + 1] = fcst_bidirec1
        print(f'{idx}', end=' ')

    # 4-2-1. acc. imputation - without detection result
    print('    * - JOINT detected acc cases - vanilla')
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx + nan_len + 1] = df['injected'][idx:idx + nan_len + 1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx + nan_len + 1] = df['mask_inj'][idx:idx + nan_len + 1]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
        df['joint'][idx + 1:idx + nan_len + 1] = fcst_bidirec1
        print(f'{idx}', end=' ')

    # 4-2-2. acc. imputation - aware detection result
    print('    * - JOINT detected acc cases - aod, aodsc')
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        # data_inj_temp[idx:idx+nan_len+1] = data_inj[idx:idx+nan_len+1]
        data_inj_temp[idx:idx + nan_len + 1] = df['injected'][idx:idx + nan_len + 1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx + nan_len + 1] = 2
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len + 1)
        df['joint_aod'][idx:idx + nan_len + 1] = fcst_bidirec1
        acc = data_inj_temp[idx]
        fcst_bidirec1 = fcst_bidirec1 * (acc / sum(fcst_bidirec1))
        df['joint_aod_sc'][idx:idx + nan_len + 1] = fcst_bidirec1
        print(f'{idx}', end=' ')

    print('    * - JOINT finished')

    ##############################
    # 4-2. LINEAR
    print('    * - LINEAR start')
    linear = df[{'values', 'injected', 'mask_detected'}].copy()
    linear['linear'] = linear['injected'].copy()
    linear['linear_aod'] = linear['injected'].copy()
    linear['linear_aodsc'] = linear['injected'].copy()
    linear = linear.reset_index()

    for idx in np.where((linear['mask_detected'] == 3) | (linear['mask_detected'] == 4))[0]:
        temp_nocon, temp_const = linear['values'].copy(), linear['values'].copy()
        temp_nocon[:idx], temp_const[:idx] = linear['values'][:idx], linear['values'][:idx]
        temp_nocon[idx:idx + nan_len + 2], temp_const[idx:idx + nan_len + 2] = linear['injected'][idx:idx + nan_len + 2], linear['injected'][idx:idx + nan_len + 2]

        # w/o const
        p, q = 0, 0
        while pd.isna(temp_nocon[idx - p]):
            p += 1
        while pd.isna(temp_nocon[idx + nan_len + 2 + q]):
            q += 1
        linear['linear'][idx:idx + nan_len + 2] = temp_nocon[idx - p:idx + nan_len + 2 + q].interpolate(method='linear')

        # w/ const
        if linear['mask_detected'][idx] == 3:
            p, q = 0, 0
            while pd.isna(temp_const[idx - p]):
                p += 1
            while pd.isna(temp_const[idx + nan_len + 2 + q]):
                q += 1
            linear['linear_aod'][idx:idx + nan_len + 2] = temp_const[idx - p:idx + nan_len + 2 + q].interpolate(method='linear')
            linear['linear_aodsc'][idx:idx + nan_len + 2] = temp_const[idx - p:idx + nan_len + 2 + q].interpolate(method='linear')

        else:  # 4
            p, q = 0, 0
            while pd.isna(temp_const[idx - 1 - p]):
                p += 1
            while pd.isna(temp_const[idx + nan_len + 2 + q]):
                q += 1
            s = temp_const[idx]
            temp_const[idx] = np.nan
            li_temp = temp_const[idx - 1 - p:idx + nan_len + 2 + q].interpolate(method='linear').loc[idx:idx + nan_len]
            linear['linear_aod'][idx:idx + nan_len + 1] = li_temp
            linear['linear_aodsc'][idx:idx + nan_len + 1] = li_temp * (s / sum(li_temp.values))
        print(f'{idx} ', end='')
    linear.index = linear['Time']

    df['linear'] = linear['linear'].copy()
    df['linear_aod'] = linear['linear_aod'].copy()
    df['linear_aodsc'] = linear['linear_aodsc'].copy()
    print('    * - LINEAR finished')

    ##############################
    # 4-3. SPLINE
    print('    * - SPLINE start')
    spline = df[{'values', 'injected', 'mask_detected'}].copy().reset_index(drop=True)
    spline['spline'] = spline['injected'].copy()
    spline['spline_aod'] = spline['injected'].copy()
    spline['spline_aodsc'] = spline['injected'].copy()

    for idx in np.where((spline['mask_detected'] == 3) | (spline['mask_detected'] == 4))[0]:
        temp_nocon, temp_const = spline['values'].copy(), spline['values'].copy()
        temp_nocon[:idx], temp_const[:idx] = spline['values'][:idx], spline['values'][:idx]
        temp_nocon[idx:idx + nan_len + 2], temp_const[idx:idx + nan_len + 2] = spline['injected'][idx:idx + nan_len + 2], spline['injected'][idx:idx + nan_len + 2]
        # w/o const
        p, q = 24, 24
        spline['spline'][idx + 1:idx + nan_len + 1] = temp_nocon[idx - p:idx + nan_len + 2 + q].interpolate(method='spline', order=3).loc[idx + 1:idx + nan_len]

        # w/ const
        if spline['mask_detected'][idx] == 3:
            p, q = 24, 24
            spline['spline_aod'][idx + 1:idx + nan_len + 1] = temp_const[idx - p:idx + nan_len + 2 + q].interpolate(method='spline', order=3).loc[idx + 1:idx + nan_len]
            spline['spline_aodsc'][idx + 1:idx + nan_len + 1] = temp_const[idx - p:idx + nan_len + 2 + q].interpolate(method='spline', order=3).loc[idx + 1:idx + nan_len]

        else:  # 4
            p, q = 24, 24
            s = temp_const[idx]
            temp_const[idx] = np.nan
            li_temp = temp_const[idx - 1 - p:idx + nan_len + 2 + q].interpolate(method='spline', order=3).loc[idx:idx + nan_len]
            spline['spline_aod'][idx:idx + nan_len + 1] = li_temp
            spline['spline_aodsc'][idx:idx + nan_len + 1] = li_temp * (s / sum(li_temp.values))
        print(f'{idx} ', end='')

    df['spline'] = spline['spline'].values.copy()
    df['spline_aod'] = spline['spline_aod'].values.copy()
    df['spline_aodsc'] = spline['spline_aodsc'].values.copy()
    print('    * - SPLINE finished')

    df.to_csv(f'result_210103/210103_knn-rev_result_{test_house}.csv')
    print(f'***** {test_house} end - elapsed time {time.time() - starttime} secs\n')


df_all.to_csv('D:/202010_energies/201214_result_kmeans-added.csv')
