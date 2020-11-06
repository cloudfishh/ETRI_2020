"""
FUNCTIONS
for Anomaly detection
    and fwd-bwd joint imputation

2020. 10. 20. Tue.
Soyeong Park
"""
import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def make_date_index(y1, y2):
    # make datetime index
    # global date_index
    date_index_str = []
    day_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_list_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for y in range(y1, y2+1):
        if y % 4 == 0:
            d_list = day_list_leap
        else:
            d_list = day_list
        for m in range(12):
            for d in range(d_list[m]):
                for h in range(24):
                    date_index_str.append('%d-%02d-%02d %02d:00:00' % (y, m+1, d+1, h))
    date_index = pd.DatetimeIndex(date_index_str, freq='H')
    return date_index


def load_calendar(y1, y2):
    dir_cal = 'D:/2020_ETRI/data/calendar'
    cal = pd.DataFrame([])
    for y in range(y1, y2+1):
        cal = pd.concat([cal, pd.read_csv(f'{dir_cal}/calendar{y}.csv', header=None)])

    calendar = pd.DataFrame(np.empty([len(cal)*24, ]), index=make_date_index(y1, y2))
    for r in range(len(cal)):
        for rr in range(24):
            calendar.iloc[r*24+rr] = cal.iloc[r][4]
    calendar.columns = ['holiday']
    return calendar


def load_apt(dir_data, location, apt):
    if location != 'label':
        load_data = pd.read_csv(f'{dir_data}/SG_data_{location}_비식별화/{apt}.csv', index_col=0)
        load_data = load_data.drop(columns=['Season', 'Weekday'])
    else:
        load_data = pd.read_csv(f'{dir_data}/label_data.csv')
    return load_data


def clear_head(data):
    sum_data = np.array([])
    for i in range(len(data.iloc[:, 0])):
        s = sum(data.iloc[i, :].fillna(0))
        sum_data = np.append(sum_data, s)
    sum_data = pd.DataFrame(sum_data, index=data.index)  # sum_data = row sum. through region (all households, rows)
    nan_data = sum_data.any(1)  # nan 체크된 columns. 전체 세대 중에 value가 하나라도 있으면 True.

    for i in range(len(nan_data)):
        if nan_data[i] != False:
            # print(i)
            break  # i = 처음으로 전체 nan이 끝나는 시점
    nan_data_rev = pd.DataFrame(nan_data.iloc[i:], index=nan_data.index[i:])

    # check the start point
    start_time = data.index[i]
    diff = int(start_time[11:13])
    data_rev = data.iloc[i - diff:, :]
    return data_rev, nan_data_rev


def count_nan_len(data_col):
    nan_length = np.array([])
    num = 0
    for j in range(len(data_col)):
        if np.isnan(data_col[j]):
            num += 1
        else:
            nan_length = np.append(nan_length, num)
            num = 0
    nan_length = np.append(nan_length, num)
    nan_len_nonzero = nan_length[nan_length != 0]
    return nan_len_nonzero


def chk_nan_bfaf(data_col):
    out = []
    nan_list = np.isnan(data_col.values)
    for i in range(1, len(nan_list)-1):
        if nan_list[i]:
            out.append(i-1)
            out.append(i)
            out.append(i+1)
    idx_list = np.array(list(set(out)))
    nan_boolean = pd.Series(np.zeros([len(data_col),]), index=data_col.index)
    for idx in idx_list:
        nan_boolean[idx] = True
    nan_boolean = nan_boolean.to_frame()
    nan_boolean.columns = ['nan']
    return nan_boolean


def inject_nan_acc3(data_col, p_nan=1, p_acc=1):
    n_len_max = 3
    nan_count = count_nan_len(data_col)
    nan_list = np.unique(nan_count, return_counts=True)

    n_inj_list = nan_list[1][0:3]*p_nan
    n1, n2, n3 = int(n_inj_list[0]), int(n_inj_list[1]), int(n_inj_list[2])
    n_inj_sum = n1+n2+n3
    n1, n2, n3 = 0, 0, n_inj_sum

    # nan, bf, af 아닌 인덱스만 뽑아낸 list를 만들고
    # 그 list의 index로 랜덤샘플링
    nan_bool = chk_nan_bfaf(data_col)
    cand = np.array(np.where(nan_bool.values == 0)[0])

    cand_diff = cand-np.roll(cand, 1)
    cand_group = list()
    temp = list()
    for cd in range(len(cand_diff)):
        if cand_diff[cd] == 1:
            temp.append(cand[cd])
        else:
            cand_group.append(temp)
            temp = []
            temp.append(cand[cd])
    cand_group.append(temp)

    candidates = np.array([])
    for g in cand_group[1:]:
        candidates = np.append(candidates, g[:-n_len_max])
    candidates = candidates.astype('int')

    random.seed(0)
    rand = random.sample(range(1, int(len(candidates)/(n_len_max+1))), k=n_inj_sum)
    rand_rev = [(n_len_max+1)*i for i in rand]
    idx_inj = list(candidates[rand_rev])
    idx_inj.sort()

    random.seed(1)
    idx_acc = random.sample(idx_inj, k=int(n_inj_sum*p_acc))
    idx_acc = [i-1 for i in idx_acc]
    idx_acc.sort()

    idx_a = pd.DataFrame([i-1 for i in idx_inj])
    idx_b = idx_a.isin(idx_acc)
    idx_nor = idx_a[idx_b == False].dropna()
    idx_nor = list(idx_nor.values.reshape([len(idx_nor), ]).astype('int'))

    len_list = np.ones(n1)
    len_list = np.append(len_list, np.ones(n2)*2)
    len_list = np.append(len_list, np.ones(n3)*3)
    random.seed(2)
    random.shuffle(len_list)

    inj_mask = nan_bool.copy()
    inj_mask = inj_mask['nan']
    for i in range(len(idx_inj)):
        idx = int(idx_inj[i])
        l_nan = int(len_list[i])
        inj_mask[idx:idx+l_nan] = 2
    inj_mask[idx_nor] = 3
    inj_mask[idx_acc] = 4

    injected = data_col.copy()
    s, k = 0, 0
    for j in range(len(inj_mask)):
        if inj_mask[j] == 4:
            while inj_mask[j+k] > 1:
                s += injected[j+k]
                k += 1
            injected[j] = s
            s = 0
            k = 0
    injected[inj_mask == 2] = np.nan

    print(f'# of injected NaN = {len(idx_inj)}')
    print(f'# of injected acc = {len(idx_acc)}      ** DONE')
    return injected, inj_mask


def nearest_neighbor(data_col, nan_mask, idx_target, calendar):
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
    target_day = int(target_timestamp[8:10])
    target_hour = int(target_timestamp[11:13])
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
    elif target_day > len(data_col)-30:
        target_data = data_day[target_hour, target_day-30:]
        target_mask = mask_day[target_hour, target_day-30:]
    else:
        target_data = data_day[target_hour, target_day-30:target_day+30]
        target_mask = mask_day[target_hour, target_day-30:target_day+30]
    ma = np.ma.masked_array(target_data, mask=target_mask)
    return np.array(ma.tolist()), ma.mean(), ma.std()


def make_bidirectional_input(d_col, n_mask):
    # forward 4주, backward 4주
    # 너무앞 or 너무뒤 or nan 많아서 4주 안 채워지면 그 만큼을 뒤 or 앞에서 더 채워넣음
    d = 24

    idx_inj = np.array(np.where(n_mask == 2))[0]
    train_x_fwd, train_x_bwd, train_y_temp, test_x_temp = [], [], [], []

    # 앞 4주, 뒤 4주 -> train_x에 포함시킬 24point에 nan이 절반 이상이면 pass시킴
    i = 168
    while len(train_x_fwd) < 8:
        # before
        idx_temp = idx_inj - i
        if (idx_temp[0]-(d+1) < 0) | (idx_temp[-1]+(d+1) > len(d_col)):
            pass
        elif (sum(n_mask[idx_temp[0]-d:idx_temp[0]]) > 10*(d/24)) | (sum(n_mask[idx_temp[-1]+1:idx_temp[-1]+(d+1)]) > 10*(d/24)):
            pass
        else:
            train_y_temp.append(d_col[idx_temp])
            train_x_fwd.append(d_col[idx_temp[0]-d:idx_temp[0]])
            train_x_bwd.append(d_col[idx_temp[-1]+1:idx_temp[-1]+(d+1)])

        # after
        idx_temp = idx_inj + i
        if (idx_temp[0]-(d+1) < 0) | (idx_temp[-1]+(d+1) > len(d_col)):
            pass
        elif (sum(n_mask[idx_temp[0]-d:idx_temp[0]]) > 10*(d/24)) | (sum(n_mask[idx_temp[-1]+1:idx_temp[-1]+(d+1)]) > 10*(d/24)):
            pass
        else:
            train_y_temp.append(d_col[idx_temp])
            train_x_fwd.append(d_col[idx_temp[0]-d:idx_temp[0]])
            train_x_bwd.append(d_col[idx_temp[-1]+1:idx_temp[-1]+(d+1)])
        i += 168
    test_x_temp = np.append(d_col[idx_inj[0]-d:idx_inj[0]], d_col[idx_inj[-1]+1:idx_inj[-1]+(d+1)])

    train_x = np.append(np.array(train_x_fwd), np.array(train_x_bwd), axis=1)
    # train_y, test_x = np.array(train_y_temp), test_x_temp.reshape((1, len(test_x_temp)))
    train_y, test_x = np.array(train_y_temp), test_x_temp.copy()

    train_x, train_y, test_x = np.nan_to_num(train_x), np.nan_to_num(train_y), np.nan_to_num(test_x)
    return train_x, train_y, test_x


def linear_prediction(train_x, train_y, test_x, f_len_fwd, f_len_bwd, n_len=3):
    d = 24

    len_tr = len(train_x[0, :])  # 시간 포인트 수
    day_t = len(train_x)
    prediction = np.empty((len(train_x), n_len))
    # fcst = np.empty((len(train_x), len_tr))

    for j in range(0, day_t):
        if day_t > 1:
            x_ar = np.delete(train_x[:, d-f_len_fwd:d+f_len_bwd], j, axis=0)
            y = np.delete(train_y, j, axis=0)
        else:
            x_ar = train_x[:, d-f_len_fwd:d+f_len_bwd]
            y = train_y

        pi_x_ar = np.linalg.pinv(x_ar)
        # lpc_c = np.empty((len(x_ar), f_len))

        lpc_c = np.matmul(pi_x_ar, y)

        test_e = train_x[j, :]
        test_ex = test_e[d-f_len_fwd:d+f_len_bwd]
        prediction[j, :] = np.matmul(test_ex, lpc_c)

    x_ar = train_x[:, d-f_len_fwd:d+f_len_bwd]
    y = train_y
    pi_x_ar = np.linalg.pinv(x_ar)
    # lpc_c = np.empty((len(x_ar), f_len))

    lpc_c = np.matmul(pi_x_ar, y)

    test_ar = train_y[0:len(train_y), :]

    # average_smape = []
    # smape_list = np.zeros((len(prediction), 1))
    # mse_list = np.zeros((len(prediction), 1))

    # for i in range(0, len(prediction)):
    #     smape_list[i] = smape(prediction[i, :], test_ar[i, :])
    #     average_smape = np.mean(smape_list)
    #     mse_list[i] = mean_squared_error(prediction[i, :], test_ar[i, :])

    test_e = test_x
    test_ex = test_e[d-f_len_fwd:d+f_len_bwd]
    test_ex[np.where(test_ex == 0)] = 0.0001

    forecast = np.matmul(test_ex, lpc_c)

    return forecast, prediction


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
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}\n')


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

    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==3)&(df['mask_detected']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_no-const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    print('2-2. true normal / predicted acc - w/o const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}\n')


    # (3) true acc / predicted normal
    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==3))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    print('3. true acc / predicted normal')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}\n')


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

    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==4)&(df['mask_detected']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_no-const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    print('4-2. true acc / predicted acc - w/o const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}\n')


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

    result_true, result_impt = [], []
    for idx in np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]:
        result_true.append(df['values'][idx:idx+4])
        result_impt.append(df['imp_no-const'][idx:idx+4])
    result_true = np.array(result_true)
    result_impt = np.array(result_impt)
    print('5-2. total accuracy - w/o const.')
    print(f'   MAE {mean_absolute_error(result_true, result_impt)}')
    print(f'  RMSE {mean_squared_error(result_true, result_impt)**(1/2)}')