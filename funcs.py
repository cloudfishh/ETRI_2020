import os
import math
import random
import pandas as pd
import numpy as np
# import mxnet as mx
# from gluonts.model.deepar import DeepAREstimator
# from gluonts.trainer import Trainer
# from gluonts.dataset.common import ListDataset


def set_dir(loc):
    # global list_apt
    dir_data = 'D:/2020_ETRI/data/'
    list_loc = ['SG_data_광주_비식별화',
                'SG_data_나주_비식별화',
                'SG_data_대전_비식별화',
                'SG_data_서울_비식별화',
                'SG_data_인천_비식별화']
    os.chdir(dir_data + list_loc[loc])
    list_apt = os.listdir(os.getcwd())
    list_apt_csv = [a for a in list_apt if a.endswith('.csv')]
    return list_apt_csv


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


def load_household(list_apt, idx=0):
    apt = list_apt[idx]
    dir_house = os.getcwd() + '/' + apt
    load_data = pd.read_csv(dir_house, index_col=0)
    load_data = load_data.drop(columns=['Season', 'Weekday'])
    return load_data


def load_labeled():
    data = pd.read_csv('D:/2020_ETRI/data/label_data.csv', index_col=0)
    data = data.drop(columns=['Season', 'Weekday'])
    return data


def load_weather(loc, y1, y2):
    dir_weather = 'D:/2020_ETRI/data/weather'
    weather = pd.DataFrame([])
    for y in range(y1, y2+1):
        weather = pd.concat([weather, pd.read_csv(f'{dir_weather}/weather_{loc}_{y}.csv', header=None, encoding='CP949')])
    weather.columns = weather.iloc[0, :]
    weather = weather.drop(0, axis=0)
    weather.index = weather['일시']
    weather = weather.drop(columns=['지점', '지점명', '일시', '운형(운형약어)']).astype(np.float64)
    return weather


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


def inject_nan_acc(data_col, p_nan=1, p_acc=1):
    nan_count = count_nan_len(data_col)
    nan_list = np.unique(nan_count, return_counts=True)

    n_inj_list = nan_list[1][0:3] * p_nan
    n1, n2, n3 = int(n_inj_list[0]), int(n_inj_list[1]), int(n_inj_list[2])
    n_inj_sum = n1 + n2 + n3

    # nan, bf, af 아닌 인덱스만 뽑아낸 list를 만들고
    # 그 list의 index로 랜덤샘플링
    # 간격 최소4 이어야 하니까 list의 index를 /4 해서 랜덤으로 뽑고 거기에 3을 곱해주면 되겟죵
    nan_bool = chk_nan_bfaf(data_col)
    candidates = np.array(np.where(nan_bool.values == 0))

    random.seed(0)
    rand = random.sample(range(1, int(len(candidates[0])/6)), k=n_inj_sum)
    rand_rev = [6 * i for i in rand]
    idx_inj = list(candidates[0][rand_rev])
    idx_inj.sort()

    random.seed(1)
    idx_acc = random.sample(idx_inj, k=int(n_inj_sum * p_acc))
    idx_acc = [i - 1 for i in idx_acc]
    idx_acc.sort()

    idx_a = pd.DataFrame([i-1 for i in idx_inj])
    idx_b = idx_a.isin(idx_acc)
    idx_nor = idx_a[idx_b == False].dropna()
    idx_nor = list(idx_nor.values.reshape([len(idx_nor), ]).astype('int'))

    len_list = np.ones(n1)
    len_list = np.append(len_list, np.ones(n2) * 2)
    len_list = np.append(len_list, np.ones(n3) * 3)
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
    # 간격 최소4 이어야 하니까 list의 index를 /4 해서 랜덤으로 뽑고 거기에 3을 곱해주면 되겟죵
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


def inject_nan_acc_nanlen(data_col, n_len=3, p_nan=1, p_acc=0.25):
    n_len_max = n_len
    nan_count = count_nan_len(data_col)
    nan_list = np.unique(nan_count, return_counts=True)

    n_inj_list = nan_list[1][0:3]*p_nan
    n1, n2, n3 = int(n_inj_list[0]), int(n_inj_list[1]), int(n_inj_list[2])
    n_inj_sum = n1+n2+n3
    # n1, n2, n3 = 0, 0, n_inj_sum
    n = n_inj_sum

    # nan, bf, af 아닌 인덱스만 뽑아낸 list를 만들고
    # 그 list의 index로 랜덤샘플링
    # 간격 최소4 이어야 하니까 list의 index를 /4 해서 랜덤으로 뽑고 거기에 3을 곱해주면 되겟죵
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
    rand = random.sample(range(1, int(len(candidates)/(n_len_max+1))), k=n)
    rand_rev = [(n_len_max+1)*i for i in rand]
    idx_inj = list(candidates[rand_rev])
    idx_inj.sort()

    random.seed(1)
    idx_acc = random.sample(idx_inj, k=int(n*p_acc))
    idx_acc = [i-1 for i in idx_acc]
    idx_acc.sort()

    idx_a = pd.DataFrame([i-1 for i in idx_inj])
    idx_b = idx_a.isin(idx_acc)
    idx_nor = idx_a[idx_b == False].dropna()
    idx_nor = list(idx_nor.values.reshape([len(idx_nor), ]).astype('int'))

    # len_list = np.ones(n1)
    # len_list = np.append(len_list, np.ones(n2)*2)
    # len_list = np.append(len_list, np.ones(n3)*3)
    len_list = np.ones(n) * n_len_max
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


def check_accumulation(data_col, calendar):
    # have to check the calendar start point
    cal_idx = calendar.index.to_frame().astype('str').reset_index(drop=True)
    str_idx = cal_idx[cal_idx == data_col.index[0]].dropna().index[0]
    end_idx = cal_idx[cal_idx == data_col.index[-1]].dropna().index[0]
    cal_rev = calendar.iloc[str_idx:end_idx + 1]
    cal_rev.index = data_col.index

    # data + nan boolean array + W/N boolean array
    nan_bool = chk_nan_bfaf(data_col)
    nan_bool.index, cal_rev.index = data_col.index, data_col.index
    # data_rev = pd.concat([data_col, nan_bool, cal_rev], axis=1)
    # data_rev.columns = [data_col.name, 'nan', 'holiday']

    # reshape data vector
    data_day = data_col.values.reshape([int(data_col.shape[0]/24), -1])
    data_day = data_day.transpose()

    idx_list = np.array([])
    idx_dict = dict()
    for i in range(1, len(data_col)-1):                 # start at 1, end at end-1
        # check before
        if np.isnan(data_col[i]) & ~np.isnan(data_col[i-1]):
            target_value = data_col[i-1]
            target_day = math.floor(i/24)
            target_hour = int(data_col.index[i-1][11:13])

            target_holi = calendar.iloc[i-1][0]
            if target_holi == 1:
                # target_cal = np.invert(data_rev['holiday'].values.astype('bool')).astype('int')
                # mask_temp = np.logical_or(data_rev['nan'].values, target_cal)
                target_cal = np.invert(cal_rev.values.astype('bool')).astype('int')
                mask_temp = np.logical_or(nan_bool.values, target_cal)
            else:
                # mask_temp = np.logical_or(data_rev['nan'].values, data_rev['holiday'].values)
                mask_temp = np.logical_or(cal_rev.values, nan_bool.values)

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
            m = ma.mean()
            s = ma.std()

            if not (m-3*s) < target_value < (m+3*s):    # over 3 sigma, check index
                idx_list = np.append(idx_list, i-1)
                idx_dict[i-1] = ma.tolist()

        # check after
        if np.isnan(data_col[i]) & ~np.isnan(data_col[i+1]):
            target_value = data_col[i+1]
            target_day = math.floor(i/24)
            target_hour = int(data_col.index[i+1][11:13])

            target_holi = calendar.iloc[i+1][0]
            if target_holi == 1:
                target_cal = np.invert(cal_rev.values.astype('bool')).astype('int')
                mask_temp = np.logical_or(nan_bool.values, target_cal)
            else:
                mask_temp = np.logical_or(cal_rev.values, nan_bool.values)

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
            m = ma.mean()
            s = ma.std()

            if not (m-3*s) < target_value < (m+3*s):    # over 3 sigma, check index
                idx_list = np.append(idx_list, i+1)
                idx_dict[i+1] = ma
    return np.unique(idx_list), idx_dict


def check_accumulation_injected(data_col, inj_mask, calendar, sigma=3):
    # have to check the calendar start point
    cal_idx = calendar.index.to_frame().astype('str').reset_index(drop=True)
    str_idx = cal_idx[cal_idx == data_col.index[0]].dropna().index[0]
    end_idx = cal_idx[cal_idx == data_col.index[-1]].dropna().index[0]
    cal_rev = calendar.iloc[str_idx:end_idx + 1]
    cal_rev.index = data_col.index

    # data + nan boolean array + W/N boolean array
    nan_bool = chk_nan_bfaf(data_col)
    nan_bool.index, cal_rev.index = data_col.index, data_col.index

    # reshape data vector
    data_day = data_col.values.reshape([int(data_col.shape[0]/24), -1])
    data_day = data_day.transpose()

    # get candidates ~ inj_mask==3 & 4
    candidate = data_col[(inj_mask == 3) | (inj_mask == 4)]

    idx_list = np.array([])
    idx_dict = dict()
    mean_list = np.array([])
    for i in range(len(candidate)):
        target_value = candidate[i]
        target_day = int(candidate.index[i][8:10])
        target_hour = int(candidate.index[i][11:13])

        target_holi = calendar.loc[candidate.index[i]][0]

        if target_holi == 1:
            target_cal = np.invert(cal_rev.values.astype('bool')).astype('int')
            mask_temp = np.logical_or(nan_bool.values, target_cal)
        else:
            mask_temp = np.logical_or(cal_rev.values, nan_bool.values)

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
        m = ma.mean()
        s = ma.std()

        if not (m-sigma*s) < target_value < (m+sigma*s):    # over 3 sigma, check index
            idx_temp = data_col.index.get_loc(candidate.index[i])
            idx_list = np.append(idx_list, idx_temp)
            idx_dict[idx_temp] = ma.tolist()
        mean_list = np.append(mean_list, m)
    return np.unique(idx_list), mean_list


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


def similar_days(df, idx_target, weather, feature):
    # 특정 기상 요소를 기준으로 비슷한 날들 가져오기.
    # 기존 nearest neighbor는 앞뒤 한 달 (총 두 달) 중에서 holiday 일치하는 거만 가져왔음.
    # 그럼 similar day는 서치하는 범위를 어케 잡아야 할 것 같나?
    # 전체로 잡자 ㅋㅋㅋㅋ ㅋㅋㅋ 뭐 어때 좀 오래 걸리긴 하겠지만 그게 확실하겠지
    # 전체로 잡고 총 개수는 nearest neighbor랑 맞추자.

    # 특정 기상 요소를 기준으로, 비슷한 n개 (n = 45 if holiday==0 else 15) 가져오기
    # holiday도 고려하기

    # target idx가 속해있는 하루를 단위로 잡고 비슷한 '날'을 n개 가져오면 되겠지 일단
    feat_col = weather[feature]
    target_hour = int(weather.index[idx_target][11:13])
    # target_day = df['values'][idx_target-target_hour:idx_target+(24-target_hour)]
    feat_day = weather[feature][idx_target-target_hour:idx_target+(24-target_hour)]
    diff_list = []
    for i in range(int(weather.shape[0]/24)):
        diff_list.append(sum(abs(feat_day.values-feat_col[24*i:24*(i+1)].values)))
    diff = pd.DataFrame(diff_list, index=[24*d+15 for d in range(len(diff_list))]).sort_values(by=0)
    diff['Time'] = df.index[diff.index]
    diff['holiday'] = df['holiday'][diff.index].values
    diff['nan'] = df['nan'][diff.index].values
    # diff['values'] = df['values'][diff.index].values

    # plt.figure()
    # plt.plot(diff_notsort[0])
    # plt.xticks(ticks=[x for x in range(0, weather.shape[0], 24*30)],
    #            labels=[diff_notsort['Time'][x+15] for x in range(0, weather.shape[0]-15, 24*30)],
    #            rotation=45)
    # plt.tight_layout()

    # sample = np.array([diff[0][x] for x in diff.index[:40] if diff['holiday'][x]==df['holiday'][idx_target]])
    sample = np.array([df['values'].iloc[x] for x in diff.index[:60]
                       if (diff['holiday'][x]==df['holiday'][idx_target])&(diff['nan'][x]==0)])

    return sample, sample.mean(), sample.std()


def inject_nan_imputation(d_col, n_mask, n_len=3):
    # d_col에서 n_len=3인 window를 moving해서 모든 데이터를 출력해주자. output: 2D data, 2D mask
    # forward, backward 각 4일씩 넣을 거니까... 앞 뒤 5주~-5주까지 범위로. -> 이건 그냥 후처리 해주자...
    data_injected = []
    mask_injected = []
    d_temp = d_col.copy()
    n_temp = n_mask.copy()
    i = 0
    while i <= len(d_col)-n_len-1:
        if sum(n_mask.values[i:i+n_len+1]) == 0:      # inject할 범위+1 중 nan이 없음 (+1 안 하면 nan 길이가 늘어나니까)
            d_temp[i:i+n_len] = np.nan
            n_temp[i:i+n_len] = 2
            d = d_temp.values
            n = n_temp.values.reshape(len(n_temp),)
            data_injected.append(d)
            mask_injected.append(n)

            d_temp = d_col.copy()
            n_temp = n_mask.copy()
            i += 1
        else:
            i += 1
    data_injected_nd = np.array(data_injected).transpose()
    mask_injected_nd = np.array(mask_injected).transpose()
    return data_injected_nd, mask_injected_nd


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

    if idx_inj[0] < 24:
        temp = np.zeros([24-idx_inj[0],])
        temp[:] = np.nan
        test_x_temp = np.append(np.append(temp, d_col[:idx_inj[0]].values), d_col[idx_inj[-1]+1:idx_inj[-1]+(d+1)])
    elif idx_inj[-1] > (len(d_col)-24):
        temp = np.zeros([idx_inj[-1]+24-len(d_col)+1,])
        temp[:] = np.nan
        test_x_temp = np.append(d_col[idx_inj[0]-d:idx_inj[0]], np.append(d_col[idx_inj[-1]+1:].values, temp))
    else:
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


def mape(A, F):
    return 100 / len(A) * np.sum((np.abs(A - F))/A)


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

#
# def model_deepar(len_unit, acc, feature=True, epochs=10):
#     if acc == 4:        # acc.
#         estimator = DeepAREstimator(
#             freq='1H',
#             prediction_length=4,
#             context_length=len_unit-4,
#             num_layers=2,
#             num_cells=40,
#             cell_type='lstm',
#             dropout_rate=0.1,
#             # use_feat_static_cat=feature,
#             # cardinality=[[672]],
#             use_feat_dynamic_real=feature,
#             # embedding_dimension=20,
#             scaling=True,
#             lags_seq=None,
#             time_features=None,
#             trainer=Trainer(ctx=mx.cpu(0), epochs=epochs, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=336, ),
#         )
#     elif acc == 3:       # normal
#         estimator = DeepAREstimator(
#             freq='1H',
#             prediction_length=3,
#             context_length=len_unit-3,
#             num_layers=2,
#             num_cells=40,
#             cell_type='lstm',
#             dropout_rate=0.1,
#             # use_feat_static_cat=feature,
#             # cardinality=[[672]],
#             use_feat_dynamic_real=feature,
#             # embedding_dimension=20,
#             scaling=True,
#             lags_seq=None,
#             time_features=None,
#             trainer=Trainer(ctx=mx.cpu(0), epochs=epochs, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=336, ),
#         )
#     else:
#         return f'acc input values wrong'
#     return estimator
#
#
# def bidirec_dataset_deepar(df, idx, len_unit, len_train):
#     if df['mask_detected'][idx] == 3:        # normal
#         pred_len = 3
#     elif df['mask_detected'][idx] == 4:
#         pred_len = 4
#     else:
#         return f'candidate idx is wrong: {idx}'
#
#     if idx < len_unit+len_train+1:
#         # backward
#         # idx_trn_bwd_start, idx_trn_bwd_end = idx+len_unit, idx+len_unit+len_train-1
#         # idx_tst_bwd_start, idx_tst_bwd_end = idx, idx+len_unit-1
#         idx_trn_bwd_start, idx_trn_bwd_end = idx+4-pred_len+len_unit, idx+4-pred_len+len_unit+len_train-1
#         idx_tst_bwd_start, idx_tst_bwd_end = idx+4-pred_len, idx+4-pred_len+len_unit-1
#
#         trn_bwd = ListDataset([{'start': df.index[idx_trn_bwd_start],
#                                 'target': df['values'][df.index[idx_trn_bwd_start]:df.index[idx_trn_bwd_end]][::-1].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_trn_bwd_start]:df.index[idx_trn_bwd_end]][::-1].values.reshape((1,len_train))}], freq='1H')
#         tst_bwd = ListDataset([{'start': df.index[idx_tst_bwd_start],
#                                 'target': df['values'][df.index[idx_tst_bwd_start]:df.index[idx_tst_bwd_end]][::-1].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_tst_bwd_start]:df.index[idx_tst_bwd_end]][::-1].values.reshape((1,len_unit))}], freq='1H')
#
#         trn_fwd, tst_fwd = trn_bwd, tst_bwd
#
#     elif idx > df.shape[0]-len_unit-len_train-1:
#         # forward
#         idx_trn_fwd_start, idx_trn_fwd_end = idx-len_unit-len_train+pred_len+1, idx-len_unit+pred_len
#         idx_tst_fwd_start, idx_tst_fwd_end = idx-len_unit+pred_len+1, idx+pred_len
#
#         trn_fwd = ListDataset([{'start': df.index[idx_trn_fwd_start],
#                                 'target': df['values'][df.index[idx_trn_fwd_start]:df.index[idx_trn_fwd_end]].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_trn_fwd_start]:df.index[idx_trn_fwd_end]].values.reshape((1,len_train))}], freq='1H')
#         tst_fwd = ListDataset([{'start': df.index[idx_tst_fwd_start],
#                                 'target': df['values'][df.index[idx_tst_fwd_start]:df.index[idx_tst_fwd_end]].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_tst_fwd_start]:df.index[idx_tst_fwd_end]].values.reshape((1,len_unit))}], freq='1H')
#
#         trn_bwd, tst_bwd = trn_fwd, tst_fwd
#
#     else:
#         # forward
#         idx_trn_fwd_start, idx_trn_fwd_end = idx-len_unit-len_train+pred_len+1, idx-len_unit+pred_len
#         idx_tst_fwd_start, idx_tst_fwd_end = idx-len_unit+pred_len+1, idx+pred_len
#
#         trn_fwd = ListDataset([{'start': df.index[idx_trn_fwd_start],
#                                 'target': df['values'][df.index[idx_trn_fwd_start]:df.index[idx_trn_fwd_end]].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_trn_fwd_start]:df.index[idx_trn_fwd_end]].values.reshape((1,len_train))}], freq='1H')
#         tst_fwd = ListDataset([{'start': df.index[idx_tst_fwd_start],
#                                 'target': df['values'][df.index[idx_tst_fwd_start]:df.index[idx_tst_fwd_end]].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_tst_fwd_start]:df.index[idx_tst_fwd_end]].values.reshape((1,len_unit))}], freq='1H')
#
#         # backward
#         idx_trn_bwd_start, idx_trn_bwd_end = idx+4-pred_len+len_unit, idx+4-pred_len+len_unit+len_train-1
#         idx_tst_bwd_start, idx_tst_bwd_end = idx+4-pred_len, idx+4-pred_len+len_unit-1
#
#         trn_bwd = ListDataset([{'start': df.index[idx_trn_bwd_start],
#                                 'target': df['values'][df.index[idx_trn_bwd_start]:df.index[idx_trn_bwd_end]][::-1].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_trn_bwd_start]:df.index[idx_trn_bwd_end]][::-1].values.reshape((1,len_train))}], freq='1H')
#         tst_bwd = ListDataset([{'start': df.index[idx_tst_bwd_start],
#                                 'target': df['values'][df.index[idx_tst_bwd_start]:df.index[idx_tst_bwd_end]][::-1].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_tst_bwd_start]:df.index[idx_tst_bwd_end]][::-1].values.reshape((1,len_unit))}], freq='1H')
#
#     return trn_fwd, tst_fwd, trn_bwd, tst_bwd
#
#
# def model_deepar_test(len_unit, feature=True, epochs=10):
#     estimator = DeepAREstimator(
#         freq='1H',
#         prediction_length=4,
#         context_length=len_unit-4,
#         num_layers=2,
#         num_cells=40,
#         cell_type='lstm',
#         dropout_rate=0.1,
#         # use_feat_static_cat=feature,
#         # cardinality=[[672]],
#         use_feat_dynamic_real=feature,
#         # embedding_dimension=20,
#         scaling=True,
#         lags_seq=None,
#         time_features=None,
#         trainer=Trainer(ctx=mx.cpu(0), epochs=epochs, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=336, ),
#     )
#     return estimator
#
#
# def bidirec_dataset_deepar_test(df, idx, len_unit, len_train):
#     pred_len = 4
#
#     if idx < len_unit+len_train+1:
#         # backward
#         # idx_trn_bwd_start, idx_trn_bwd_end = idx+len_unit, idx+len_unit+len_train-1
#         # idx_tst_bwd_start, idx_tst_bwd_end = idx, idx+len_unit-1
#         idx_trn_bwd_start, idx_trn_bwd_end = idx+4-pred_len+len_unit, idx+4-pred_len+len_unit+len_train-1
#         idx_tst_bwd_start, idx_tst_bwd_end = idx+4-pred_len, idx+4-pred_len+len_unit-1
#
#         trn_bwd = ListDataset([{'start': df.index[idx_trn_bwd_start],
#                                 'target': df['values'][df.index[idx_trn_bwd_start]:df.index[idx_trn_bwd_end]][::-1].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_trn_bwd_start]:df.index[idx_trn_bwd_end]][::-1].values.reshape((1,len_train))}], freq='1H')
#         tst_bwd = ListDataset([{'start': df.index[idx_tst_bwd_start],
#                                 'target': df['values'][df.index[idx_tst_bwd_start]:df.index[idx_tst_bwd_end]][::-1].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_tst_bwd_start]:df.index[idx_tst_bwd_end]][::-1].values.reshape((1,len_unit))}], freq='1H')
#
#         trn_fwd, tst_fwd = trn_bwd, tst_bwd
#
#     elif idx > df.shape[0]-len_unit-len_train-1:
#         # forward
#         idx_trn_fwd_start, idx_trn_fwd_end = idx-len_unit-len_train+pred_len+1, idx-len_unit+pred_len
#         idx_tst_fwd_start, idx_tst_fwd_end = idx-len_unit+pred_len+1, idx+pred_len
#
#         trn_fwd = ListDataset([{'start': df.index[idx_trn_fwd_start],
#                                 'target': df['values'][df.index[idx_trn_fwd_start]:df.index[idx_trn_fwd_end]].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_trn_fwd_start]:df.index[idx_trn_fwd_end]].values.reshape((1,len_train))}], freq='1H')
#         tst_fwd = ListDataset([{'start': df.index[idx_tst_fwd_start],
#                                 'target': df['values'][df.index[idx_tst_fwd_start]:df.index[idx_tst_fwd_end]].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_tst_fwd_start]:df.index[idx_tst_fwd_end]].values.reshape((1,len_unit))}], freq='1H')
#
#         trn_bwd, tst_bwd = trn_fwd, tst_fwd
#
#     else:
#         # forward
#         idx_trn_fwd_start, idx_trn_fwd_end = idx-len_unit-len_train+pred_len+1, idx-len_unit+pred_len
#         idx_tst_fwd_start, idx_tst_fwd_end = idx-len_unit+pred_len+1, idx+pred_len
#
#         trn_fwd = ListDataset([{'start': df.index[idx_trn_fwd_start],
#                                 'target': df['values'][df.index[idx_trn_fwd_start]:df.index[idx_trn_fwd_end]].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_trn_fwd_start]:df.index[idx_trn_fwd_end]].values.reshape((1,len_train))}], freq='1H')
#         tst_fwd = ListDataset([{'start': df.index[idx_tst_fwd_start],
#                                 'target': df['values'][df.index[idx_tst_fwd_start]:df.index[idx_tst_fwd_end]].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_tst_fwd_start]:df.index[idx_tst_fwd_end]].values.reshape((1,len_unit))}], freq='1H')
#
#         # backward
#         idx_trn_bwd_start, idx_trn_bwd_end = idx+4-pred_len+len_unit, idx+4-pred_len+len_unit+len_train-1
#         idx_tst_bwd_start, idx_tst_bwd_end = idx+4-pred_len, idx+4-pred_len+len_unit-1
#
#         trn_bwd = ListDataset([{'start': df.index[idx_trn_bwd_start],
#                                 'target': df['values'][df.index[idx_trn_bwd_start]:df.index[idx_trn_bwd_end]][::-1].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_trn_bwd_start]:df.index[idx_trn_bwd_end]][::-1].values.reshape((1,len_train))}], freq='1H')
#         tst_bwd = ListDataset([{'start': df.index[idx_tst_bwd_start],
#                                 'target': df['values'][df.index[idx_tst_bwd_start]:df.index[idx_tst_bwd_end]][::-1].values,
#                                 'feat_dynamic_real': df['holiday'][
#                                                    df.index[idx_tst_bwd_start]:df.index[idx_tst_bwd_end]][::-1].values.reshape((1,len_unit))}], freq='1H')
#
#     return trn_fwd, tst_fwd, trn_bwd, tst_bwd
