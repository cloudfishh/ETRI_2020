from funcs import *


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
    ma = np.ma.masked_array(target_data, mask=target_mask)
    return np.array(ma.tolist()), ma.mean(), ma.std()



from funcs import *
case_good = ['d6fd26b7', '08ca25cc', '11c1d3af', 'c5b2795b', 'f3c1791c',
             '82c7ecbb', 'ed590bab', '9bca65e1', '8fd0851d', '848b73db']
case_bad = ['410b92b1', '2c993de7', '1243cee2', 'b01fe23c', '28cf72fb',
            '6b5790f4', '1338c9d0', 'cbcc2b02', 'f69df1a6', 'aa71b3cb',
            '0583b3a6', '417407c0', '949c8301', '3b7d937e', 'a11b248c',
            'a625216e', 'b45961d4', 'c2aaf22f', '8d3d1b3a', '248dc456',
            '166fa275', 'b2adba75', 'fabefca4', 'b6d9a31f', '6055591b']


data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
calendar = load_calendar(2017, 2019)

test_house = case_bad[0]

rmse_near, rmse_near_rev, rmse_before = [], [], []
for test_house in case_bad:
    data_col = data[test_house]
    df = pd.DataFrame([], index=data_col.index)
    df['values'] = data_col.copy()
    df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)
    df['nan'] = chk_nan_bfaf(data_col)
    df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
    df['org_idx'] = np.arange(0, len(data_col))

    nan_mask = df['nan']
    idx_target = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0][101]



    idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
    sample_list = list()
    sample_list_rev = list()
    for i in range(len(idx_list)):
        idx_target = idx_list[i]
        sample, _, _ = nearest_neighbor(data_col, df['nan'].copy(), idx_target, calendar)
        sample_rev, _, _ = nearest_neighbor_rev(data_col, df['nan'].copy(), idx_target, calendar)
        sample_list.append(sample)
        sample_list_rev.append(sample_rev)
    sample_near = pd.DataFrame(sample_list)
    sample_near_rev = pd.DataFrame(sample_list_rev)

    mean_near = sample_near.mean(axis=1).values
    std_near = sample_near.std(axis=1).values
    mean_near_rev = sample_near_rev.mean(axis=1).values
    std_near_rev = sample_near_rev.std(axis=1).values
    true = df['values'][idx_list].values
    before = np.nan_to_num(df['values'][idx_list-1].values, 0)

    diff_near = true - mean_near
    diff_near_rev = true - mean_near_rev
    diff_before = true - before

    rmse_near.append(np.sqrt(np.mean(diff_near ** 2)))
    rmse_near_rev.append(np.sqrt(np.mean(diff_near_rev ** 2)))
    rmse_before.append(np.sqrt(np.mean(diff_before ** 2)))

    print(f'RMSE - {test_house}')
    print(f'     k-NN : {np.sqrt(np.mean(diff_near ** 2))}')
    print(f' k-NN rev : {np.sqrt(np.mean(diff_near_rev ** 2))}')
    print(f'   before : {np.sqrt(np.mean(diff_before ** 2))}\n')


rmse_near = np.nan_to_num(np.array(rmse_near), 0)
rmse_near_rev = np.nan_to_num(np.array(rmse_near_rev), 0)
print('\n')
print('TOTAL RMSE')
print(f'     k-NN : {sum(rmse_near)/len(rmse_near)}')
print(f' k-NN rev : {sum(rmse_near_rev)/len(rmse_near_rev)}')
print(f'   before : {sum(rmse_before)/len(rmse_before)}')