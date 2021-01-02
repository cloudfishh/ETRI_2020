from funcs import *

data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
calendar = load_calendar(2017, 2019)

n_good, n_bad = 0, 0
case_good, case_bad = [], []

h = 0
while (n_good<10)|(n_bad<10):
    test_house = data.columns[h]

    data_col = data[test_house]
    df = pd.DataFrame([], index=data_col.index)
    df['values'] = data_col.copy()
    df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)
    df['nan'] = chk_nan_bfaf(data_col)
    df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
    df['org_idx'] = np.arange(0, len(data_col))

    idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
    sample_list = list()
    for i in range(len(idx_list)):
        idx_target = idx_list[i]
        sample, _, _ = nearest_neighbor(data_col, df['nan'].copy(), idx_target, calendar)
        sample_list.append(sample)
    sample_near = pd.DataFrame(sample_list)

    mean_near = sample_near.mean(axis=1).values
    std_near = sample_near.std(axis=1).values
    true = df['values'][idx_list].values
    before = np.nan_to_num(df['values'][idx_list-1].values, 0)

    diff_near = true - mean_near
    diff_before = true - before

    rmse_near = np.sqrt(np.mean(diff_near ** 2))
    rmse_before = np.sqrt(np.mean(diff_before ** 2))

    if rmse_near < rmse_before:
        case_good.append(test_house)
        n_good += 1
    else:
        case_bad.append(test_house)
        n_bad += 1

    print(f'RMSE - {test_house} / # of good & bad - ({n_good}, {n_bad})')
    print(f'     k-NN : {rmse_near}')
    print(f'   before : {rmse_before}\n')

    h += 1