from funcs import *

# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
# test_house = '1dcb5feb'
test_house = 'ab7a314a'

data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
calendar = load_calendar(2017, 2019)

rmse_near, rmse_before = [], []
for test_house in data.columns:
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

    rmse_near.append(np.sqrt(np.mean(diff_near ** 2)))
    rmse_before.append(np.sqrt(np.mean(diff_before ** 2)))

    print(f'RMSE - {test_house}')
    print(f'     k-NN : {np.sqrt(np.mean(diff_near ** 2))}')
    print(f'   before : {np.sqrt(np.mean(diff_before ** 2))}\n')


rmse_near = np.nan_to_num(np.array(rmse_near), 0)
print('\n')
print('TOTAL RMSE')
print(f'     k-NN : {sum(rmse_near)/len(rmse_near)}')
print(f'   before : {sum(rmse_before)/len(rmse_before)}')