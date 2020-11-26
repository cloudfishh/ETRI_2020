from funcs import *

#
# test_house = '68181c16'
# f_fwd, f_bwd = 24, 24
# nan_len = 3
# sigma = 5
# data_raw = load_labeled()
# data, nan_data = clear_head(data_raw)
# data_col = data[test_house]
# calendar = load_calendar(2017, 2019)
#
# p_nan, p_acc = 1, 0.5

if __name__=='__main__':
    # loc_list = ['Gwangju', 'Naju', 'Daejeon', 'Seoul', 'Incheon']
    # for l in range(5):
    #     loc = loc_list[l]
    #     list_apt = set_dir(l)
    #     for apt in range(len(list_apt)):
    #         result = pd.read_csv(f'D:/PycharmProjects/ETRI_2020/csv/200907_realacc_z_{loc}_{apt}.csv', index_col=0)
    #         print(f'{loc}_{apt} total: {result["total"].mean():.5}')
    #         print(f'{loc}_{apt}  chk3: {result["chk_3"].mean():.5}')
    #         print(f'{loc}_{apt}  mean: {result["ratio3"].mean():.5}\n')
    file_dir = 'D:/2020_ETRI/NAB_data/NAB_data'
    direc = 'realAdExchange'

    csv_list = os.listdir(f'{file_dir}/data/{direc}')
    for c in range(len(csv_list)):
        data = pd.read_csv(f'{file_dir}/data/{direc}/{csv_list[c]}', index_col=0)
        freq = pd.to_datetime(data.index[1])-pd.to_datetime(data.index[0])
        print(f'{file_dir}/{direc}/{csv_list[c]} ~ shape:{data.shape}, freq:{freq}')



    from funcs import *

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

    ##############################
    # 2. injection
    df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)

    ##############################
    # 2.5. sepearte cases
    df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
    df['org_idx'] = np.arange(0, len(data_col))
    df_holi0 = df[df['holiday'] == 0]  # work, holi0
    df_holi1 = df[df['holiday'] == 1]  # non-work, holi1

    idx_list_holi0 = df_holi0['org_idx'][(df_holi0['mask_inj'] == 3) | (df_holi0['mask_inj'] == 4)]
    idx_list_holi1 = df_holi1['org_idx'][(df_holi1['mask_inj'] == 3) | (df_holi1['mask_inj'] == 4)]

    holi0 = pd.read_csv('201005_separate_holiday0_4weeks.csv', index_col=0)
    holi1 = pd.read_csv('201005_separate_holiday1_4weeks.csv', index_col=0)

    a0, a1 = pd.DataFrame(), pd.DataFrame()
    a0['idx'], a1['idx'] = idx_list_holi0, idx_list_holi1

    b0 = pd.concat([a0.reset_index(drop=True), holi0], axis=1, ignore_index=True)
    b1 = pd.concat([a1.reset_index(drop=True), holi1], axis=1, ignore_index=True)

    c = pd.concat([b0, b1])
    cc = c.sort_values(by=0)
    idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]

    cc.drop(columns=[0]).to_csv('result_deepar_separate_4weeks.csv')




############################################################
# analyse results ~ accuracy ~ detection 결과에 따라 4가지로 나눠서 계산
col_list = ['imp_const', 'imp_no-const',
            'imp_linear_const', 'imp_linear_no-const',
            'imp_spline_const', 'imp_spline_no-const']

idx_33 = np.where(df['mask_detected'] == 3)[0]
obs_33 = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_33]).reshape([len(idx_33)*6, ])
mae_nor = np.empty([6, ])
for i in range(6):
    prd_3 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_33]).reshape([len(idx_33)*6, ])
    mae_nor[i] = MAE(obs_33, np.nan_to_num(prd_3))

idx_34 = np.where((df['mask_detected']==3)|(df['mask_detected'] == 4))[0]
obs_34 = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_34]).reshape([len(idx_34)*6, ])
mae_tot = np.empty([6, ])
for i in range(6):
    prd_34 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_34]).reshape([len(idx_34)*6, ])
    mae_tot[i] = MAE(obs_34, np.nan_to_num(prd_34))

idx_43 = np.where(df['mask_detected'] == 4)[0]
obs_43 = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_43]).reshape([len(idx_43)*6, ])
mae_acc = np.empty([6, ])
for i in range(6):
    prd_4 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_43]).reshape([len(idx_43)*6, ])
    mae_acc[i] = MAE(obs_43, np.nan_to_num(prd_4))

idx_44 = np.where(df['mask_detected'] == 3)[0]
obs_44 = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_44]).reshape([len(idx_44)*6, ])
mae_nor = np.empty([6, ])
for i in range(6):
    prd_3 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_44]).reshape([len(idx_44)*6, ])
    mae_nor[i] = MAE(obs_44, np.nan_to_num(prd_3))


print(f'w/o outlier cases [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'        = {mae_nor}')
print(f'w/  outlier cases [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'        = {mae_acc}')
print(f'            total [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'        = {mae_tot}')