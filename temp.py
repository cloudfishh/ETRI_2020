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

