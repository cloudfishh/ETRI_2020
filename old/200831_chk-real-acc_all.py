from funcs import *
from matplotlib import pyplot as plt


loc_list = ['Gwangju', 'Naju', 'Daejeon', 'Seoul', 'Incheon']
for l in range(5):
    loc = loc_list[l]
    list_apt = set_dir(l)

    for apt in range(len(list_apt)):
        data_raw = load_household(list_apt, idx=apt)
        data, nan_data = clear_head(data_raw)

        list_ratio = pd.DataFrame([], index=data_raw.columns, columns=['total', 'checked', 'ratio'])
        for test_house in data_raw.columns:
            # test_house = random.sample(list(data_raw.columns), k=1)[0]
            data_col = data[test_house]

            idx_nan = np.where(np.isnan(data_col.values) == True)[0]

            df = pd.DataFrame([])
            df['values'] = data_col.copy()
            df['nan'] = np.isnan(data_col.values)

            num = 0
            num_col, val_col = [], []
            for i in range(df.shape[0]):
                if df['nan'][i] == True:
                    num += 1
                elif (df['nan'][i] == False) & (num != 0):
                    val_col.append(df['values'][i-num-1])
                    num_col.append(num)
                    num = 0

            mask_valid = np.invert(chk_nan_bfaf(data_col).values.astype('bool').reshape(data_col.shape))
            val_valid = data_col.values[mask_valid]
            num_valid = np.zeros(val_valid.shape)

            val_col = np.array(val_col)
            num_col = np.array(num_col)

            val_col = np.append(val_col, val_valid)
            num_col = np.append(num_col, num_valid)

            plot_df = pd.DataFrame([])
            plot_df['num'] = num_col
            plot_df['val'] = val_col
            plot_df = plot_df.sort_values(['num', 'val'])

            plot_mean = pd.DataFrame([])
            plot_mean['num'] = np.arange(26)
            plot_mean['val'] = (np.arange(26)+1) * plot_df['val'][plot_df['num'] == 0].values.mean()

            plot_hor = pd.DataFrame([])
            plot_hor['num'] = np.arange(26)
            plot_hor['val'] = np.ones(26) * plot_df['val'][plot_df['num'] == 0].values.mean()


            ##############################
            plt.figure(figsize=(7.5,5))
            plt.plot(plot_df['num'], plot_df['val'], '.')
            plt.plot(plot_mean['num'], plot_mean['val'], color='tomato', linewidth=1)
            plt.plot(plot_hor['num'], plot_hor['val'], color='seagreen', linewidth=1)
            plt.fill_between(plot_hor['num'],
                             # plot_mean['val'] - plot_df['val'][plot_df['num'] == 0].values.std()*3,
                             plot_hor['val'] - plot_df['val'][plot_df['num'] == 0].values.std()*3,
                             plot_hor['val'] + plot_df['val'][plot_df['num'] == 0].values.std()*3, color='seagreen', alpha=0.2)
            plt.xlim([-1, 25])
            # plt.ylim([-0.2, 3.7])
            # plt.xticks(ticks=[i for i in range(0, 25)])
            plt.xlabel('length of NaNs')
            plt.ylabel('Value (Power [kW])')
            plt.title(f'{test_house}')
            plt.tight_layout()
            plt.savefig(f'D:/PycharmProjects/ETRI_2020/chkrealacc_{loc}_apt{apt}_{test_house}.png')
            plt.close()


            ##############################
            # counting accumulation candidates
            count = dict()
            for i in range(plot_df.shape[0]):
                if plot_df['val'][i] > plot_hor['val'][0] + plot_df['val'][plot_df['num'] == 0].values.std()*3:
                    try:
                        count[plot_df['num'][i]] += 1
                    except KeyError:
                        count[plot_df['num'][i]] = 1

            for keys, values in sorted(count.items()):
                if keys != 0:
                    print(f'{keys}: {values}')

            tot = sum(plot_df["num"] != 0)
            print(f'{loc} {test_house} total # of values: {sum(plot_df["num"] != 0)}')
            try:
                checked = sum(count.values())-count[0]
                print(f'{loc} {test_house} # of checked values: {sum(count.values())-count[0]}\n')
            except KeyError:
                checked = sum(count.values())
                print(f'{loc} {test_house} # of checked values: {sum(count.values())}\n')

            list_ratio.loc[test_house] = [tot, checked, checked/tot]
        list_ratio.to_csv(f'realacc_ratio_{loc}_{apt}.csv')