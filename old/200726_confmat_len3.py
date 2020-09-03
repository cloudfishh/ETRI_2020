from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from funcs import *


##################################################
# set variables
test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
# test_house = '68181c16'
# test_house = '1dcb5feb'

# sigma = 3
for sigma in range(3, 6):
    result_all_true = np.array([])
    result_all_pred = np.array([])
    for test_house in test_house_list:
        ##################################################
        # load data
        data_raw = load_labeled()
        data, nan_data = clear_head(data_raw)
        data_col_raw = data[test_house]
        calendar = load_calendar(2017, 2019)

        data_col, inj_mask = inject_nan_acc3(data_col_raw, p_nan=1, p_acc=0.5)

        idx_list_result, idx_dict = check_accumulation_injected(data_col, inj_mask, calendar, sigma=sigma)

        # candidate idx list
        idx_list_candidate = np.array([])
        candidate = inj_mask[(inj_mask == 3) | (inj_mask == 4)]
        for i in range(len(candidate)):
            idx_list_candidate = np.append(idx_list_candidate, inj_mask.index.get_loc(candidate.index[i]))

        ##################################################
        # calculate result - false positive & false negative
        # candidate idx list
        idx_list_candidate = np.array([])
        candidate = inj_mask[(inj_mask == 3) | (inj_mask == 4)]
        for i in range(len(candidate)):
            idx_list_candidate = np.append(idx_list_candidate, inj_mask.index.get_loc(candidate.index[i]))

        idx_list_acc = np.array([])
        acc = inj_mask[inj_mask == 3]
        for i in range(len(acc)):
            idx_list_acc = np.append(idx_list_acc, inj_mask.index.get_loc(acc.index[i]))

        idx_list_normal = np.array([])
        nor = inj_mask[inj_mask == 4]
        for i in range(len(nor)):
            idx_list_normal = np.append(idx_list_normal, inj_mask.index.get_loc(nor.index[i]))

        npnan = np.empty((len(idx_list_candidate),))
        npnan[:] = np.nan
        result_true = pd.DataFrame(np.array([idx_list_candidate, npnan]).transpose(), columns=['idx', 'result'])
        result_pred = pd.DataFrame(np.array([idx_list_candidate, npnan]).transpose(), columns=['idx', 'result'])

        for i in range(len(result_true)):
            if np.isin(result_true['idx'][i], idx_list_acc):
                result_true['result'][i] = 1
            else:
                result_true['result'][i] = 0

        for i in range(len(result_pred)):
            if np.isin(result_pred['idx'][i], idx_list_result):
                result_pred['result'][i] = 1
            else:
                result_pred['result'][i] = 0

        result_all_true = np.append(result_all_true, result_true['result'].values)
        result_all_pred = np.append(result_all_pred, result_pred['result'].values)

        result_cm = confusion_matrix(result_true['result'], result_pred['result'])

        # plt.rcParams.update({'font.size': 14})
        # plt.figure()
        # sns.heatmap(result_cm, annot=True, fmt='d', square=True,
        #             xticklabels=['normal', 'accumulation'], yticklabels=['normal', 'accumulation'])
        # plt.title(f'{test_house}, nan_length=3, {sigma}$\sigma$', fontsize=14)
        # plt.xlabel('Predicted label')
        # plt.ylabel('True label')
        # plt.savefig(f'D:/2020_ETRI/200727_fig/{test_house}_nan3_{sigma}s_cm.png')

        ##################################################
        idx_acc = idx_list_result.copy()
        injected = data_col.copy()
        inj_mark = inj_mask.copy()

        idx_acc = idx_acc.astype('int')

        plot_point_inj = np.empty((len(data_col),))
        plot_point_inj[:] = np.nan
        plot_point_inj = pd.Series(plot_point_inj, index=data_col.index)
        plot_point_inj[inj_mask == 3] = injected[inj_mask == 3]

        # inj_idx_acc = np.where(inj_mark.values == 3)[0]
        plot_point_detect = np.empty((len(data_col),))
        plot_point_detect[:] = np.nan
        plot_point_detect = pd.Series(plot_point_detect, index=data_col.index)
        idx_list_result = idx_list_result.astype('int')
        plot_point_detect[idx_list_result] = injected[idx_list_result]

        ##################################################
        # plt.rcParams.update({'font.size': 13})
        #
        # plt.figure(figsize=[15, 5])
        #
        # plt.plot(data_col_raw.values, linewidth=2)
        # plt.plot(injected.values, linewidth=0.7)
        # plt.plot(plot_point_inj.values, 'o', color='black', markersize=4)
        # plt.plot(plot_point_detect.values, 'x', color='forestgreen', markersize=6)
        # plt.legend(['raw data', 'injected data', 'injected acc.', 'detection'], fontsize=14)
        #
        # plt.xticks(ticks=[i for i in range(0, len(injected), 2400)],
        #            labels=[data_col.index[i][2:10] for i in range(0, len(injected), 2400)],
        #            fontsize=14)
        # plt.grid([i for i in range(0, len(injected), 2400)])
        # plt.title(f'{test_house}, nan_length=3, {sigma}$\sigma$', fontsize=14)
        # plt.xlabel('Time')
        # plt.ylabel('Power')
        # plt.tight_layout()
        # plt.gcf().subplots_adjust(bottom=0.15)
        # # plt.xticks(range(0, idx2-idx1, 48))
        # plt.savefig(f'D:/2020_ETRI/200727_fig/{test_house}_nan3_{sigma}s_graph.png')

        # ##################################################
        # plt.figure(figsize=[7.5, 5])
        #
        # plt.plot(data_col_raw.values, linewidth=4)
        # plt.plot(injected.values, linewidth=2)
        # plt.plot(plot_point_inj.values, 'o', color='black', markersize=4)
        # plt.plot(plot_point_detect.values, 'x', color='forestgreen', markersize=6)
        # plt.legend(['raw data', 'injected data', 'injected acc.', 'detection'], fontsize=14)
        #
        # plt.xticks(ticks=[i for i in range(0, len(injected), 48)],
        #            labels=[data_col.index[i][2:10] for i in range(0, len(injected), 48)],
        #
        # # plt.title('%s %s - %s' % (loc_list[loc], list_apt[apt][0:10], data.columns[house]))
        #            fontsize=12)
        # plt.grid([i for i in range(0, len(injected), 48)])
        # # plt.grid()
        # plt.xlabel('Time')
        # plt.ylabel('Power')
        # plt.xlim([400, 700])
        # plt.ylim([-0.25, 3.8])
        # plt.gcf().subplots_adjust(bottom=0.15)

    result_all_cm = confusion_matrix(result_all_true, result_all_pred)
    plt.rcParams.update({'font.size': 14})
    plt.figure()
    sns.heatmap(result_all_cm, annot=True, annot_kws={'size': 20}, fmt='d', square=True,
                cmap='Greys', linewidths=0.7, linecolor='black',
                xticklabels=['normal', 'accumulation'], yticklabels=['normal', 'accumulation'])
    plt.title(f'5 houses, nan_length=3, {sigma}$\sigma$', fontsize=14)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'D:/2020_ETRI/200804_monthly/5 houses_nan3_{sigma}s_cm.png')
    print(f'{sigma} COMPLETED')
