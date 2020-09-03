from matplotlib import pyplot as plt
from funcs import *


##################################################
# load data
# loc_list = ['Gwangju', 'Naju', 'Daejeon', 'Seoul', 'Incheon']

data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col_raw = data['68181c16']
calendar = load_calendar(2017, 2019)

data_col, inj_mask = inject_nan_acc(data_col_raw, p_nan=1, p_acc=0.5)

idx_list_result, idx_dict = check_accumulation(data_col, calendar)


idx_acc = idx_list_result.copy()
injected = data_col.copy()
inj_mark = inj_mask.copy()


##################################################
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
plt.rcParams.update({'font.size': 13})

plt.figure(figsize=[15, 5])

plt.plot(data_col_raw.values, linewidth=2)
plt.plot(injected.values, linewidth=0.7)
plt.plot(plot_point_inj.values, 'o', color='black', markersize=4)
plt.plot(plot_point_detect.values, 'x', color='forestgreen', markersize=6)
plt.legend(['raw data', 'injected data', 'injected acc.', 'detection'], fontsize=14)

plt.xticks(ticks=[i for i in range(0, len(injected), 2400)],
           labels=[data_col.index[i][2:10] for i in range(0, len(injected), 2400)],
           fontsize=14)
plt.grid([i for i in range(0, len(injected), 2400)])
# plt.title('%s %s - %s' % (loc_list[loc], list_apt[apt][0:10], data.columns[house]))
plt.xlabel('Time')
plt.ylabel('Power')
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
# plt.xticks(range(0, idx2-idx1, 48))


##################################################
plt.figure(figsize=[7.5, 5])

plt.plot(data_col_raw.values, linewidth=4)
plt.plot(injected.values, linewidth=2)
plt.plot(plot_point_inj.values, 'o', color='black', markersize=4)
plt.plot(plot_point_detect.values, 'x', color='forestgreen', markersize=6)
plt.legend(['raw data', 'injected data', 'injected acc.', 'detection'], fontsize=14)

plt.xticks(ticks=[i for i in range(0, len(injected), 48)],
           labels=[data_col.index[i][2:10] for i in range(0, len(injected), 48)],
           fontsize=12)
plt.grid([i for i in range(0, len(injected), 48)])
# plt.grid()

# plt.title('%s %s - %s' % (loc_list[loc], list_apt[apt][0:10], data.columns[house]))
plt.xlabel('Time')
plt.ylabel('Power')
plt.xlim([400, 700])
plt.ylim([-0.25, 3.8])
plt.gcf().subplots_adjust(bottom=0.15)


##################################################
a = inj_mark[inj_mark == 3].index.to_list()
b = data_col[idx_list_result].index.to_list()
c_result = []
for i in range(len(a)):
    c_result.append(1) if a[i] in b else c_result.append(0)
