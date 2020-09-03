from matplotlib import pyplot as plt
from funcs import *


loc, apt, house = 3, 0, 96

##################################################
# load data
loc_list = ['Gwangju', 'Naju', 'Daejeon', 'Seoul', 'Incheon']
list_apt = set_dir(loc)                              # 0=gwangju; 1=naju
data_raw = load_household(list_apt, apt)
data, nan_data = clear_head(data_raw)
data_col = data.iloc[:, house]
calendar = load_calendar(2017, 2018)

injected, inj_mark = inject_nan_acc(data_col)


idx_acc = check_accumulation(injected, calendar)


##################################################
idx_acc = idx_acc.astype('int')

data_plot_point = np.empty((len(data_col),))
data_plot_point[:] = np.nan
data_plot_point = pd.Series(data_plot_point, index=data_col.index)
data_plot_point[idx_acc] = injected[idx_acc]

inj_idx_acc = np.where(inj_mark.values == 4)[0]
inj_plot_point = np.empty((len(data_col),))
inj_plot_point[:] = np.nan
inj_plot_point = pd.Series(inj_plot_point, index=data_col.index)
inj_plot_point[inj_idx_acc] = injected[inj_idx_acc]


##################################################
plt.rcParams.update({'font.size': 15})

plt.figure(figsize=[15, 5])

plt.plot(data_col.values, linewidth=0.7)
plt.plot(injected.values)
plt.plot(inj_plot_point.values, 'o', color='black', markersize=4)
plt.plot(data_plot_point.values, 'x', color='forestgreen', markersize=6)
plt.legend(['raw data', 'injected data', 'injected acc.', 'detection'], fontsize=14)

# plt.xticks(ticks=[i for i in range(0, len(injected), 2400)],
#            labels=[data_col.index[i][2:10] for i in range(0, len(injected), 2400)],
#            fontsize=14)

plt.title('%s %s - %s' % (loc_list[loc], list_apt[apt][0:10], data.columns[house]))
plt.xlabel('Time')
plt.ylabel('Power')
plt.gcf().subplots_adjust(bottom=0.15)
# plt.xticks(range(0, idx2-idx1, 48))


##################################################
plt.figure(figsize=[7.5, 5])

plt.plot(data_col.values, linewidth=0.7)
plt.plot(injected.values)
plt.plot(inj_plot_point.values, 'o', color='black', markersize=4)
plt.plot(data_plot_point.values, 'x', color='forestgreen', markersize=6)
plt.legend(['raw data', 'injected data', 'injected acc.', 'detection'], fontsize=14)

plt.xticks(ticks=[i for i in range(0, len(injected), 48)],
           labels=[data_col.index[i][2:10] for i in range(0, len(injected), 48)],
           fontsize=14)
# plt.grid()

plt.title('%s %s - %s' % (loc_list[loc], list_apt[apt][0:10], data.columns[house]))
plt.xlabel('Time')
plt.ylabel('Power')
plt.xlim([14520, 14690])
plt.gcf().subplots_adjust(bottom=0.15)

