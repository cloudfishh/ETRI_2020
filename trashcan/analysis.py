from matplotlib import pyplot as plt
from funcs import *


##################################################
# load data
list_apt = set_dir(0)                              # 0=gwangju; 1=naju
data_raw = load_household(list_apt)
data, nan_data = clear_head(data_raw)
data_first = data.iloc[:, 0]            # first house

# count nan length
nan_length_nonzero = count_nan_len(data_first)
nan_length_nonzero = np.delete(nan_length_nonzero, 0)   # first nan is fake
nan_len, nan_num = np.unique(nan_length_nonzero, return_counts=True)
nan_df = pd.DataFrame(nan_num, index=nan_len)

idx_acc = check_accumulation(data_first)
list_acc = pd.Series(np.zeros([len(data_first), ]), index=data_first.index)
for idx in idx_acc:
    list_acc[int(idx)] = True


##################################################
idx_acc_int = np.array([], dtype=int)
for i in idx_acc:
    idx = int(i)
    idx_acc_int = np.append(idx_acc_int, idx)

data_plot_point = np.empty((len(data_first),))
data_plot_point[:] = np.nan
data_plot_point = pd.Series(data_plot_point, index=data_first.index)
data_plot_point[idx_acc_int] = data_first[idx_acc_int]


idx1 = 13400
idx2 = 13500
plt.figure(figsize=(18, 4.5))
plt.plot(data_first[idx1:idx2])
plt.plot(data_plot_point[idx1:idx2], 'o', color='red')
plt.title('Gwangju - %s - %s' % (list_apt[0][0:10], data.columns[0]))
plt.xlabel('Time')
plt.ylabel('Power')
plt.xticks(range(0, idx2-idx1, 48))

