from funcs import *
from matplotlib import pyplot as plt


test_house = '68181c16'

data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
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

plot_df = pd.DataFrame([])
plot_df['num'] = num_col
plot_df['val'] = val_col
plot_df = plot_df.sort_values(['num', 'val'])


##############################
plt.figure(figsize=(7.5,5))
plt.plot(plot_df['num'], plot_df['val'], '.')
plt.xlim([0, 25])
plt.xticks(ticks=[i for i in range(0, 25)])
plt.xlabel('length of NaNs')
plt.ylabel('Value (Power [kW])')
plt.title(f'{test_house}')
plt.tight_layout()

# len_NaN=3 까지만 플롯
plt.figure()
plt.plot(plot_df['num'], plot_df['val'], '.')
plt.xlim([0.5, 3.5])
plt.xticks(ticks=[i for i in range(4)])
plt.xlabel('length of NaNs')
plt.ylabel('Value (Power [kW])')
plt.title(f'{test_house}')
plt.tight_layout()
