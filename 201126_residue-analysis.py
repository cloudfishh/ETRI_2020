"""
Normalized residue analysis

2020. 11. 26. Thu
Soyeong Park
"""
##############################
from funcs import *
from matplotlib import pyplot as plt
from scipy.stats import norm


##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 5


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
calendar = load_calendar(2017, 2019)

z = []
for test_house in data.columns[:10]:
    data_col = data[test_house]
    df = pd.DataFrame([], index=data_col.index)
    df['values'] = data_col
    df['nan'] = chk_nan_bfaf(data_col)
    df['injected'], df['mask_inj'] = inject_nan_acc_nanlen(data_col, n_len=nan_len, p_nan=1, p_acc=0.25)
    df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
    df['org_idx'] = np.arange(0, len(data_col))

    idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]


    # ##############################
    # # 3. accumulation detection
    # # save the nearest neighbor samples
    # sample_list, mean_list, std_list = list(), list(), list()
    # for i in range(len(idx_list)):
    #     idx_target = idx_list[i]
    #     sample, m, s = nearest_neighbor(data_col, df['nan'].copy(), idx_target, calendar)
    #     sample_list.append(sample)
    #     mean_list.append(m)
    #     std_list.append(s)
    # smlr_sample = pd.DataFrame(sample_list)
    # smlr_sample.to_csv(f'result_201113/201027_{test_house}_nan{nan_len}_nearest.csv')


    ##############################
    smlr_sample = pd.read_csv(f'D:/2020_ETRI/result_201115_total-nearest/201027_{test_house}_nan{nan_len}_nearest.csv', index_col=0)
    val = df['values'][idx_list].values
    m = smlr_sample.mean(axis=1)
    s = smlr_sample.std(axis=1)
    z.extend(((val-m)/s).values.tolist())

z = np.array(z)


z = np.load('D:/202010_energies/normalized_residue_analysis.npy')

nor = np.random.normal(0, 1, size=100000)
sigma, mu, bins = 1, 0, 800

x = np.arange(-12, 12, 0.01)
y  = np.exp(-(x ** 2))
y /= (0.01 * y).sum()
cy = np.cumsum(0.01 * y)

fig, ax = plt.subplots(figsize=(6,4), dpi=200)
ax.hist(z, bins=bins, density=True, histtype='step', cumulative=True, label='Result from nearest neighbor', linestyle='--', color='r')
plt.plot(x, norm.cdf(x, 0, 1), 'k--', linewidth=1.5, label='Normal distribution')
# ax.plot(x, cy, 'k--', linewidth=1.5, label='Normal distribution')
plt.xlim([-6, 6])
plt.ylim([0, 1])
plt.xlabel('Standard score')
plt.ylabel('Empirical CDF')
plt.legend(loc='lower right')


fig, ax = plt.subplots(figsize=(6,4), dpi=200)
ax.hist(z, bins=bins, density=True, histtype='step', cumulative=False, label='Result from nearest neighbor', linestyle='--', color='r', linewidth=1.5)
plt.plot(x, norm.pdf(x, 0, 1), 'k--', linewidth=1.5, label='Normal distribution')
# ax.plot(x, cy, 'k--', linewidth=1.5, label='Normal distribution')
plt.xlim([-6, 6])
plt.ylim([0, 0.5])
plt.xlabel('Standard score')
plt.ylabel('Empirical PDF')
plt.legend(loc='lower right')
