from funcs import *
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm


test_house = '1dcb5feb'
f_fwd, f_bwd = 24, 24
nan_len = 3

# df = pd.read_csv('D:/202010_energies/201214_result_kmeans-added.csv', index_col=0)
# df_temp = df[df['house']==test_house]
# df_temp.to_csv('201230_result_temp.csv')

df = pd.read_csv('201230_result_temp.csv', index_col=0)
df = df.reset_index(drop=True)
idx_list = np.where((df['mask_inj']==3)|(df['mask_inj']==4))[0]
y_true = df['values'][idx_list].values

sample_deep = pd.read_csv(f'result/{test_house}/201017_detection_deepar_{test_house}.csv', index_col=0)
sample_deep = sample_deep.iloc[:, 1:]
sample_near = pd.read_csv(f'result/{test_house}/201017_detection_nearest_{test_house}.csv', index_col=0)

mean_deep = sample_deep.mean(axis=1).values
mean_near = sample_near.mean(axis=1).values
std_deep = sample_deep.std(axis=1).values
std_near = sample_near.std(axis=1).values

diff_deep = y_true-mean_deep
diff_near = y_true-mean_near

z_deep = diff_deep/std_deep
z_near = diff_near/std_near

z_deep[96] = 0

############################################################
x = np.arange(-12, 12, 0.01)
# bins = round(len(y_true)/2)

# estimation error 분포 - standard score
bins = round(len(y_true)/2)
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax.hist(z_deep, bins=bins, density=True, histtype='step', cumulative=False, label='DeepAR', linestyle='-', color='g', linewidth=1.5)
ax.hist(z_near, bins=bins, density=True, histtype='step', cumulative=False, label='k-NN', linestyle='-', color='r', linewidth=1.5)
plt.plot(x, norm.pdf(x, 0, 1), 'k--', linewidth=1.5, label='Normal distribution')
plt.xlim([-20, 20])
# plt.ylim([0, 1])
plt.xlabel('Standard score')
plt.ylabel('Empirical PDF')
plt.legend(['Normal distr.', 'DeepAR', 'k-NN'], loc='upper right', fontsize=13)
plt.tight_layout()
plt.savefig('201230_distr_z.png')


# estimation error 분포 - difference
bins = 30
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax.hist(diff_deep, bins=bins, density=True, histtype='step', cumulative=False, label='DeepAR', linestyle='-', color='g', linewidth=1.5)
ax.hist(diff_near, bins=bins, density=True, histtype='step', cumulative=False, label='k-NN', linestyle='-', color='r', linewidth=1.5)
plt.plot(x, norm.pdf(x, 0, 1), 'k--', linewidth=1.5, label='Normal distribution')
plt.xlim([-5, 5])
# plt.ylim([0, 1])
plt.xlabel('Estimation Error')
plt.ylabel('Empirical PDF')
plt.legend(['Normal distr.', 'DeepAR', 'k-NN'], loc='upper right', fontsize=13)
plt.tight_layout()
plt.savefig('201230_distr_diff.png')


# scatter plot ~ sigma & z-score
plt.figure()
plt.scatter(std_deep, z_deep, label='DeepAR', marker='o', facecolors='none', edgecolors='g')
plt.scatter(std_near, z_near, label='k-NN', marker='o', facecolors='none', edgecolors='r', alpha=0.5)
# plt.xlim([0, 6])
plt.ylim([-50, 100])
plt.xlabel(r'$\sigma$')
plt.ylabel('Standard Score')
plt.legend(['DeepAR', 'k-NN'])
plt.tight_layout()
plt.savefig('201230_scatter_z.png')

# scatter plot ~ sigma & difference
plt.figure()
plt.scatter(std_deep, diff_deep, label='DeepAR', marker='o', facecolors='none', edgecolors='g')
plt.scatter(std_near, diff_near, label='k-NN', marker='o', facecolors='none', edgecolors='r', alpha=0.5)
# plt.xlim([0, 6])
plt.ylim([-1, 2])
plt.xlabel(r'$\sigma$')
plt.ylabel('Estimation Error')
plt.legend(['DeepAR', 'k-NN'])
plt.tight_layout()
plt.savefig('201230_scatter_diff.png')


# scatter ~ correlation between errors
plt.figure(figsize=(6,6))
plt.scatter(diff_deep, diff_near, marker='o', facecolors='none', edgecolors='b')
plt.grid(alpha=0.3)
plt.xlabel('Error - DeepAR')
plt.ylabel('Error - k-NN')


y_before = np.nan_to_num(df['values'][idx_list-1].values, 0)
diff_before = y_true - y_before



print(f'DeepAR: {np.sqrt(np.mean(diff_deep ** 2))}')
print(f'  k-NN: {np.sqrt(np.mean(diff_near ** 2))}')
print(f'before: {np.sqrt(np.mean(diff_before ** 2))}')


plt.figure()
plt.scatter(diff_near, diff_before, marker='o', facecolors='none', edgecolors='b')