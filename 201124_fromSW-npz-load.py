import numpy as np
import pandas as pd


def MAE(A, B):
    MAE_temp = 0
    for kk in range(0, len(A)):
        MAE_temp += abs(A[kk]-B[kk])/len(A)
    return MAE_temp


def RMSE(A, B):
    MAE_temp = 0
    for kk in range(0, len(A)):
        MAE_temp += ((A[kk]-B[kk])**2)/len(A)
    MAE_temp = np.sqrt(MAE_temp)
    return MAE_temp


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))



############################################################
# load results
nan_len = 5
method = 'nearest'
# method = 'similar'
df = eval(f'pd.read_csv("D:/2020_ETRI/201115_result_{method}_final.csv", index_col=0)')

# house_list = np.unique(df['house'])
house_idx = np.unique(df['house'], return_index=True)[1]
house_list = [df['house'][idx] for idx in sorted(house_idx)]


############################################################
# load comparison results
# filename = [f for f in os.listdir('D:/202010_energies/201124_compare') if f.endswith('.npz')]
# fn = filename[0]
# val = np.load('D:/202010_energies/201124_compare/'+fn, allow_pickle=True)['Value']  # key => home index

# house = house_list[0]
comp = np.zeros([df.shape[0], 6])
for h in range(len(house_list)):
    house = house_list[h]
    val = np.load(f'D:/202010_energies/201124_compare/Result_{house}.npz')['Value']
    comp[h*19896:(h+1)*19896, :] = val

# col 2=MARS w/o, 3=MARS w/, 4=OWA w/o, 5=OWA w/
df['imp_mars_no-const'] = comp[:, 2]
df['imp_mars_const'] = comp[:, 3]
df['imp_owa_no-const'] = comp[:, 4]
df['imp_owa_const'] = comp[:, 5]

df.to_csv('D:/202010_energies/201124_compare.csv')
