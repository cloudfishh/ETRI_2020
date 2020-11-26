"""
analyse the all results from whole households
 2020. 11. 15. Sun
SYPark
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


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
df = pd.read_csv('D:/202010_energies/201125_result_aodsc.csv', index_col=0)


############################################################
# load OWA
# filename = [f for f in os.listdir('D:/202010_energies/201124_compare') if f.endswith('.npz')]
# fn = filename[0]
# val = np.load('D:/202010_energies/201124_compare/'+fn, allow_pickle=True)['Value']  # key => home index

# house_list = np.unique(df['house'])
house_idx = np.unique(df['house'], return_index=True)[1]
house_list = [df['house'][idx] for idx in sorted(house_idx)]

# house = house_list[0]
comp = np.zeros([df.shape[0], 6])
for h in range(len(house_list)):
    house = house_list[h]
    val = np.load(f'D:/202010_energies/201125_compare_owa/MARS_OWA_{house}.npz')['Value']
    comp[h*19896:(h+1)*19896, :] = val

# col 4=vanilla, 5=aod, 6=aodsc
df['owa'] = comp[:, 4]
df['owa_aod'] = comp[:, 5]
df['owa_aodsc'] = comp[:, 5]

df.to_csv('D:/202010_energies/20115_result_aodsc+owa.csv')


############################################################
# analyse results ~ accuracy ~ detection 결과에 따라 4가지로 나눠서 계산
col_list = ['values', 'mask_inj', 'mask_detected',
             'joint', 'joint_aod', 'joint_aodsc',
            'linear', 'linear_aod', 'linear_aodsc',
            'spline', 'spline_aod', 'spline_aodsc',
            'mars', 'mars_aod', 'mars_aodsc',
            'owa', 'owa_aod', 'owa_aodsc']
case_list = col_list[3:]
num_case = len(case_list)

df_cut = df[(df['mask_inj']==2)|(df['mask_inj']==3)|(df['mask_inj']==4)][col_list].dropna(axis=0)


idx_33 = np.where((df_cut['mask_inj']==3)&(df_cut['mask_detected']==3))[0]
obs_33 = np.array([df_cut['values'][idx+1:idx+nan_len+1] for idx in idx_33]).reshape([len(idx_33)*5, ])
mae_33 = np.empty([num_case, ])
for i in range(num_case):
    prd_33 = np.array([df_cut[case_list[i]][idx+1:idx+nan_len+1] for idx in idx_33]).reshape([len(idx_33)*5, ])
    mae_33[i] = MAE(obs_33, prd_33)

idx_43 = np.where((df_cut['mask_inj']==4)&(df_cut['mask_detected']==3))[0]
obs_43 = np.array([df_cut['values'][idx+1:idx+nan_len+1] for idx in idx_43]).reshape([len(idx_43)*5, ])
mae_43 = np.empty([num_case, ])
for i in range(num_case):
    prd_43 = np.array([df_cut[case_list[i]][idx+1:idx+nan_len+1] for idx in idx_43]).reshape([len(idx_43)*5, ])
    mae_43[i] = MAE(obs_43, prd_43)

idx_34 = np.where((df_cut['mask_inj']==3)&(df_cut['mask_detected']==4))[0]
obs_34 = np.array([df_cut['values'][idx:idx+nan_len+1] for idx in idx_34]).reshape([len(idx_34)*6, ])
mae_34 = np.empty([num_case, ])
for i in range(num_case):
    prd_34 = np.array([df_cut[case_list[i]][idx:idx+nan_len+1] for idx in idx_34]).reshape([len(idx_34)*6, ])
    mae_34[i] = MAE(obs_34, prd_34)

idx_44 = np.where((df_cut['mask_inj']==4)&(df_cut['mask_detected']==4))[0]
obs_44 = np.array([df_cut['values'][idx:idx+nan_len+1] for idx in idx_44]).reshape([len(idx_44)*6, ])
mae_44 = np.empty([num_case, ])
for i in range(num_case):
    prd_44 = np.array([df_cut[case_list[i]][idx:idx+nan_len+1] for idx in idx_44]).reshape([len(idx_44)*6, ])
    mae_44[i] = MAE(obs_44, prd_44)

print('* detected nor.')
print(f' True negative {case_list}\n'
      f'             = {mae_33}')
print(f'False negative {case_list}\n'
      f'             = {mae_43}')
print('* detected acc. ')
print(f'False positive {case_list}\n'
      f'             = {mae_34}')
print(f' True positive {case_list}\n'
      f'             = {mae_44}')

print('* total')
print(f'{case_list}\n')
print(f'= [{MAE(df_cut["values"].values, df_cut["imp_const"].values), MAE(df_cut["values"].values, df_cut["imp_no-const"].values)}, ', end='')
print(f'{MAE(df_cut["values"].values, df_cut["imp_linear_const"].values)}, {MAE(df_cut["values"].values, df_cut["imp_linear_no-const"].values)}, ', end='')
print(f'{MAE(df_cut["values"].values, df_cut["imp_spline_const"].values)}, {MAE(df_cut["values"].values, df_cut["imp_spline_no-const"].values)}, ', end='')
print(f'{MAE(df_cut["values"].values, df_cut["imp_mars_const"].values)}, {MAE(df_cut["values"].values, df_cut["imp_mars_no-const"].values)}, ', end='')
print(f'{MAE(df_cut["values"].values, df_cut["imp_owa_const"].values)}, {MAE(df_cut["values"].values, df_cut["imp_owa_no-const"].values)}]', end='')


############################################################
# analyse results ~ accuracy ~ 통째로 모아서 계산 (시퀀스 하나로 만듦)
col_list = ['values', 'mask_inj', 'mask_detected',
             'joint', 'joint_aod', 'joint_aodsc',
            'linear', 'linear_aod', 'linear_aodsc',
            'spline', 'spline_aod', 'spline_aodsc',
            'mars', 'mars_aod', 'mars_aodsc',
            'owa', 'owa_aod', 'owa_aodsc']
case_list = col_list[3:]
num_case = len(case_list)

idx_3 = np.where(df['mask_detected']==3)[0]
obs_3 = np.array([df['values'][idx+1:idx+nan_len+1] for idx in idx_3]).reshape([len(idx_3)*5,])
mae_3 = np.empty([num_case, ])
for i in range(num_case):
    prd_3 = np.array([df[col_list[i]][idx+1:idx+nan_len+1] for idx in idx_3]).reshape([len(idx_3)*5,])
    mae_3[i] = MAE(obs_3, np.nan_to_num(prd_3))

idx_4 = np.where(df['mask_detected']==4)[0]
obs_4 = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_4]).reshape([len(idx_4)*6,])
mae_4 = np.empty([num_case, ])
for i in range(num_case):
    prd_4 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_4]).reshape([len(idx_4)*6,])
    mae_4[i] = MAE(obs_4, np.nan_to_num(prd_4))

idx_tot = np.where((df['mask_detected']==3)|(df['mask_detected'] == 4))[0]
obs_tot = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_tot]).reshape([len(idx_tot)*6, ])
mae_tot = np.empty([num_case, ])
for i in range(num_case):
    prd_34 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_tot]).reshape([len(idx_tot)*6, ])
    mae_tot[i] = MAE(obs_tot, np.nan_to_num(prd_34))

print(f'w/o outlier cases {case_list}\n'
      f'        = {mae_3}')
print(f'w/  outlier cases {case_list}\n'
      f'        = {mae_4}')
print(f'            total {case_list}\n'
      f'        = {mae_tot}')
