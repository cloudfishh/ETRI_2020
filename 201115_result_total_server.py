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


############################################################
# 데이터 일단 다 붙여
# method = 'nearest'
method = 'similar'

result_filelist = [f for f in os.listdir(f'/home/ubuntu/Documents/sypark/2020_ETRI/result_201115_total-{method}') if f.endswith('_result.csv')]
df_concat = pd.DataFrame([])
for f in result_filelist:
    df_load = pd.read_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/result_201115_total-{method}/{f}')
    df_load.insert(0, 'house', np.empty([df_load.shape[0], ]))
    df_load['house'] = f[7:15]
    df_concat = pd.concat([df_concat, df_load], axis=0)

df = df_concat.copy().reset_index(drop=True)
df.to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/201115_result_{method}_justconcat.csv')

if sum(np.isin(os.listdir('/home/ubuntu/Documents/sypark/2020_ETRI'), f'201115_result_{method}_justconcat.csv'))==0:
    df.to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/201115_result_{method}_justconcat.csv')
else:
    print('** justconcat.csv SAVED **')


############################################################
# LINEAR INTERPOLATION
# df = pd.read_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/201115_result_{method}_justconcat.csv')
nan_len = 5

linear = df[{'values', 'injected', 'mask_detected'}].copy()
linear['imp_linear_const'] = linear['injected'].copy()
linear['imp_linear_no-const'] = linear['injected'].copy()

# injection 바로 앞에 또 injection이 있는 경우에 이어서 하면 안 되잖아
for idx in np.where((linear['mask_detected']==3)|(linear['mask_detected']==4))[0]:
    temp_nocon, temp_const = linear['values'].copy(), linear['values'].copy()
    temp_nocon[:idx], temp_const[:idx] = linear['imp_linear_no-const'][:idx], linear['imp_linear_const'][:idx]
    temp_nocon[idx:idx+nan_len+2], temp_const[idx:idx+nan_len+2] = linear['injected'][idx:idx+nan_len+2], linear['injected'][idx:idx+nan_len+2]

    # w/o const
    p, q = 0, 0
    while pd.isna(temp_nocon[idx-p]):
        p += 1
    while pd.isna(temp_nocon[idx+nan_len+2+q]):
        q += 1
    linear['imp_linear_no-const'][idx:idx+nan_len+2] = temp_nocon[idx-p:idx+nan_len+2+q].interpolate(method='linear')

    # w/ const
    if linear['mask_detected'][idx] == 3:
        p, q = 0, 0
        while pd.isna(temp_const[idx-p]):
            p += 1
        while pd.isna(temp_const[idx+nan_len+2+q]):
            q += 1
        linear['imp_linear_const'][idx:idx+nan_len+2] = temp_const[idx-p:idx+nan_len+2+q].interpolate(method='linear')

    else:   # 4
        p, q = 0, 0
        while pd.isna(temp_const[idx-1-p]):
            p += 1
        while pd.isna(temp_const[idx+nan_len+2+q]):
            q += 1
        s = temp_const[idx]
        temp_const[idx] = np.nan
        li_temp = temp_const[idx-1-p:idx+nan_len+2+q].interpolate(method='linear').loc[idx:idx+nan_len]
        linear['imp_linear_const'][idx:idx+nan_len+1] = li_temp*(s/sum(li_temp.values))
    print(f'{idx} ', end='')

df['imp_linear_const'] = linear['imp_linear_const'].copy()
df['imp_linear_no-const'] = linear['imp_linear_no-const'].copy()

df.to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/201115_result_{method}_final.csv')

if sum(np.isin(os.listdir('/home/ubuntu/Documents/sypark/2020_ETRI'), f'201115_result_{method}_final.csv'))==0:
    df.to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/201115_result_{method}_final.csv')
else:
    print('** final.csv (linear) SAVED **')


############################################################
# LINEAR INTERPOLATION ~ SPLINE
# df = pd.read_csv(f'D:/2020_ETRI/201115_result_{method}_final.csv', index_col=0)
nan_len = 5

spline = df[{'values', 'injected', 'mask_detected'}].copy().reset_index(drop=True)
spline['imp_spline_const'] = spline['injected'].copy()
spline['imp_spline_no-const'] = spline['injected'].copy()

print('***** SPLINE INTERPOLATION *****')
# injection 바로 앞에 또 injection이 있는 경우에 이어서 하면 안 되잖아
for idx in np.where((spline['mask_detected']==3)|(spline['mask_detected']==4))[0]:
    temp_nocon, temp_const = spline['values'].copy(), spline['values'].copy()
    temp_nocon[:idx], temp_const[:idx] = spline['values'][:idx], spline['values'][:idx]
    # temp_nocon[:idx], temp_const[:idx] = spline['imp_spline_no-const'][:idx], spline['imp_spline_const'][:idx]
    temp_nocon[idx:idx+nan_len+2], temp_const[idx:idx+nan_len+2] = spline['injected'][idx:idx+nan_len+2], spline['injected'][idx:idx+nan_len+2]

    # w/o const
    p, q = 24, 24
    # while pd.isna(temp_nocon[idx-p]):
    #     p += 1
    # while pd.isna(temp_nocon[idx+nan_len+2+q]):
    #     q += 1
    spline['imp_spline_no-const'][idx+1:idx+nan_len+1] = temp_nocon[idx-p:idx+nan_len+2+q].interpolate(method='spline', order=3).loc[idx+1:idx+nan_len]

    # w/ const
    if spline['mask_detected'][idx] == 3:
        p, q = 24, 24
        # while pd.isna(temp_const[idx-p]):
        #     p += 1
        # while pd.isna(temp_const[idx+nan_len+2+q]):
        #     q += 1
        spline['imp_spline_const'][idx+1:idx+nan_len+1] = temp_const[idx-p:idx+nan_len+2+q].interpolate(method='spline', order=3).loc[idx+1:idx+nan_len]

    else:   # 4
        p, q = 24, 24
        # while pd.isna(temp_const[idx-1-p]):
        #     p += 1
        # while pd.isna(temp_const[idx+nan_len+2+q]):
        #     q += 1
        s = temp_const[idx]
        temp_const[idx] = np.nan
        li_temp = temp_const[idx-1-p:idx+nan_len+2+q].interpolate(method='spline', order=3).loc[idx:idx+nan_len]
        spline['imp_spline_const'][idx:idx+nan_len+1] = li_temp*(s/sum(li_temp.values))
    print(f'{idx} ', end='')

df['imp_spline_const'] = spline['imp_spline_const'].values.copy()
df['imp_spline_no-const'] = spline['imp_spline_no-const'].values.copy()

os.remove(f'/home/ubuntu/Documents/sypark/2020_ETRI/201115_result_{method}_final.csv')
df.to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/201115_result_{method}_final.csv')

if sum(np.isin(os.listdir('/home/ubuntu/Documents/sypark/2020_ETRI'), f'201115_result_{method}_final.csv'))==0:
    df.to_csv(f'/home/ubuntu/Documents/sypark/2020_ETRI/201115_result_{method}_final.csv')
else:
    print('** final.csv (spline) SAVED **')

