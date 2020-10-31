import pandas as pd
import numpy as np


def MAE(A,B):
    MAE_temp = 0
    for kk in range(0, len(A)):
        MAE_temp += abs(A[kk]-B[kk])/len(A)
    return MAE_temp

def RMSE(A,B):
    MAE_temp = 0
    for kk in range(0, len(A)):
        MAE_temp += ((A[kk]-B[kk])**2)/len(A)
    MAE_temp = np.sqrt(MAE_temp)
    return MAE_temp


nan_len = 3


df = pd.read_csv('201022_result.csv')

linear = df[{'injected', 'mask_inj'}].copy()
linear['imp_linear'] = linear['injected'].copy()
for idx in np.where((linear['mask_inj']==3)|(linear['mask_inj']==4))[0]:
    if linear['mask_inj'][idx] == 3:
        p = 0
        while pd.isna(linear['imp_linear'][idx-p]):
            p += 1
        q = 0
        while pd.isna(linear['imp_linear'][idx+nan_len+2+q]):
            q += 1
        linear['imp_linear'][idx:idx+nan_len+2] = linear['injected'][idx:idx+nan_len+2].interpolate(method='linear')

    else:   # 4
        p = 0
        while pd.isna(linear['imp_linear'][idx-1-p]):
            p += 1
        q = 0
        while pd.isna(linear['imp_linear'][idx+nan_len+2+q]):
            q += 1
        linear['imp_linear'][idx] = np.nan
        linear['imp_linear'][idx-1-p:idx+nan_len+2+q] = linear['imp_linear'][idx-1-p:idx+nan_len+2+q].interpolate(method='linear')
df['imp_linear'] = linear['imp_linear'].copy()


# without outlier -> mask_inj == 3
idx_3 = np.where(df['mask_detected']==3)[0]
mae_3 = np.empty([len(idx_3), 3])
for i in range(len(idx_3)):
    idx = idx_3[i]
    mae1 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    mae2 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_no-const'][idx+1:idx+nan_len+1].values)
    mae3 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear'][idx+1:idx+nan_len+1].values)
    temp = [mae1, mae2, mae3]
    mae_3[i, :] = temp
print(f'[joint w/, joint w/o, linear] = {np.nanmean(mae_3, axis=0)}')


# with outlier -> mask_inj == 4
idx_4 = np.where(df['mask_detected']==4)[0]
mae_4 = np.empty([len(idx_4), 3])
for i in range(len(idx_4)):
    idx = idx_4[i]
    mae1 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_const'][idx:idx+nan_len+1].values)
    mae2 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_no-const'][idx:idx+nan_len+1].values)
    mae3 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear'][idx:idx+nan_len+1].values)
    temp = [mae1, mae2, mae3]
    mae_4[i, :] = temp



# total
mae_total = [MAE(df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values, df['imp_const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values),
             MAE(df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values, df['imp_no-const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values),
             MAE(df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values, df['imp_linear'][(df['mask_inj']==2)|(df['mask_detected']==4)].values)]
# df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values
# df['imp_const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values
# df['imp_no-const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values
# df['imp_linear'][(df['mask_inj']==2)|(df['mask_detected']==4)].values

print(f'w/o outlier cases [joint w/, joint w/o, linear] = {np.nanmean(mae_3, axis=0)}')
print(f'w/  outlier cases [joint w/, joint w/o, linear] = {np.nanmean(mae_4, axis=0)}')
print(f'            total [joint w/, joint w/o, linear] = {mae_total}')
