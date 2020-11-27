from funcs import *
import os
from matplotlib import pyplot as plt


##############################
# 0. parameter setting
f_fwd, f_bwd = 24, 24
nan_len = 5


##############################
# 1. load dataset
df = pd.read_csv('D:/202010_energies/201125_result_aodsc+owa.csv', index_col=0)

idx_detected_nor = np.where(df['mask_detected']==3)[0]
idx_detected_acc = np.where(df['mask_detected']==4)[0]
data_col = df['values'].copy()


##############################
# 4-3. SPLINE
print('**          - SPLINE start')
spline = df[{'values', 'injected', 'mask_detected'}].copy().reset_index(drop=True)
spline['spline'] = spline['injected'].copy()
spline['spline_aod'] = spline['injected'].copy()
spline['spline_aodsc'] = spline['injected'].copy()

count = 0
for idx in np.where((spline['mask_detected'] == 3) | (spline['mask_detected'] == 4))[0]:
    temp_nocon, temp_const = spline['values'].copy(), spline['values'].copy()
    temp_nocon[:idx], temp_const[:idx] = spline['values'][:idx], spline['values'][:idx]
    temp_nocon[idx:idx+nan_len+2], temp_const[idx:idx+nan_len+2] = spline['injected'][idx:idx+nan_len+2], spline['injected'][idx:idx+nan_len+2]
    # w/o const
    p, q = 24, 24
    spline['spline'][idx+1:idx+nan_len+1] = temp_nocon[idx-p:idx+nan_len+2+q].interpolate(method='spline', order=3).loc[idx+1:idx+nan_len]

    # w/ const
    if spline['mask_detected'][idx] == 3:
        p, q = 24, 24
        spline['spline_aod'][idx+1:idx+nan_len+1] = temp_const[idx-p:idx+nan_len+2+q].interpolate(method='spline', order=3).loc[idx+1:idx+nan_len]
        spline['spline_aodsc'][idx+1:idx+nan_len+1] = temp_const[idx-p:idx+nan_len+2+q].interpolate(method='spline', order=3).loc[idx+1:idx+nan_len]

    else:  # 4
        p, q = 24, 24
        s = temp_const[idx]
        temp_const[idx] = np.nan
        li_temp = temp_const[idx-1-p:idx+nan_len+2+q].interpolate(method='spline', order=3).loc[idx:idx+nan_len]
        spline['spline_aod'][idx:idx+nan_len+1] = li_temp
        spline['spline_aodsc'][idx:idx+nan_len+1] = li_temp*(s/sum(li_temp.values))

    count += 1
    if count % 100 == 0:
        print(f'{idx} ', end='')


df['spline'] = spline['spline'].values.copy()
df['spline_aod'] = spline['spline_aod'].values.copy()
df['spline_aodsc'] = spline['spline_aodsc'].values.copy()
print('**          - SPLINE finished')

df.to_csv('201125_result_aodsc+owa_spline-rev.csv')
