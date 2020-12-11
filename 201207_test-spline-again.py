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
    # interpolation 적용할 index의 injection만 남기고 나머지는 raw data
    temp_nocon, temp_const = spline['values'].copy(), spline['values'].copy()
    temp_nocon[:idx], temp_const[:idx] = spline['values'][:idx], spline['values'][:idx]
    temp_nocon[idx:idx+nan_len+2], temp_const[idx:idx+nan_len+2] = spline['injected'][idx:idx+nan_len+2], spline['injected'][idx:idx+nan_len+2]


    # make sequences for interpolation ~ nocon은 step 6, const는 step 7로 value 띄엄띄엄 남기기
    p, q = 24, 24
    temp_nocon_part = temp_nocon[idx-p:idx+nan_len+2+q].copy()
    for ii in range(temp_nocon_part.index[0], temp_nocon_part.index[-1]):
        if ii % (nan_len+1) != idx % (nan_len+1):
            temp_nocon_part[ii] = np.nan
    while sum(~pd.isna(temp_nocon_part)) < 4:
        p += nan_len+1
        q += nan_len+1
        temp_nocon_part = temp_nocon[idx-p:idx+nan_len+2+q].copy()
        for ii in range(temp_nocon_part.index[0], temp_nocon_part.index[-1]):
            if ii%(nan_len+1) != idx%(nan_len+1):
                temp_nocon_part[ii] = np.nan


    p, q = (nan_len+2)*4, (nan_len+2)*4
    temp_const_part = temp_const[idx-p-1:idx+nan_len+2+q].copy()
    for ii in range(temp_const_part.index[0], temp_const_part.index[-1]):
        if ii % (nan_len+2) != (idx-1) % (nan_len+2):
            temp_const_part[ii] = np.nan
    while sum(~pd.isna(temp_const_part)) < 4:
        p += nan_len+2
        q += nan_len+2
        temp_const_part = temp_const[idx-p-1:idx+nan_len+2+q].copy()
        for ii in range(temp_const_part.index[0], temp_const_part.index[-1]):
            if ii%(nan_len+2) != (idx-1)%(nan_len+2):
                temp_const_part[ii] = np.nan
    print(p, q)


    # interpolation
    if spline['mask_detected'][idx] == 3:
        # p, q = 24, 24
        spline['spline'][idx+1:idx+nan_len+1] = temp_nocon_part.interpolate(method='polynomial', order=3).loc[idx+1:idx+nan_len]
        spline['spline_aod'][idx+1:idx+nan_len+1] = temp_nocon_part.interpolate(method='polynomial', order=3).loc[idx+1:idx+nan_len]
        spline['spline_aodsc'][idx+1:idx+nan_len+1] = temp_nocon_part.interpolate(method='polynomial', order=3).loc[idx+1:idx+nan_len]

    else:  # 4
        # p, q = 24, 24
        spline['spline'][idx+1:idx+nan_len+1] = temp_nocon_part.interpolate(method='polynomial', order=3).loc[idx+1:idx+nan_len]
        s = temp_const[idx]
        # temp_const[idx] = np.nan
        # li_temp = temp_const[idx-1-p:idx+nan_len+2+q].interpolate(method='polynomial', order=3).loc[idx:idx+nan_len]
        li_temp = temp_const_part.interpolate(method='polynomial', order=3).loc[idx:idx+nan_len]
        spline['spline_aod'][idx:idx+nan_len+1] = li_temp
        spline['spline_aodsc'][idx:idx+nan_len+1] = li_temp*(s/sum(li_temp.values))

    count += 1
    if count % 100 == 0:
        print(f'{idx} ', end='')


df['spline'] = spline['spline'].values.copy()
df['spline_aod'] = spline['spline_aod'].values.copy()
df['spline_aodsc'] = spline['spline_aodsc'].values.copy()
print('**          - SPLINE finished')

df.to_csv('201207_result_aodsc+owa_spline-rev-again.csv')

#
# for idx in [135, 159, 171, 189]:
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['values'][idx-24:idx+30].values, '.-')
#     plt.plot(df['joint_aod'][idx-24:idx+30].values, '.-')
#     plt.plot(df['linear'][idx-24:idx+30].values, '.-')
#     plt.plot(df['spline'][idx-24:idx+30].values, '.-')
#     plt.plot(df['spline_aod'][idx-24:idx+30].values, '.-')
#     plt.plot(df['spline_aodsc'][idx-24:idx+30].values, '.-')
#     plt.axvline(24, color='k', linestyle='--', alpha=0.3)
#     plt.axvline(24+nan_len, color='k', linestyle='--', alpha=0.3)
#     plt.legend(['raw', 'joint', 'linear', 'spline', 'spline_aod', 'spline_aodsc'])
#     plt.tight_layout()


# idx = 192613 에서 에러가 나씀 Value Error:  The number of derivatives at boundaries does not match: expected 1, got 0+0