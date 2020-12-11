from funcs import *
import os
from matplotlib import pyplot as plt
from scipy import interpolate


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

# idx = np.where((spline['mask_detected'] == 3) | (spline['mask_detected'] == 4))[0][30]
idx = np.where((spline['mask_detected'] == 3))[0][100]
count = 0

for idx in np.where((spline['mask_detected'] == 3) | (spline['mask_detected'] == 4))[0]:
    temp_nocon, temp_const = spline['values'].copy(), spline['values'].copy()
    temp_nocon[:idx], temp_const[:idx] = spline['values'][:idx], spline['values'][:idx]
    temp_nocon[idx:idx+nan_len+2], temp_const[idx:idx+nan_len+2] = spline['injected'][idx:idx+nan_len+2], spline['injected'][idx:idx+nan_len+2]
    # w/o const
    p, q = 24, 24
    # p, q = 24*7, 24*7
    # 그럼 얼만큼이나 넣어줘야겠는가?
    # 일단 둘다 해보고 결과가 얼마나 많이 다른지 비교해볼것.
    temp_nocon_part = temp_nocon[idx-p:idx+nan_len+2+q].copy()
    for ii in range(temp_nocon_part.index[0], temp_nocon_part.index[-1]):
        if ii % (nan_len+1) != idx % (nan_len+1):
            temp_nocon_part[ii] = np.nan

    # scipy interpolation
    x_in = temp_nocon_part.index[~pd.isna(temp_nocon_part)].values
    y_in = temp_nocon_part[~pd.isna(temp_nocon_part)].values
    f = interpolate.interp1d(x_in, y_in, kind='cubic')
    x_out = np.arange(idx-p, idx+nan_len+2+q)
    y_out = f(x_out)



    plt.figure(figsize=(11, 6))

    plt.plot(temp_nocon[idx-p:idx+nan_len+2+q], '.-')
    plt.plot(temp_nocon_part, '*', markersize=15)
    plt.plot(temp_nocon_part.interpolate(method='spline', order=3), '.-')
    # plt.plot(temp_nocon_part.loc[idx-6:idx+nan_len+7].interpolate(method='spline', order=3), 'P-')

    plt.plot(temp_nocon_part.interpolate(method='polynomial', order=3), 'X-')
    # plt.plot(temp_nocon_part.loc[idx-6:idx+nan_len+7].interpolate(method='polynomial', order=3), '.-')
    # plt.plot(temp_nocon[idx-p:idx+nan_len+2+q].interpolate(method='polynomial', order=3).loc[idx+1:idx+nan_len], '.-')

    plt.plot(x_out, y_out, '.-')

    plt.axvline(x=idx, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=idx+nan_len+1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    # plt.legend(['raw', 'remain', 'spline, preprocessed (idx-24:idx+24)', 'spline, preprocessed (idx-6:idx+6)', 'poly, preprocessed (idx-24:idx+24)', 'poly, preprocessed (idx-6:idx+6)'], 'poly, non-prepro')
    plt.xlim([idx-p-1, idx+nan_len+2+q])
    plt.ylim([0.05, 0.65])
    plt.legend(['raw', 'point', 'pd spline', 'pd poly', 'scipy cubic', 'scipy poly'])
    plt.tight_layout()



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
