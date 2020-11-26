"""
imputation test
vanilla / AOD-AI / AOD-AISC

2020. 11. 25. Wed
Soyeong Park
"""
##############################
from funcs import *
import os

##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
# test_house = '1dcb5feb'
f_fwd, f_bwd = 24, 24
nan_len = 5


##############################
# 1. load dataset
df = pd.read_csv('201124_compare.csv', index_col=0)

idx_detected_nor = np.where(df['mask_detected']==3)
idx_detected_acc = np.where(df['mask_detected']==4)
data_col = df['values'].copy()


##############################
# 4-1. JOINT
print('**          - JOINT start')

df['joint'] = df['injected'].copy()
df['joint_aod'] = df['injected'].copy()
df['joint_aod_sc'] = df['injected'].copy()

# 4-1. normal imputation - idx_detected_nor
print('*          - JOINT detected nor cases')
for idx in idx_detected_nor:
    # idx 있는 곳만 injection 남겨서 imputation
    data_inj_temp = data_col.copy()
    data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+nan_len+1] = df['mask_inj'][idx:idx+nan_len+1]
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
    df['joint_aod_sc'][idx+1:idx+nan_len+1] = fcst_bidirec1
    df['joint_aod'][idx+1:idx+nan_len+1] = fcst_bidirec1
    df['joint'][idx+1:idx+nan_len+1] = fcst_bidirec1
    print(f'{idx}', end=' ')

# 4-2-1. acc. imputation - without detection result
print('*          - JOINT detected acc cases - vanilla')
for idx in idx_detected_acc:
    data_inj_temp = data_col.copy()
    data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+nan_len+1] = df['mask_inj'][idx:idx+nan_len+1]
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
    df['joint'][idx+1:idx+nan_len+1] = fcst_bidirec1
    print(f'{idx}', end=' ')

# 4-2-2. acc. imputation - aware detection result
print('*          - JOINT detected acc cases - aod, aodsc')
for idx in idx_detected_acc:
    data_inj_temp = data_col.copy()
    # data_inj_temp[idx:idx+nan_len+1] = data_inj[idx:idx+nan_len+1]
    data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
    mask_inj_temp = np.isnan(data_col).astype('float')
    mask_inj_temp[idx:idx+nan_len+1] = 2
    trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
    fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len+1)
    df['joint_aod'][idx:idx+nan_len+1] = fcst_bidirec1
    acc = data_inj_temp[idx]
    fcst_bidirec1 = fcst_bidirec1*(acc/sum(fcst_bidirec1))
    df['joint_aod_sc'][idx:idx+nan_len+1] = fcst_bidirec1
    print(f'{idx}', end=' ')

print('**          - JOINT finished')


##############################
# 4-2. LINEAR
print('**          - LINEAR start')
linear = df[{'values', 'injected', 'mask_detected'}].copy()
linear['linear'] = linear['injected'].copy()
linear['linear_aod'] = linear['injected'].copy()
linear['linear_aodsc'] = linear['injected'].copy()

for idx in np.where((linear['mask_detected'] == 3) | (linear['mask_detected'] == 4))[0]:
    temp_nocon, temp_const = linear['values'].copy(), linear['values'].copy()
    temp_nocon[:idx], temp_const[:idx] = linear['values'][:idx], linear['values'][:idx]
    temp_nocon[idx:idx+nan_len+2], temp_const[idx:idx+nan_len+2] = linear['injected'][idx:idx+nan_len+2], linear['injected'][idx:idx+nan_len+2]

    # w/o const
    p, q = 0, 0
    while pd.isna(temp_nocon[idx-p]):
        p += 1
    while pd.isna(temp_nocon[idx+nan_len+2+q]):
        q += 1
    linear['linear'][idx:idx+nan_len+2] = temp_nocon[idx-p:idx+nan_len+2+q].interpolate(method='linear')

    # w/ const
    if linear['mask_detected'][idx] == 3:
        p, q = 0, 0
        while pd.isna(temp_const[idx-p]):
            p += 1
        while pd.isna(temp_const[idx+nan_len+2+q]):
            q += 1
        linear['linear_aod'][idx:idx+nan_len+2] = temp_const[idx-p:idx+nan_len+2+q].interpolate(method='linear')
        linear['linear_aodsc'][idx:idx+nan_len+2] = temp_const[idx-p:idx+nan_len+2+q].interpolate(method='linear')

    else:  # 4
        p, q = 0, 0
        while pd.isna(temp_const[idx-1-p]):
            p += 1
        while pd.isna(temp_const[idx+nan_len+2+q]):
            q += 1
        s = temp_const[idx]
        temp_const[idx] = np.nan
        li_temp = temp_const[idx-1-p:idx+nan_len+2+q].interpolate(method='linear').loc[idx:idx+nan_len]
        linear['linear_aod'][idx:idx+nan_len+1] = li_temp
        linear['linear_aodsc'][idx:idx+nan_len+1] = li_temp*(s/sum(li_temp.values))
    print(f'{idx} ', end='')

df['linear'] = linear['linear'].copy()
df['linear_aod'] = linear['linear_aod'].copy()
df['linear_aodsc'] = linear['linear_aodsc'].copy()
print('**          - LINEAR finished')


##############################
# 4-3. SPLINE
print('**          - SPLINE start')
spline = df[{'values', 'injected', 'mask_detected'}].copy().reset_index(drop=True)
spline['spline'] = spline['injected'].copy()
spline['spline_aod'] = spline['injected'].copy()
spline['spline_aodsc'] = spline['injected'].copy()

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
    print(f'{idx} ', end='')

df['spline'] = spline['spline'].values.copy()
df['spline_aod'] = spline['spline'].values.copy()
df['spline_aodsc'] = spline['spline_aodsc'].values.copy()
print('**          - SPLINE finished')

df.to_csv('201125_result_aodsc.csv')