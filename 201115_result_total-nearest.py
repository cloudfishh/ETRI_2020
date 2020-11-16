"""
analyse the all results from whole households
 ~ NEAREST NEIGHBOR RESULT ~
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
method = 'nearest'

result_filelist = [f for f in os.listdir(f'D:/2020_ETRI/result_201115_total-{method}') if f.endswith('_result.csv')]
df_concat = pd.DataFrame([])
for f in result_filelist:
    df_load = pd.read_csv(f'D:/2020_ETRI/result_201115_total-{method}/{f}')
    df_load.insert(0, 'house', np.empty([df_load.shape[0], ]))
    df_load['house'] = f[7:15]
    df_concat = pd.concat([df_concat, df_load], axis=0)

df = df_concat.copy().reset_index(drop=True)
df.to_csv('D:/2020_ETRI/201115_result_justconcat.csv')


############################################################
# LINEAR INTERPOLATION
df = pd.read_csv('D:/2020_ETRI/201115_result_justconcat.csv')
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

df.to_csv('D:/2020_ETRI/201115_result_final.csv')


############################################################
# LINEAR INTERPOLATION ~ SPLINE
# df = pd.read_csv('D:/2020_ETRI/201101_result_final.csv', index_col=0)
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

df.to_csv('D:/2020_ETRI/201101_result_final.csv')


############################################################
# analyse results ~ accuracy
df = pd.read_csv('D:/2020_ETRI/201115_result_final.csv')
nan_len = 5

# without outlier -> mask_inj == 3
idx_3 = np.where(df['mask_detected']==3)[0]
mae_3 = np.empty([len(idx_3), 4])
for i in range(len(idx_3)):
    idx = idx_3[i]
    mae1 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    mae2 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_no-const'][idx+1:idx+nan_len+1].values)
    mae3 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values)
    mae4 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_no-const'][idx+1:idx+nan_len+1].values)
    df_concat = [mae1, mae2, mae3, mae4]
    mae_3[i, :] = df_concat
# print(f'[joint w/, joint w/o, li w/, li w/o] = {np.nanmean(mae_3, axis=0)}')


# with outlier -> mask_inj == 4
idx_4 = np.where(df['mask_detected']==4)[0]
mae_4 = np.empty([len(idx_4), 4])
for i in range(len(idx_4)):
    idx = idx_4[i]
    mae1 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_const'][idx:idx+nan_len+1].values)
    mae2 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_no-const'][idx:idx+nan_len+1].values)
    mae3 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_const'][idx:idx+nan_len+1].values)
    mae4 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_no-const'][idx:idx+nan_len+1].values)
    df_concat = [mae1, mae2, mae3, mae4]
    mae_4[i, :] = df_concat


# total
mae_total = [MAE(df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values, df['imp_const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values),
             MAE(df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values, df['imp_no-const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values),
             MAE(df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values, df['imp_linear_const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values),
             MAE(df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values, df['imp_linear_no-const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values)]
# df['values'][(df['mask_inj']==2)|(df['mask_detected']==4)].values
# df['imp_const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values
# df['imp_no-const'][(df['mask_inj']==2)|(df['mask_detected']==4)].values
# df['imp_linear'][(df['mask_inj']==2)|(df['mask_detected']==4)].values

print(f'w/o outlier cases [joint w/, joint w/o, linear w/, linear w/o] = {np.nanmean(mae_3, axis=0)}')
print(f'w/  outlier cases [joint w/, joint w/o, linear w/, linear w/o] = {np.nanmean(mae_4, axis=0)}')
print(f'            total [joint w/, joint w/o, linear w/, linear w/o] = {mae_total}')


############################################################
# analyse results ~ confusion matrix
# 1. proposed method
df = pd.read_csv('D:/2020_ETRI/201115_result_final.csv')

idx_injected = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
idx_detected_acc = np.where(df['mask_detected'] == 4)[0]
idx_real_acc = np.where(df['mask_inj'] == 4)[0]

idx_detected = np.isin(idx_injected, idx_detected_acc)
idx_real = np.isin(idx_injected, idx_real_acc)
cm = confusion_matrix(idx_real, idx_detected)

group_names = ['TN', 'FP', 'FN', 'TP']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
cm_label = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
cm_label = np.asarray(cm_label).reshape(2, 2)

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(4, 4), dpi=100)
sns.heatmap(cm, annot=cm_label, fmt='', square=True, cmap='Greys', annot_kws={'size': 15},  # 'gist_gray': reverse
            xticklabels=['normal', 'anomaly'], yticklabels=['normal', 'anomaly'], cbar=False)
# plt.title(f'{test_house}, {method[17:]}, nan_length=3, threshold={threshold}', fontsize=14)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()


# 2. threshold=3
threshold = 3

idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
# detected = np.zeros(len(data_col))
# detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
# detected[idx_detected_acc.astype('int')] = 4
# df['mask_detected'] = detected

idx_injected = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
idx_real_nor = np.where(df['mask_inj'] == 3)[0]
idx_real_acc = np.where(df['mask_inj'] == 4)[0]

idx_detected = np.isin(idx_injected, idx_detected_acc)
idx_real = np.isin(idx_injected, idx_real_acc)
cm = confusion_matrix(idx_real, idx_detected)

group_names = ['TN', 'FP', 'FN', 'TP']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
cm_label = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
cm_label = np.asarray(cm_label).reshape(2, 2)

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(4, 4), dpi=100)
sns.heatmap(cm, annot=cm_label, fmt='', square=True, cmap='Greys', annot_kws={'size': 15},  # 'gist_gray': reverse
            xticklabels=['normal', 'anomaly'], yticklabels=['normal', 'anomaly'], cbar=False)
# plt.title(f'{test_house}, {method[17:]}, nan_length=3, threshold={threshold}', fontsize=14)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()


############################################################
# analyse results ~ line plot
df = pd.read_csv('D:/2020_ETRI/201115_result_final.csv')
nan_len = 5

idx = np.where(df['mask_detected']==3)[0][0]

# without accumulated outlier
for idx in np.where(df['mask_detected']==3)[0][300:350]:
    diff = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values) - MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    if diff >= 0.04:
        print(idx, diff)

        plt.figure()
        plt.plot(df['values'][idx+1:idx+nan_len+1], '-bx', linewidth=1, markersize=12)
        plt.plot(df['imp_const'][idx+1:idx+nan_len+1], '-rd', linewidth=1, markersize=12)
        plt.plot(df['imp_linear_const'][idx+1:idx+nan_len+1], '-cP', linewidth=1, markersize=12)
        # plt.plot(df['imp_linear_no-const'][idx+1:idx+nan_len+1], '-g*', linewidth=1, markersize=12)
        plt.ylim([0, 1.0])
        plt.xlabel('Time [h]')
        plt.ylabel('Power [kW]')
        plt.legend(['Observed data', 'Joint w/ const.', 'LI w/ const.'])


idx = 21186
h = int(df['Time'][idx][11:13])
idx_0h = idx-h
idx_23h = idx+(24-h)

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(6, 6), dpi=100)
plt.plot(df['values'][idx_0h:idx_23h+1], '-bx', linewidth=1, markersize=12)
plt.plot(np.arange(idx+1,idx+nan_len+1), df['imp_const'][idx+1:idx+nan_len+1], '-rd', linewidth=1, markersize=12)
plt.plot(np.arange(idx+1,idx+nan_len+1), df['imp_linear_const'][idx+1:idx+nan_len+1], '-cP', linewidth=1, markersize=12)

plt.plot([idx+1, idx+1], [0, 100], '--k', linewidth=.3)
plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)
plt.ylim([0, 1])

plt.xticks(ticks=[t for t in range(idx_0h, idx_23h+1, 6)], labels=[0, 6, 12, 18, 24])
x_str = idx_0h if h<12 else idx_0h+12
x_end = idx_0h+12 if h<12 else idx_23h
plt.xlim([x_str, x_end])

plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
plt.legend(['Observed data', 'Joint Imp.', 'LI interp.'])
plt.tight_layout()
# plt.savefig('Fig_line_(a).pdf', dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format='pdf',
#             transparent=False, bbox_inches=None, pad_inches=0.1,
#             frameon=None, metadata=None)


# without accumulated outlier
for idx in np.where(df['mask_detected']==4)[0][4300:4350]:
    diff = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values) - MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    if diff >= 0.1:
        print(idx, diff)

        plt.figure()
        plt.plot(df['values'][idx-5:idx+nan_len+1+5], '-bx', linewidth=1, markersize=12)
        plt.plot(df['imp_no-const'][idx-5:idx+nan_len+1+5], '-mv', linewidth=1, markersize=12)
        plt.plot(df['imp_const'][idx-5:idx+nan_len+1+5], '-rd', linewidth=1, markersize=12)
        plt.plot(df['imp_linear_const'][idx-5:idx+nan_len+1+5], '-cP', linewidth=1, markersize=12)
        plt.plot(df['imp_linear_no-const'][idx-5:idx+nan_len+1+5], '-g*', linewidth=1, markersize=12)
        plt.plot([idx, idx], [0, 100], '--k', linewidth=.3)
        plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)
        plt.ylim([0, 3])
        plt.xlabel('Time [h]')
        plt.ylabel('Power [kW]')
        plt.legend(['Observed data', 'Joint w/o const.', 'Joint w/ const.', 'LI w/ const.', 'LI w/o const.'])


idx = 50929
# idx = 1213570
h = int(df['Time'][idx][11:13])
idx_0h = idx-h
idx_23h = idx+(24-h)

plt.figure(figsize=(6, 6), dpi=100)
plt.plot(df['values'][idx_0h:idx_23h+1], '-bx', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_const'][idx:idx+nan_len+1], '-rd', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_no-const'][idx:idx+nan_len+1], '-mv', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_linear_const'][idx:idx+nan_len+1], '-cP', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_linear_no-const'][idx:idx+nan_len+1], '-g*', linewidth=1, markersize=12)
plt.plot(idx, df['injected'][idx], 'ks', linewidth=.7, markersize=12)

plt.plot([idx, idx], [0, 100], '--k', linewidth=.3)
plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)

plt.ylim([0, 1.5])

plt.xticks(ticks=[t for t in range(idx_0h, idx_23h+1, 6)], labels=[0, 6, 12, 18, 24])
x_str = idx_0h if h<12 else idx_0h+12
x_end = idx_0h+12 if h<12 else idx_23h
# plt.xlim([x_str+6, x_end+6])
plt.xlim([x_str, x_end])

plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
plt.legend(['Observed data', 'Joint w/ const.', 'Joint w/o const.', 'LI w/ const.', 'LI w/o const.', 'Outlier'], loc='upper right')
plt.tight_layout()
# plt.savefig('Fig_line_(b).pdf', dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format='pdf',
#             transparent=False, bbox_inches=None, pad_inches=0.1,
#             frameon=None, metadata=None)



############################################################
# analyse results ~ bar plot
idx_detected_nor = np.where(df['mask_detected']==3)[0]
idx_detected_acc = np.where(df['mask_detected']==4)[0]

MAE_nor = np.zeros([2, len(idx_detected_nor)])
for i in range(len(idx_detected_nor)):
    # [joint w/ const, li w/ const]
    idx = idx_detected_nor[i]
    MAE_nor[0, i] = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    MAE_nor[1, i] = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values)

MAE_acc = np.zeros([4, len(idx_detected_acc)])
for i in range(len(idx_detected_acc)):
    # [joint w/ const, joint w/o const, li w/ const, li w/o const]
    idx = idx_detected_acc[i]
    MAE_acc[0, i] = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_const'][idx:idx+nan_len+1].values)
    MAE_acc[1, i] = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_no-const'][idx:idx+nan_len+1].values)
    MAE_acc[2, i] = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_const'][idx:idx+nan_len+1].values)
    MAE_acc[3, i] = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_no-const'][idx:idx+nan_len+1].values)


yy = pd.DataFrame(MAE_nor.T, columns=['Joint Imp.', 'LI interp.'])
# plt.figure(figsize=(6, 6), dpi=100)
sns.set(style="ticks", palette=[(0.4, 0.7607843137254902, 0.6470588235294118),(0.5529411764705883, 0.6274509803921569, 0.796078431372549)], font='Helvetica')
# sns.set(style="ticks", palette='Set2', font='Helvetica')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 1.1})
g = sns.factorplot(data=yy, kind="box", size=7, aspect=0.6,
                   width=.5, fliersize=2.5, linewidth=1.1, notch=False, orient="v")
plt.ylim([0, 5])
plt.ylabel('MAE [kW]')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Fig_MAE (a).pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)


yy = pd.DataFrame(MAE_acc.T, columns=['Joint w/ const.', 'Joint w/o const', 'LI w/ const.', 'LI w/o const.'])
hfont = {'fontname': 'Helvetica'}
# plt.figure(figsize=(6, 6), dpi=400)
sns.set(style="ticks", palette='Set2', font='Helvetica')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 1.1})
g = sns.factorplot(data=yy, kind="box", size=7, aspect=0.6,
                   width=.8, fliersize=2.5, linewidth=1.1, notch=False, orient="v")
plt.ylabel('MAE [kW]', **hfont)
plt.rcParams["font.family"] = "Helvetica"
plt.ylim([0, 5])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Fig_MAE (b).pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)


############################################################
# mask_detect == 3 에서 impt_linear_const, impt_linear_no-const 왜 다른지 확인 좀 해봅시다
idx_diff = []
for idx in np.where(df['mask_detected']==3)[0]:
    if sum(df['imp_linear_const'][idx:idx+nan_len+1].values == df['imp_linear_no-const'][idx:idx+nan_len+1].values) != nan_len+1:
        idx_diff.append(idx)
print(f'{len(np.where(df["mask_detected"]==3)[0])}, {len(idx_diff)}')


print(sum(df['imp_linear_const'].fillna(0).values == df_original['imp_linear_const'].fillna(0).values))
print(sum(df['imp_linear_no-const'].fillna(0).values == df_original['imp_linear_no-const'].fillna(0).values))

a = np.where(df['imp_linear_const'].fillna(0) != df_original['imp_linear_const'].fillna(0))[0]