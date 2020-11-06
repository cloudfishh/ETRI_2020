"""
analyse results
 - joint / LI / ** spline **
2020. 11. 05. Thu
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
# 모든 result 로드, dataframe 하나에 병합
result_filelist = [f for f in os.listdir('D:/2020_ETRI/result_201027') if f.endswith('_result.csv')]
temp = pd.DataFrame([])
for f in result_filelist:
    # temp = pd.concat([temp, pd.read_csv(f'result_201027/{f}', index_col=0)], axis=0)
    temp = pd.concat([temp, pd.read_csv(f'D:/2020_ETRI/result_201027/{f}')], axis=0)


############################################################
# 데이터프레임 인덱스 리셋, nan length 설정
df = temp.copy().reset_index()
nan_len = 5


############################################################
# LINEAR INTERPOLATION - w/ const, w/o const
# 근데 지금 여기서 linear w/o const 결과가 같아야하는데 다르게 나오거든?
# 이걸 확인해봐야됨
linear = df[{'injected', 'mask_detected'}].copy()
linear['imp_linear_const'] = linear['injected'].copy()
linear['imp_linear_no-const'] = linear['injected'].copy()

for idx in np.where((linear['mask_detected']==3)|(linear['mask_detected']==4))[0]:
    if linear['mask_detected'][idx] == 3:
        p = 0
        while pd.isna(linear['imp_linear_const'][idx-p]):
            p += 1
        q = 0
        while pd.isna(linear['imp_linear_const'][idx+nan_len+2+q]):
            q += 1
        linear['imp_linear_const'][idx:idx+nan_len+2] = linear['injected'][idx:idx+nan_len+2].interpolate(method='linear')

    else:   # 4
        p = 0
        while pd.isna(linear['imp_linear_const'][idx-1-p]):
            p += 1
        q = 0
        while pd.isna(linear['imp_linear_const'][idx+nan_len+2+q]):
            q += 1
        s = linear['imp_linear_const'][idx]
        linear['imp_linear_const'][idx] = np.nan
        li_temp = linear['imp_linear_const'][idx-1-p:idx+nan_len+2+q].interpolate(method='linear')
        linear['imp_linear_const'][idx-1-p:idx+nan_len+2+q] = li_temp*(s/sum(li_temp.values))

    linear['imp_linear_no-const'][idx:idx+nan_len+2] = linear['injected'][idx:idx+nan_len+2].interpolate(method='linear')
df['imp_linear_const'] = linear['imp_linear_const'].copy()
df['imp_linear_no-const'] = linear['imp_linear_no-const'].copy()

df.to_csv('D:/2020_ETRI/201101_result_final.csv')



############################################################
# analyse results ~ accuracy
df = pd.read_csv('D:/2020_ETRI/201101_result_final.csv')
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
    temp = [mae1, mae2, mae3, mae4]
    mae_3[i, :] = temp
# print(f'[joint w/, joint w/o, linear] = {np.nanmean(mae_3, axis=0)}')


# with outlier -> mask_inj == 4
idx_4 = np.where(df['mask_detected']==4)[0]
mae_4 = np.empty([len(idx_4), 4])
for i in range(len(idx_4)):
    idx = idx_4[i]
    mae1 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_const'][idx:idx+nan_len+1].values)
    mae2 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_no-const'][idx:idx+nan_len+1].values)
    mae3 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_const'][idx:idx+nan_len+1].values)
    mae4 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_no-const'][idx:idx+nan_len+1].values)
    temp = [mae1, mae2, mae3, mae4]
    mae_4[i, :] = temp


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
df = pd.read_csv('D:/2020_ETRI/201101_result_final.csv')

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
df = pd.read_csv('D:/2020_ETRI/201101_result_final.csv')
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
plt.figure(figsize=(6, 6), dpi=400)
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
plt.savefig('Fig_line_(a).pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)


# without accumulated outlier
for idx in np.where(df['mask_detected']==4)[0][4300:4350]:
    diff = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values) - MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    if diff >= 0.9:
        print(idx, diff)

        plt.figure()
        plt.plot(df['values'][idx:idx+nan_len+1], '-bx', linewidth=1, markersize=12)
        plt.plot(df['imp_no-const'][idx:idx+nan_len+1], '-mv', linewidth=1, markersize=12)
        plt.plot(df['imp_const'][idx:idx+nan_len+1], '-rd', linewidth=1, markersize=12)
        plt.plot(df['imp_linear_const'][idx:idx+nan_len+1], '-cP', linewidth=1, markersize=12)
        plt.plot(df['imp_linear_no-const'][idx:idx+nan_len+1], '-g*', linewidth=1, markersize=12)
        # plt.ylim([0, 1.0])
        plt.xlabel('Time [h]')
        plt.ylabel('Power [kW]')
        plt.legend(['Observed data', 'Joint w/o const.', 'Joint w/ const.', 'LI w/ const.', 'LI w/o const.'])


idx = 50929
# idx = 1218465
h = int(df['Time'][idx][11:13])
idx_0h = idx-h
idx_23h = idx+(24-h)

plt.figure(figsize=(6, 6), dpi=400)
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
plt.savefig('Fig_line_(b).pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)



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

