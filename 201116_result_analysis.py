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
method = 'nearest'
# method = 'similar'
df = eval(f'pd.read_csv("D:/2020_ETRI/201115_result_{method}_final.csv", index_col=0)')

# df_nearest = pd.read_csv('D:/2020_ETRI/201115_result_nearest_final.csv', index_col=0)
# df_similar = pd.read_csv('D:/2020_ETRI/201115_result_similar_final.csv')
# df = eval(f'df_{method}.copy()')


############################################################
# analyse results ~ confusion matrix
# 1. proposed method
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
plt.title(f'method={method}, threshold=optimal', fontsize=14)
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
plt.title(f'method={method}, threshold= 3 ', fontsize=14)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()


############################################################
# analyse results ~ accuracy
# without outlier -> mask_inj == 3
# idx_3 = np.where(df['mask_detected']==3)[0]
# mae_3 = np.empty([len(idx_3), 6])
# for i in range(len(idx_3)):
#     idx = idx_3[i]
#     mae1 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
#     mae2 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_no-const'][idx+1:idx+nan_len+1].values)
#     mae3 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values)
#     mae4 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_no-const'][idx+1:idx+nan_len+1].values)
#     mae5 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_spline_const'][idx+1:idx+nan_len+1].values)
#     mae6 = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_spline_no-const'][idx+1:idx+nan_len+1].values)
#     temp = [mae1, mae2, mae3, mae4, mae5, mae6]
#     mae_3[i, :] = temp
#
#
# # with outlier -> mask_inj == 4
# idx_4 = np.where(df['mask_detected']==4)[0]
# mae_4 = np.empty([len(idx_4), 6])
# for i in range(len(idx_4)):
#     idx = idx_4[i]
#     mae1 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_const'][idx:idx+nan_len+1].values)
#     mae2 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_no-const'][idx:idx+nan_len+1].values)
#     mae3 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_const'][idx:idx+nan_len+1].values)
#     mae4 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_no-const'][idx:idx+nan_len+1].values)
#     mae5 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_spline_const'][idx:idx+nan_len+1].values)
#     mae6 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_spline_no-const'][idx:idx+nan_len+1].values)
#     temp = [mae1, mae2, mae3, mae4, mae5, mae6]
#     mae_4[i, :] = temp
#
#
# # total
# idx_34 = np.where((df['mask_detected']==3)|(df['mask_detected']==4))[0]
# mae_34 = np.empty([len(idx_34), 6])
# for i in range(len(idx_34)):
#     idx = idx_34[i]
#     mae1 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_const'][idx:idx+nan_len+1].values)
#     mae2 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_no-const'][idx:idx+nan_len+1].values)
#     mae3 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_const'][idx:idx+nan_len+1].values)
#     mae4 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_linear_no-const'][idx:idx+nan_len+1].values)
#     mae5 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_spline_const'][idx:idx+nan_len+1].values)
#     mae6 = MAE(df['values'][idx:idx+nan_len+1].values, df['imp_spline_no-const'][idx:idx+nan_len+1].values)
#     temp = [mae1, mae2, mae3, mae4, mae5, mae6]
#     mae_34[i, :] = temp
#
#
# print(f'w/o outlier cases [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
#       f'        = {np.nanmean(mae_3, axis=0)}')
# print(f'w/  outlier cases [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
#       f'        = {np.nanmean(mae_4, axis=0)}')
# print(f'            total [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
#       f'        = {np.nanmean(mae_34, axis=0)}')


############################################################
# analyse results ~ accuracy ~ 통째로 모아서 계산 (시퀀스 하나로 만듦)
col_list = ['imp_const', 'imp_no-const',
            'imp_linear_const', 'imp_linear_no-const',
            'imp_spline_const', 'imp_spline_no-const']

idx_3 = np.where(df['mask_detected']==3)[0]
obs_3 = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_3]).reshape([len(idx_3)*6,])
mae_3 = np.empty([6, ])
for i in range(6):
    prd_3 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_3]).reshape([len(idx_3)*6,])
    mae_3[i] = MAE(obs_3, np.nan_to_num(prd_3))

idx_4 = np.where(df['mask_detected']==4)[0]
obs_4 = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_4]).reshape([len(idx_4)*6,])
mae_4 = np.empty([6, ])
for i in range(6):
    prd_4 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_4]).reshape([len(idx_4)*6,])
    mae_4[i] = MAE(obs_4, np.nan_to_num(prd_4))

idx_tot = np.where((df['mask_detected']==3)|(df['mask_detected'] == 4))[0]
obs_tot = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_tot]).reshape([len(idx_tot)*6, ])
mae_tot = np.empty([6, ])
for i in range(6):
    prd_34 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_tot]).reshape([len(idx_tot)*6, ])
    mae_tot[i] = MAE(obs_tot, np.nan_to_num(prd_34))

print(f'w/o outlier cases [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'        = {mae_3}')
print(f'w/  outlier cases [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'        = {mae_4}')
print(f'            total [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'        = {mae_tot}')


############################################################
# analyse results ~ line plot
idx = np.where(df['mask_detected']==3)[0][0]

# without accumulated outlier
for idx in np.where(df['mask_detected']==3)[0][360:400]:
    diff = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values) - MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    if diff >= 0.04:
        print(idx, diff)

        plt.figure()
        plt.plot(df['values'][idx+1:idx+nan_len+1], '-bx', linewidth=1, markersize=10)
        plt.plot(df['imp_const'][idx+1:idx+nan_len+1], '-rd', linewidth=1, markersize=10)
        plt.plot(df['imp_linear_const'][idx+1:idx+nan_len+1], '-cP', linewidth=1, markersize=10)
        plt.plot(df['imp_spline_const'][idx+1:idx+nan_len+1], '-y.', linewidth=1, markersize=10)

        plt.ylim([0, 1.0])
        plt.xlabel('Time [h]')
        plt.ylabel('Power [kW]')
        plt.legend(['Observed data', 'Joint w/ const.', 'LI w/ const.', 'Spline w/ const.'])


idx = 35564
# 31621 35564
h = int(df['Time'][idx][11:13])
idx_0h = idx-h
idx_23h = idx+(24-h)+6

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 6), dpi=400)
plt.plot(df['values'][idx_0h:idx_23h+1], '-bx', linewidth=1, markersize=10)
plt.plot(np.arange(idx+1,idx+nan_len+1), df['imp_const'][idx+1:idx+nan_len+1], '-rd', linewidth=1, markersize=10)
plt.plot(np.arange(idx+1,idx+nan_len+1), df['imp_linear_const'][idx+1:idx+nan_len+1], '-cP', linewidth=1, markersize=10)
plt.plot(np.arange(idx+1,idx+nan_len+1), df['imp_spline_const'][idx+1:idx+nan_len+1], '-y.', linewidth=1, markersize=10)

plt.plot([idx+1, idx+1], [0, 100], '--k', linewidth=.3)
plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)
plt.ylim([0, 1.2])

plt.xticks(ticks=[t for t in range(idx_0h, idx_23h+1, 6)], labels=[0, 6, 12, 18, 24, 0])
x_str = idx_0h if h<12 else idx_0h+12
x_end = idx_0h+12 if h<12 else idx_23h
# plt.xlim([x_str, x_end])
plt.xlim([idx_0h+18, idx_23h])

plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
plt.legend(['Observed data', 'Joint w/ const.', 'LI w/ const.', 'Spline w/ const.'])
plt.tight_layout()
plt.savefig('Fig_line_(a).pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)


# without accumulated outlier
for idx in np.where(df['mask_detected']==4)[0][5050:5070]:
    diff = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values) - MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    if diff >= 0.02:
        print(idx, diff)

        plt.figure(figsize=(7, 5))
        plt.plot(df['values'][idx-5:idx+nan_len+1+5], '-bx', linewidth=1, markersize=10)
        plt.plot(df['imp_no-const'][idx-5:idx+nan_len+1+5], '-mv', linewidth=1, markersize=10)
        plt.plot(df['imp_const'][idx-5:idx+nan_len+1+5], '-rd', linewidth=1, markersize=10)
        plt.plot(df['imp_linear_const'][idx-5:idx+nan_len+1+5], '-cP', linewidth=1, markersize=10)
        plt.plot(df['imp_linear_no-const'][idx-5:idx+nan_len+1+5], '-g*', linewidth=1, markersize=10)
        plt.plot(df['imp_spline_const'][idx-5:idx+nan_len+1+5], '-y.', linewidth=1, markersize=10)
        plt.plot(df['imp_spline_no-const'][idx-5:idx+nan_len+1+5], '-^', linewidth=1, markersize=10, color='orange')
        plt.plot(df['values'][idx-5:idx+nan_len+1+5], '-bx', linewidth=1, markersize=10)
        plt.plot([idx, idx], [0, 100], '--k', linewidth=.3)
        plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)
        plt.ylim([0, 4])
        plt.xlabel('Time [h]')
        plt.ylabel('Power [kW]')
        plt.legend(['Observed data', 'Joint w/o const.', 'Joint w/ const.', 'LI w/ const.', 'LI w/o const.', 'Spline w/ const.', 'Spline w/o const.'])


idx = 1376491
h = int(df['Time'][idx][11:13])
idx_0h = idx-h
# idx_23h = idx+(24-h)
idx_23h = idx+(24-h)+6

plt.figure(figsize=(8, 6), dpi=400)
plt.plot(df['values'][idx_0h:idx_23h+1], '-bx', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_const'][idx:idx+nan_len+1], '-rd', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_no-const'][idx:idx+nan_len+1], '-mv', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_linear_const'][idx:idx+nan_len+1], '-cP', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_linear_no-const'][idx:idx+nan_len+1], '-g*', linewidth=1, markersize=12)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_spline_const'][idx:idx+nan_len+1], '-y.', linewidth=1, markersize=10)
plt.plot(np.arange(idx,idx+nan_len+1), df['imp_spline_no-const'][idx:idx+nan_len+1], '-^', linewidth=1, markersize=10, color='orange')
plt.plot(idx, df['injected'][idx], 'ks', linewidth=.7, markersize=12)

plt.plot([idx, idx], [0, 100], '--k', linewidth=.3)
plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)

plt.ylim([0, 3])

plt.xticks(ticks=[t for t in range(idx_0h, idx_23h+1, 6)], labels=[0, 6, 12, 18, 24, 0])
# plt.xticks(ticks=[t for t in range(idx_0h, idx_23h+1, 3)], labels=[0, 3, 6, 9, 12, 15, 18, 21, 24])
x_str = idx_0h if h<12 else idx_0h+12
x_end = idx_0h+12 if h<12 else idx_23h
# plt.xlim([x_str+6, x_end+6])
# plt.xlim([x_str, x_end])
plt.xlim([idx_0h+18, idx_23h])

plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
plt.legend(['Observed data', 'Joint w/ const.', 'Joint w/o const.', 'LI w/ const.', 'LI w/o const.', 'Spline w/ const.', 'Spline w/o const.', 'Outlier'], loc='upper right')
plt.tight_layout()
plt.savefig('Fig_line_(b).pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)



############################################################
# analyse results ~ bar plot
list_house = np.unique(df['house'])
# idx_detected_nor = np.where(df['mask_detected']==3)[0]
# idx_detected_acc = np.where(df['mask_detected']==4)[0]

MAE_nor = np.zeros([3, len(list_house)])
MAE_acc = np.zeros([6, len(list_house)])
for i in range(len(list_house)):
    df_temp = df[df['house']==list_house[i]]

    # [joint w/ const, li w/ const, spline w/ const]
    idx_detected_nor = np.where(df_temp['mask_detected'] == 3)[0]
    obs_nor = np.nan_to_num(np.array([df['values'][idx:idx+nan_len+1] for idx in idx_detected_nor]).reshape([len(idx_detected_nor)*6, ]), 0)
    MAE_nor[0, i] = MAE(obs_nor,
                        np.nan_to_num(np.array([df['imp_const'][idx:idx+nan_len+1] for idx in idx_detected_nor]).reshape([len(idx_detected_nor)*6, ]), 0))
    MAE_nor[1, i] = MAE(obs_nor,
                        np.nan_to_num(np.array([df['imp_linear_const'][idx:idx+nan_len+1] for idx in idx_detected_nor]).reshape([len(idx_detected_nor)*6, ]), 0))
    MAE_nor[2, i] = MAE(obs_nor,
                        np.nan_to_num(np.array([df['imp_spline_const'][idx:idx+nan_len+1] for idx in idx_detected_nor]).reshape([len(idx_detected_nor)*6, ]), 0))

    # [joint w/ const, joint w/o const, li w/ const, li w/o const, spline w/ const, spline w/o const]
    idx_detected_acc = np.nan_to_num(np.where(df_temp['mask_detected'] == 4)[0], 0)
    obs_acc = np.nan_to_num(np.array([df['values'][idx:idx+nan_len+1] for idx in idx_detected_acc]).reshape([len(idx_detected_acc)*6, ]), 0)
    MAE_acc[0, i] = MAE(obs_acc,
                        np.nan_to_num(np.array([df['imp_const'][idx:idx+nan_len+1] for idx in idx_detected_acc]).reshape([len(idx_detected_acc)*6, ]), 0))
    MAE_acc[1, i] = MAE(obs_acc,
                        np.nan_to_num(np.array([df['imp_no-const'][idx:idx+nan_len+1] for idx in idx_detected_acc]).reshape([len(idx_detected_acc)*6, ]), 0))
    MAE_acc[2, i] = MAE(obs_acc,
                        np.nan_to_num(np.array([df['imp_linear_const'][idx:idx+nan_len+1] for idx in idx_detected_acc]).reshape([len(idx_detected_acc)*6, ]), 0))
    MAE_acc[3, i] = MAE(obs_acc,
                        np.nan_to_num(np.array([df['imp_linear_no-const'][idx:idx+nan_len+1] for idx in idx_detected_acc]).reshape([len(idx_detected_acc)*6, ]), 0))
    MAE_acc[4, i] = MAE(obs_acc,
                        np.nan_to_num(np.array([df['imp_spline_const'][idx:idx+nan_len+1] for idx in idx_detected_acc]).reshape([len(idx_detected_acc)*6, ]), 0))
    MAE_acc[5, i] = MAE(obs_acc,
                        np.nan_to_num(np.array([df['imp_spline_no-const'][idx:idx+nan_len+1] for idx in idx_detected_acc]).reshape([len(idx_detected_acc)*6, ]), 0))

    print(f'{i}, nor {len(idx_detected_nor)}, acc {len(idx_detected_acc)}, total {len(idx_detected_nor)+len(idx_detected_acc)}')


yy = pd.DataFrame(MAE_nor.T, columns=['Joint w/ const.', 'LI w/ const.', 'Spline w/ const.'])
# plt.figure(figsize=(6, 6), dpi=100)
sns.set(style="ticks", palette=[(1,0,0), (0,1,1), (1,1,0)], font='Helvetica')
# sns.set(style="ticks", palette='Set2', font='Helvetica')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 1.1})
g = sns.factorplot(data=yy, kind="box", size=7, aspect=1,
                   width=.6, fliersize=2.5, linewidth=1.1, notch=False, orient="v")
plt.ylim([0, 0.15])
plt.ylabel('MAE [kW]')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('Fig_MAE (a).pdf', dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format='pdf',
#             transparent=False, bbox_inches=None, pad_inches=0.1,
#             frameon=None, metadata=None)


yy = pd.DataFrame(MAE_acc.T, columns=['Joint w/ const.', 'Joint w/o const.', 'LI w/ const.', 'LI w/o const.', 'Spline w/ const.', 'Spline w/o const.'])
hfont = {'fontname': 'Helvetica'}
# plt.figure(figsize=(6, 6), dpi=400)
sns.set(style="ticks", palette=[(1,0,0), (1,0,1), (0,1,1), (0,1,0), (1,1,0), (1.0, 0.5, 0.25)], font='Helvetica')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 1.1})
g = sns.factorplot(data=yy, kind="box", size=7, aspect=1.3,
                   width=.6, fliersize=2.5, linewidth=1.1, notch=False, orient="v")
plt.ylabel('MAE [kW]', **hfont)
plt.rcParams["font.family"] = "Helvetica"
plt.ylim([0, 0.5])
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('Fig_MAE (b).pdf', dpi=None, facecolor='w', edgecolor='w',
#             orientation='portrait', papertype=None, format='pdf',
#             transparent=False, bbox_inches=None, pad_inches=0.1,
#             frameon=None, metadata=None)


############################################################
# analyse results ~ accuracy ~ detection 결과에 따라 4가지로 나눠서 계산
col_list = ['imp_const', 'imp_no-const',
            'imp_linear_const', 'imp_linear_no-const',
            'imp_spline_const', 'imp_spline_no-const']

df_raw = eval(f'pd.read_csv("D:/2020_ETRI/201115_result_{method}_final.csv", index_col=0)')
a = df[(df['mask_inj']!=0)&(df['mask_inj']!=1)][{'values', 'mask_inj', 'mask_detected', 'imp_const', 'imp_no-const', 'imp_linear_const', 'imp_linear_no-const', 'imp_spline_const', 'imp_spline_no-const'}]
b = a.dropna(axis=0)
df = b

idx_33 = np.where((df['mask_inj']==3)|(df['mask_detected']==3))[0]
obs_33 = np.nan_to_num(np.array([df['values'][idx:idx+nan_len+1] for idx in idx_33]).reshape([len(idx_33)*6, ]), 0)
mae_33 = np.empty([6, ])
for i in range(6):
    prd_33 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_33]).reshape([len(idx_33)*6, ])
    obs_33[[idx for idx in np.where(obs_33 == 0)[0]]] = 0.001
    prd_33[[idx for idx in np.where(prd_33 == 0)[0]]] = 0.001
    mae_33[i] = MAE(obs_33, np.nan_to_num(prd_33))

idx_43 = np.where((df['mask_inj']==4)|(df['mask_detected']==3))[0]
obs_43 = np.nan_to_num(np.array([df['values'][idx:idx+nan_len+1] for idx in idx_43]).reshape([len(idx_43)*6, ]), 0)
mae_43 = np.empty([6, ])
for i in range(6):
    prd_43 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_43]).reshape([len(idx_43)*6, ])
    obs_43[[idx for idx in np.where(obs_43 == 0)[0]]] = 0.001
    prd_43[[idx for idx in np.where(prd_43 == 0)[0]]] = 0.001
    mae_43[i] = MAE(obs_43, np.nan_to_num(prd_43))

idx_34 = np.where((df['mask_inj'] == 3) | (df['mask_detected'] == 4))[0]
obs_34 = np.nan_to_num(np.array([df['values'][idx:idx+nan_len+1] for idx in idx_34]).reshape([len(idx_34)*6, ]), 0)
mae_34 = np.empty([6, ])
for i in range(6):
    prd_34 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_34]).reshape([len(idx_34)*6, ])
    obs_34[[idx for idx in np.where(obs_34 == 0)[0]]] = 0.001
    prd_34[[idx for idx in np.where(prd_34 == 0)[0]]] = 0.001
    mae_34[i] = MAE(obs_34, np.nan_to_num(prd_34))

idx_44 = np.where((df['mask_inj']==4)|(df['mask_detected']==4))[0]
obs_44 = np.nan_to_num(np.array([df['values'][idx:idx+nan_len+1] for idx in idx_44]).reshape([len(idx_44)*6, ]), 0)
mae_44 = np.empty([6, ])
for i in range(6):
    prd_44 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_44]).reshape([len(idx_44)*6, ])
    obs_44[[idx for idx in np.where(obs_44 == 0)[0]]] = 0.001
    prd_44[[idx for idx in np.where(prd_44 == 0)[0]]] = 0.001
    mae_44[i] = MAE(obs_44, np.nan_to_num(prd_44))

print('* detected nor.')
print(f' True negative [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {mae_33}')
print(f'False negative [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {mae_43}')
print('* detected acc. ')
print(f'False positive [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {mae_34}')
print(f' True positive [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {mae_44}')

print('* total')
print(f'[joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'= {[MAE(df["values"].values, df["imp_const"].values), MAE(df["values"].values, df["imp_no-const"].values), MAE(df["values"].values, df["imp_linear_const"].values), MAE(df["values"].values, df["imp_linear_no-const"].values), MAE(df["values"].values, df["imp_spline_const"].values), MAE(df["values"].values, df["imp_spline_no-const"].values)]}')




############################################################
# analyse results ~ accuracy ~ detection 결과에 따라 4가지로 나눠서 계산
# 각 injection마다 mae 구해서 평균내보자. 이제 할 수 있는 게 이것 뿐이다
col_list = ['imp_const', 'imp_no-const',
            'imp_linear_const', 'imp_linear_no-const',
            'imp_spline_const', 'imp_spline_no-const']

idx_33 = np.where((df['mask_inj']==3)|(df['mask_detected']==3))[0]
idx_43 = np.where((df['mask_inj']==4)|(df['mask_detected']==3))[0]
idx_34 = np.where((df['mask_inj'] == 3) | (df['mask_detected'] == 4))[0]
idx_44 = np.where((df['mask_inj']==4)|(df['mask_detected']==4))[0]

mae_33 = np.empty([6, len(idx_33)])
mae_43 = np.empty([6, len(idx_43)])
mae_34 = np.empty([6, len(idx_34)])
mae_44 = np.empty([6, len(idx_44)])
for i in range(6):
    temp = []
    for idx in idx_33:
        temp.append(MAE(df['values'][idx:idx+nan_len+1].values, df[col_list[i]][idx:idx+nan_len+1].values))
    mae_33[i, :] = np.array(temp)

    temp = []
    for idx in idx_43:
        temp.append(MAE(df['values'][idx:idx+nan_len+1].values, df[col_list[i]][idx:idx+nan_len+1].values))
    mae_43[i, :] = np.array(temp)

    temp = []
    for idx in idx_34:
        temp.append(MAE(df['values'][idx:idx+nan_len+1].values, df[col_list[i]][idx:idx+nan_len+1].values))
    mae_34[i, :] = np.array(temp)

    temp = []
    for idx in idx_44:
        temp.append(MAE(df['values'][idx:idx+nan_len+1].values, df[col_list[i]][idx:idx+nan_len+1].values))
    mae_44[i, :] = np.array(temp)



print('* detected nor.')
print(f' True negative [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {np.nan_to_num(mae_33, 0).mean(axis=1)}')
print(f'False negative [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {np.nan_to_num(mae_43, 0).mean(axis=1)}')
print('* detected acc. ')
print(f'False positive [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {np.nan_to_num(mae_34, 0).mean(axis=1)}')
print(f' True positive [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {np.nan_to_num(mae_44, 0).mean(axis=1)}')



print('* detected nor.')
print(f' True negative [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {mae_33.mean(axis=1)}')
print(f'False negative [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {mae_43.mean(axis=1)}')
print('* detected acc. ')
print(f'False positive [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {mae_34.mean(axis=1)}')
print(f' True positive [joint w/, joint w/o, linear w/, linear w/o, spline w/, spline w/o]\n'
      f'             = {mae_44.mean(axis=1)}')
