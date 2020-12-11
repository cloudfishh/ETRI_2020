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
# analyse results ~ accuracy ~ detection 결과에 따라 4가지로 나눠서 계산
nan_len = 5
# df = pd.read_csv('D:/202010_energies/201125_result_aodsc+owa_spline-rev.csv')
df = pd.read_csv('D:/202010_energies/201207_result_aodsc+owa_spline-rev-again.csv')

col_list = ['values', 'mask_inj', 'mask_detected',
             'joint', 'joint_aod', 'joint_aod_sc',
            'linear', 'linear_aod', 'linear_aodsc',
            'spline', 'spline_aod', 'spline_aodsc',
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

idx_tot = np.where((df_cut['mask_inj']==3)|(df_cut['mask_inj']==4))[0]
obs_tot = np.array([df_cut['values'][idx:idx+nan_len+1] for idx in idx_tot]).reshape([len(idx_tot)*6, ])
mae_tot = np.empty([num_case, ])
for i in range(num_case):
    prd_tot = np.array([df_cut[case_list[i]][idx:idx+nan_len+1] for idx in idx_tot]).reshape([len(idx_tot)*6, ])
    mae_tot[i] = MAE(obs_tot, prd_tot)

np.savez('D:/202010_energies/201207_MAEs.npz', mae_33=mae_33, mae_34=mae_34, mae_43=mae_43, mae_44=mae_44, mae_tot=mae_tot)

mae = np.load('D:/202010_energies/201207_MAEs.npz')

print('* detected nor.')
print(f' True negative {case_list}\n'
      f'             = {mae["mae_33"]}')
print(f'False negative {case_list}\n'
      f'             = {mae["mae_43"]}')
print('* detected acc. ')
print(f'False positive {case_list}\n'
      f'             = {mae["mae_34"]}')
print(f' True positive {case_list}\n'
      f'             = {mae["mae_44"]}\n')
print(f'         TOTAL {case_list}\n'
      f'             = {mae["mae_tot"]}\n')


# ############################################################
# # analyse results ~ accuracy ~ 통째로 모아서 계산 (시퀀스 하나로 만듦)
# col_list = ['values', 'mask_inj', 'mask_detected',
#              'joint', 'joint_aod', 'joint_aodsc',
#             'linear', 'linear_aod', 'linear_aodsc',
#             'spline', 'spline_aod', 'spline_aodsc',
#             'mars', 'mars_aod', 'mars_aodsc',
#             'owa', 'owa_aod', 'owa_aodsc']
# case_list = col_list[3:]
# num_case = len(case_list)
#
# idx_3 = np.where(df['mask_detected']==3)[0]
# obs_3 = np.array([df['values'][idx+1:idx+nan_len+1] for idx in idx_3]).reshape([len(idx_3)*5,])
# mae_3 = np.empty([num_case, ])
# for i in range(num_case):
#     prd_3 = np.array([df[col_list[i]][idx+1:idx+nan_len+1] for idx in idx_3]).reshape([len(idx_3)*5,])
#     mae_3[i] = MAE(obs_3, np.nan_to_num(prd_3))
#
# idx_4 = np.where(df['mask_detected']==4)[0]
# obs_4 = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_4]).reshape([len(idx_4)*6,])
# mae_4 = np.empty([num_case, ])
# for i in range(num_case):
#     prd_4 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_4]).reshape([len(idx_4)*6,])
#     mae_4[i] = MAE(obs_4, np.nan_to_num(prd_4))
#
# idx_tot = np.where((df['mask_detected']==3)|(df['mask_detected'] == 4))[0]
# obs_tot = np.array([df['values'][idx:idx+nan_len+1] for idx in idx_tot]).reshape([len(idx_tot)*6, ])
# mae_tot = np.empty([num_case, ])
# for i in range(num_case):
#     prd_34 = np.array([df[col_list[i]][idx:idx+nan_len+1] for idx in idx_tot]).reshape([len(idx_tot)*6, ])
#     mae_tot[i] = MAE(obs_tot, np.nan_to_num(prd_34))
#
# print(f'w/o outlier cases {case_list}\n'
#       f'        = {mae_3}')
# print(f'w/  outlier cases {case_list}\n'
#       f'        = {mae_4}')
# print(f'            total {case_list}\n'
#       f'        = {mae_tot}')



##############################
# Bar Chart
mae = np.load('D:/202010_energies/201207_MAEs.npz')
# mae_tot = mae['mae_tot']
cases = ['mae_33', 'mae_34', 'mae_43', 'mae_44', 'mae_tot']
titles = ['True negative', 'False negative', 'False negative', 'True positive', 'Total']

bar_width = 0.3
alpha = 0.5
index = np.arange(4)
# index = np.arange(3)

for case in ['mae_34', 'mae_44', 'mae_tot']:
    mae_temp = mae[case]
    mae_vanilla = [mae_temp[x] for x in range(0, 12, 3)]
    mae_aod = [mae_temp[x+1] for x in range(0, 12, 3)]
    mae_aodsc = [mae_temp[x+2] for x in range(0, 12, 3)]
    # mae_temp = np.delete(mae[case], slice(6, 9), 0)
    # mae_vanilla = [mae_temp[x] for x in range(0, 9, 3)]
    # mae_aod = [mae_temp[x+1] for x in range(0, 9, 3)]
    # mae_aodsc = [mae_temp[x+2] for x in range(0, 9, 3)]

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(7,5), dpi=100)
    p1 = plt.bar(index-bar_width*0.5, mae_vanilla,
                 bar_width, color='r', alpha=alpha, label='Without AOD-AI')
    p2 = plt.bar(index + 0.5*bar_width, mae_aod,
                 bar_width, color='g', alpha=alpha, label='AOD-AI')
    # p3 = plt.bar(index + bar_width*2, mae_aodsc,
    #              bar_width, color='b', alpha=alpha, label='AOD-SC')

    # plt.title(titles[i])
    plt.ylabel('MAE [kW]', fontsize=18)
    plt.xlabel('Methods', fontsize=18)
    plt.xticks(index, ['Joint', 'Linear', 'Spline', 'OWA'], fontsize=15)
    plt.legend((p1[0], p2[0]), ('Without AOD-AI', 'With AOD-AI'), fontsize=15)
    # plt.legend((p1[0], p2[0], p3[0]), ('Vanilla', 'AOD', 'AOD-SC'), fontsize=15)
    if case == 'mae_34':
        plt.yticks(ticks=[y for y in np.arange(0.1, 0.35, 0.05)], labels=[np.round(y, 2) for y in np.arange(0.1, 0.35, 0.05)])
        plt.ylim([0.1, 0.325])
    elif case == 'mae_tot':
        plt.ylim([0, 0.5])
        plt.legend((p1[0], p2[0]), ('Without AOD-AI', 'With AOD-AI'), fontsize=15, loc='upper left')
    plt.tight_layout()
    # plt.savefig(f'Fig_{case}.pdf', dpi=None, facecolor='w', edgecolor='w',
    #         orientation='portrait', papertype=None, format='pdf',
    #         transparent=False, bbox_inches=None, pad_inches=0.1,
    #         frameon=None, metadata=None)

for case in ['mae_33', 'mae_43']:
    mae_temp = mae[case]
    mae_vanilla = [mae_temp[x] for x in range(0, 12, 3)]
    # mae_temp = np.delete(mae[case], slice(6,9), 0)
    # mae_vanilla = [mae_temp[x] for x in range(0, 9, 3)]

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(7,5), dpi=100)
    p1 = plt.bar(index, mae_vanilla,
                 bar_width*1.5, color='r', alpha=alpha, label='Without AOD-AI')
    # p2 = plt.bar(index + bar_width, mae_aod,
    #              bar_width, color='g', alpha=alpha, label='AOD')
    # p3 = plt.bar(index + bar_width*2, mae_aodsc,
    #              bar_width, color='b', alpha=alpha, label='AOD-SC')

    # plt.title(titles[i])
    plt.ylabel('MAE [kW]', fontsize=18)
    plt.xlabel('Methods', fontsize=18)
    plt.xticks(index, ['Joint', 'Linear', 'Spline', 'OWA'], fontsize=15)
    # plt.legend((p1[0], p2[0], p3[0]), ('Vanilla', 'AOD', 'AOD-SC'), fontsize=15)
    if case == 'mae_33':
        # plt.yticks(ticks=[y for y in np.arange(0.1, 0.35, 0.05)], labels=[np.round(y, 2) for y in np.arange(0.1, 0.35, 0.05)])
        plt.ylim([0.04, 0.16])
    elif case == 'mae_43':
        plt.ylim([0, 0.5])
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f'Fig_{case}.pdf', dpi=None, facecolor='w', edgecolor='w',
    #         orientation='portrait', papertype=None, format='pdf',
    #         transparent=False, bbox_inches=None, pad_inches=0.1,
    #         frameon=None, metadata=None)


############################################################
# analyse results ~ line plot
idx = np.where(df['mask_detected']==3)[0][0]

# without accumulated outlier
a = 0
for idx in np.where(df['mask_detected']==3)[0][a:a+100]:
# for idx in np.where(df['mask_detected']==3)[0]:
    diff = MAE(df['values'][idx+1:idx+nan_len+1].values, df['owa'][idx+1:idx+nan_len+1].values) - MAE(df['values'][idx+1:idx+nan_len+1].values, df['linear'][idx+1:idx+nan_len+1].values)
    mae_jo =  MAE(df['values'][idx+1:idx+nan_len+1].values, df['joint'][idx+1:idx+nan_len+1].values)
    mae_ow =  MAE(df['values'][idx+1:idx+nan_len+1].values, df['owa'][idx+1:idx+nan_len+1].values)
    mae_li =  MAE(df['values'][idx+1:idx+nan_len+1].values, df['linear'][idx+1:idx+nan_len+1].values)

    if (mae_jo<mae_li)&(mae_ow<=mae_jo)&(mae_jo<mae_li):
        print(idx, end=' ')

        plt.figure()
        plt.plot(df['values'][idx+1:idx+nan_len+1], '-bx', linewidth=1, markersize=10)
        plt.plot(df['joint'][idx+1:idx+nan_len+1], '-ro', linewidth=1, markersize=10)
        plt.plot(df['owa'][idx+1:idx+nan_len+1], '-cP', linewidth=1, markersize=10)
        plt.plot(df['linear'][idx+1:idx+nan_len+1], '-gd', linewidth=1, markersize=10)
        # plt.plot(df['spline'][idx+1:idx+nan_len+1], '-m*', linewidth=1, markersize=10)

        # plt.ylim([0, 1.0])
        plt.xlabel('Time [h]')
        plt.ylabel('Power [kW]')
        # plt.legend(['Observed data', 'Joint', 'OWA', 'Linear', 'Spline'])
        plt.legend(['Observed data', 'Joint', 'OWA', 'Linear'])


idx = 5112
# 814
h = int(df['Time'][idx][11:13])
idx_0h = idx-h
idx_23h = idx+(24-h)
# idx_23h = idx+(24-h)+6

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 6), dpi=100)
p_va = plt.plot(df['values'][idx_0h:idx_23h+1], '-bx', linewidth=2, markersize=10)
p_si = plt.plot(np.arange(idx+1,idx+nan_len+1), df['spline'][idx+1:idx+nan_len+1], ':c*', linewidth=2, markersize=16)
p_li = plt.plot(np.arange(idx+1,idx+nan_len+1), df['linear'][idx+1:idx+nan_len+1], ':gd', linewidth=2, markersize=12)
p_ow = plt.plot(np.arange(idx+1,idx+nan_len+1), df['owa'][idx+1:idx+nan_len+1], '--mP', linewidth=2, markersize=12)
p_jo = plt.plot(np.arange(idx+1,idx+nan_len+1), df['joint'][idx+1:idx+nan_len+1], '-ro', linewidth=2, markersize=10)

plt.plot([idx+1, idx+1], [0, 100], '--k', linewidth=.3)
plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)
plt.ylim([0.1, 0.6])

plt.xticks(ticks=[t for t in range(idx_0h, idx_23h+1, 6)], labels=[0, 6, 12, 18, 24, 0])
x_str = idx_0h if h<12 else idx_0h+12
x_end = idx_0h+12 if h<12 else idx_23h
plt.xlim([x_str, x_end])
# plt.xlim([x_str+6, x_end])
# plt.xlim([idx_0h+18, idx_23h])

plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
# plt.legend([p_va[0], p_jo[0], p_ow[0], p_li[0], p_si[0]], ['Observed data', 'Joint', 'OWA', 'Linear', 'Spline'])
plt.legend([p_va[0], p_jo[0], p_ow[0], p_li[0], p_si[0]], ['Observed data', 'Joint', 'OWA', 'Linear', 'Spline'])
plt.tight_layout()
plt.savefig('Fig_line_(a).pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)


# with accumulated outlier
for idx in np.where(df['mask_detected']==4)[0][:50]:
    diff = MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_linear_const'][idx+1:idx+nan_len+1].values) - MAE(df['values'][idx+1:idx+nan_len+1].values, df['imp_const'][idx+1:idx+nan_len+1].values)
    mae_jo =  MAE(df['values'][idx+1:idx+nan_len+1].values, df['joint'][idx+1:idx+nan_len+1].values)
    mae_ow =  MAE(df['values'][idx+1:idx+nan_len+1].values, df['owa'][idx+1:idx+nan_len+1].values)
    mae_li =  MAE(df['values'][idx+1:idx+nan_len+1].values, df['linear'][idx+1:idx+nan_len+1].values)

    if (mae_jo<mae_li)&(mae_ow<=mae_jo)&(mae_jo<mae_li):
        print(idx, end=' ')

        plt.figure(figsize=(7, 5))
        plt.plot(df['values'][idx:idx+nan_len+1], '-bx', linewidth=1, markersize=10)
        plt.plot(df['joint_aod'][idx:idx+nan_len+1], '-ro', linewidth=1, markersize=10)
        plt.plot(df['owa_aod'][idx:idx+nan_len+1], '-cP', linewidth=1, markersize=10)
        plt.plot(df['linear_aod'][idx:idx+nan_len+1], '-gd', linewidth=1, markersize=10)
        plt.plot(df['spline_aod'][idx:idx+nan_len+1], '-m*', linewidth=1, markersize=10)
        plt.plot(idx, df['injected'][idx], 'ks', linewidth=.7, markersize=12)

        # plt.plot(df['values'][idx-5:idx+nan_len+1+5], '-bx', linewidth=1, markersize=10)
        # plt.plot(df['joint'][idx-5:idx+nan_len+1+5], '-mv', linewidth=1, markersize=10)
        # plt.plot(df['joint_aod'][idx-5:idx+nan_len+1+5], '-rd', linewidth=1, markersize=10)
        # plt.plot(df['owa'][idx-5:idx+nan_len+1+5], '-cP', linewidth=1, markersize=10)
        # plt.plot(df['owa_aod'][idx-5:idx+nan_len+1+5], '-g*', linewidth=1, markersize=10)
        # plt.plot(df['linear'][idx-5:idx+nan_len+1+5], '-y.', linewidth=1, markersize=10)
        # plt.plot(df['linear_aod'][idx-5:idx+nan_len+1+5], '-^', linewidth=1, markersize=10, color='orange')
        plt.plot([idx, idx], [0, 10], '--k', linewidth=.3)
        plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)
        plt.ylim([0, df['injected'][idx]+0.03])
        plt.xlabel('Time [h]')
        plt.ylabel('Power [kW]')
        plt.legend(['Observed data',  'Joint AOD-AI', 'OWA AOD-AI', 'Linear AOD-AI', 'Spline AOD-AI'])


idx = 2738
h = int(df['Time'][idx][11:13])
idx_0h = idx-h
idx_23h = idx+(24-h)
# idx_23h = idx+(24-h)+6

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 6), dpi=100)
p_va = plt.plot(df['values'][idx_0h:idx_23h+1], '-bx', linewidth=2, markersize=12)
p_jo = plt.plot(np.arange(idx,idx+nan_len+1), df['joint_aod'][idx:idx+nan_len+1], '-ro', linewidth=2, markersize=10)
p_ow = plt.plot(np.arange(idx,idx+nan_len+1), df['owa_aod'][idx:idx+nan_len+1], '--mP', linewidth=2, markersize=12)
p_li = plt.plot(np.arange(idx,idx+nan_len+1), df['linear_aod'][idx:idx+nan_len+1], ':gd', linewidth=2, markersize=12)
p_si = plt.plot(np.arange(idx,idx+nan_len+1), df['spline_aod'][idx:idx+nan_len+1], ':c*', linewidth=2, markersize=16)
p_ou = plt.plot(idx, df['injected'][idx], 'ks', linewidth=.7, markersize=12)

plt.plot([idx, idx], [0, 100], '--k', linewidth=.3)
plt.plot([idx+nan_len, idx+nan_len], [0, 100], '--k', linewidth=.3)

plt.ylim([0, 1.2])

plt.xticks(ticks=[t for t in range(idx_0h, idx_23h+1, 6)], labels=[0, 6, 12, 18, 24, 0])
# plt.xticks(ticks=[t for t in range(idx_0h, idx_23h+1, 3)], labels=[0, 3, 6, 9, 12, 15, 18, 21, 24])
x_str = idx_0h if h<12 else idx_0h+12
x_end = idx_0h+12 if h<12 else idx_23h
# plt.xlim([x_str+6, x_end+6])
plt.xlim([x_str, x_end])
# plt.xlim([idx_0h+18, idx_23h])

plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
plt.legend([p_va[0], p_jo[0], p_ow[0], p_li[0], p_si[0], p_ou[0]],
           ['Observed data', 'Joint AOD-AI', 'OWA AOD-AI', 'Linear AOD-AI', 'Spline AOD-AI', 'Outlier'], loc='upper right')
# plt.legend([p_va[0], p_jo[0], p_ow[0], p_li[0], p_ou[0]],
#            ['Observed data', 'Joint AOD-AI', 'OWA AOD-AI', 'Linear AOD-AI', 'Outlier'], loc='upper right')
plt.tight_layout()
plt.savefig('Fig_line_(b).pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)

