"""
Anomaly detection with clustering
 - Analysis

2020. 12. 28. Mon.
Soyeong Park
"""
from funcs import *
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import confusion_matrix


def MAE(A, B):
    MAE_temp = 0
    for kk in range(0, len(A)):
        MAE_temp += abs(A[kk]-B[kk])/len(A)
    return MAE_temp


df = pd.read_csv('D:/202010_energies/201214_result_kmeans-added.csv', index_col=0)
case = 'mask_detected'

# for case in ['mask_detected_km_v', 'mask_detected_km_z', 'mask_detected']:
for case in ['mask_detected_km_v', 'mask_detected']:
    idx_injected = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
    idx_real_nor = np.where(df['mask_inj'] == 3)[0]
    idx_real_acc = np.where(df['mask_inj'] == 4)[0]

    idx_detected_nor = np.where(df[case]==3)[0]
    idx_detected_acc = np.where(df[case]==4)[0]

    idx_detected = np.isin(idx_injected, idx_detected_acc)
    idx_real = np.isin(idx_injected, idx_real_acc)
    cm = confusion_matrix(idx_real, idx_detected)

    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    cm_label = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    cm_label = np.asarray(cm_label).reshape(2, 2)

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(4, 4), dpi=100)
    sns.heatmap(cm, annot=cm_label, fmt='', square=True, cmap='Greys', annot_kws={'size': 15}, # 'gist_gray': reverse
                xticklabels=['normal', 'anomaly'], yticklabels=['normal', 'anomaly'], cbar=False)
    # plt.title(f'{test_house}, {method[17:]}, nan_length=3, threshold={threshold}', fontsize=14)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig(f'201229_cm_{case}.png')


# imputation 정확도도 비교
nan_len = 5
for case in ['mask_detected_km_v', 'mask_detected']:
    idx_detected_nor = np.where(df[case] == 3)[0]
    idx_detected_acc = np.where(df[case] == 4)[0]

    data_col = df['values']
    result_con = df['injected'].copy()
    result_noc = df['injected'].copy()

    # 4-1. normal imputation - idx_detected_nor
    for idx in idx_detected_nor:
        # idx 있는 곳만 injection 남겨서 imputation
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+nan_len+1] = df['mask_inj'][idx:idx+nan_len+1]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
        result_con[idx+1:idx+nan_len+1] = fcst_bidirec1
        result_noc[idx+1:idx+nan_len+1] = fcst_bidirec1
        print(idx, end=' ')
    print(f'{case} - detected nor end\n\n')

    # 4-2. acc. imputation - idx_detected_acc
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        # data_inj_temp[idx:idx+nan_len+1] = data_inj[idx:idx+nan_len+1]
        data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+nan_len+1] = 2
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len+1)
        acc = data_inj_temp[idx]
        fcst_bidirec1 = fcst_bidirec1*(acc/sum(fcst_bidirec1))
        result_con[idx:idx+nan_len+1] = fcst_bidirec1
        print(idx, end=' ')
    print(f'{case} - detected acc, const end\n\n')

    # 4-2-2. acc. imputation - no constraints
    for idx in idx_detected_acc:
        data_inj_temp = data_col.copy()
        data_inj_temp[idx:idx+nan_len+1] = df['injected'][idx:idx+nan_len+1]
        mask_inj_temp = np.isnan(data_col).astype('float')
        mask_inj_temp[idx:idx+nan_len+1] = df['mask_inj'][idx:idx+nan_len+1]
        trn_x, trn_y, tst_x = make_bidirectional_input(data_inj_temp, mask_inj_temp)
        fcst_bidirec1, _ = linear_prediction(trn_x, trn_y, tst_x, f_len_fwd=1, f_len_bwd=1, n_len=nan_len)
        result_noc[idx+1:idx+nan_len+1] = fcst_bidirec1
        print(idx, end=' ')
    print(f'{case} - detected acc, no const end\n\n')

    print(f'*** {case}')
    print(f'     const: {MAE(data_col.values, result_con.values)}')
    print(f'  no const:{MAE(data_col.values, result_noc.values)}\n')
