"""
Anomaly detection with clustering

2020. 12. 13. Sun.
Soyeong Park
"""
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
# from matplotlib import cm
# import seaborn as sns
# from sklearn.metrics import confusion_matrix


nan_len = 5
df = pd.read_csv('D:/202010_energies/201207_result_aodsc+owa_spline-rev-again.csv', index_col=0)

df = df.astype({'house': 'str'})

h_list_raw = np.unique(df['house'].values.astype('str'), return_index=True)
house_list = h_list_raw[0].astype('str')[np.argsort(h_list_raw[1])]

house = house_list[5]
# house = house_list[238]
# house = '68181c16'

result_km_v = np.array([])
result_km_z = np.array([])
for house in house_list:
    starttime = time.time()
    df_temp = df[df['house']==house]

    cand_v = df_temp['injected'][(df_temp['mask_inj'] == 3) | (df_temp['mask_inj'] == 4)].values
    cand_v = np.nan_to_num(cand_v)
    temp_v = np.array([np.zeros(cand_v.shape), cand_v]).transpose()
    kmeans_v = KMeans(n_clusters=2).fit(temp_v)
    label_km_v = kmeans_v.labels_

    cand_z = df_temp['z_score'][(df_temp['mask_inj'] == 3) | (df_temp['mask_inj'] == 4)].values
    cand_z = np.nan_to_num(cand_z)
    if sum(cand_z>10**100) > 0:
        cand_z[np.where(cand_z > 10**100)[0]] = 50
    temp_z = np.array([np.zeros(cand_z.shape), cand_z]).transpose()
    kmeans_z = KMeans(n_clusters=2).fit(temp_z)
    label_km_z = kmeans_z.labels_

    a = pd.DataFrame(np.array([cand_v, label_km_v]).transpose())
    b = a.sort_values(1).reset_index(drop=True)
    plt.figure()
    plt.plot(b[0], '.')
    plt.plot(b[1], '.')
    plt.plot([0, len(label_km_v)], kmeans_v.cluster_centers_[:, 1], 'r*', markersize=8, markeredgecolor='k')
    plt.legend(['values', 'labels', 'centers'])
    #
    #
    # a = pd.DataFrame(np.array([cand_z, label_km_z]).transpose())
    # b = a.sort_values(1).reset_index(drop=True)
    # plt.figure()
    # plt.plot(b[0], '.')
    # plt.plot(b[1], '.')
    # plt.ylim([-0.05, 5])
    if (len(np.where(label_km_v==0)[0])==0)|(len(np.where(label_km_v==1)[0])==0):
        km_v_nor = 0 if len(np.where(label_km_v==1)[0])==0 else 1
    else:
        if cand_v[np.where(label_km_v==0)[0]].min() < cand_v[np.where(label_km_v==1)[0]].min():
            km_v_nor = 0
        else:
            km_v_nor = 1

    if (len(np.where(label_km_z==0)[0])==0)|(len(np.where(label_km_z==1)[0])==0):
        km_z_nor = 0 if len(np.where(label_km_z==1)[0])==0 else 1
    else:
        if cand_z[np.where(label_km_z == 0)[0]].min() < cand_z[np.where(label_km_z == 1)[0]].min():
            km_z_nor = 0
        else:
            km_z_nor = 1

    result_km_v_temp, result_km_z_temp = df_temp['mask_detected'].values.copy(), df_temp['mask_detected'].values.copy()
    j = 0
    for i in np.where((result_km_v_temp==3)|(result_km_v_temp==4))[0]:
        result_km_v_temp[i] = 3 if label_km_v[j]==km_v_nor else 4
        result_km_z_temp[i] = 3 if label_km_z[j]==km_z_nor else 4
        j += 1

    result_km_v = np.append(result_km_v, result_km_v_temp)
    result_km_z = np.append(result_km_z, result_km_z_temp)

    print(f'{house} FINISHED - elasped time {(time.time()-starttime):.3f} secs / {len(result_km_v_temp)}, {len(result_km_z_temp)} / {len(result_km_v)}, {len(result_km_z)} / {df_temp.shape[0]*(np.where(house_list==house)[0][0]+1)}')
    if (len(result_km_v_temp)!=df_temp.shape[0])|(len(result_km_z_temp)!=df_temp.shape[0]):
        print(f'\n     {house} SHOULD BE CHECKED\n')

print(len(result_km_v), len(result_km_z))

df['mask_detected_km_v'] = result_km_v
df['mask_detected_km_z'] = result_km_z

df.to_csv('D:/202010_energies/201214_result_kmeans-added.csv')
