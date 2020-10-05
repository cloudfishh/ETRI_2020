import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


##############################
file_dir = 'D:/2020_ETRI/NAB_data/NAB_data'
direc = ['artificialNoAnomaly',
         'artificialWithAnomaly',
         'realAdExchange',
         'realAWSCloudwatch',
         'realKnownCause',
         'realTraffic',
         'realTweets']

for d in range(len(direc)):
    csv_list = os.listdir(f'{file_dir}/data/{direc[d]}')
    for c in range(len(csv_list)):
        data = pd.read_csv(f'{file_dir}/data/{direc[d]}/{csv_list[c]}', index_col=0)
        plt.figure(figsize=(18,5))
        plt.plot(data.values)
        plt.xlim([0, len(data.values)])
        plt.xticks(ticks=[i for i in range(0, len(data.values), 24*12)],
                   labels=[data.index[i] for i in range(0, len(data.values), 24*12)],
                   rotation=90)
        plt.title(f'{direc[d]}/{csv_list[c]}')
        plt.tight_layout()
        plt.savefig(f'{file_dir}/figs/{direc[d]} - {csv_list[c][:-4]}.png')
        plt.close()
        freq = pd.to_datetime(data.index[1]) - pd.to_datetime(data.index[0])
        print(f'{file_dir}/{direc[d]}/{csv_list[c]} ~ shape:{data.shape}, freq:{freq}')


##############################
# with labeled data
with open('D:/2020_ETRI/NAB_data/NAB_data/labels/combined_labels.json') as json_data:
    labels = json.load(json_data)

for k in labels.keys():
    data = pd.read_csv(f'{file_dir}/data/{k}', index_col=0)
    # timeidx_ano = labels[k]
    idx_ano = []
    for l in labels[k]:
        idx_ano.append(np.where(data.index.values==l)[0][0])
    idx_ano = np.array(idx_ano)

    freq = pd.to_datetime(data.index[1])-pd.to_datetime(data.index[0])

    plt.figure(figsize=(18,5))
    plt.plot(data.values)
    plt.plot(idx_ano, data.iloc[idx_ano].values, 'X', color='orangered', markersize=10)

    plt.xlim([0, len(data.values)])
    plt.xticks(ticks=[i for i in range(0, len(data.values), 24*12)],
               labels=[data.index[i] for i in range(0, len(data.values), 24*12)],
               rotation=90)
    plt.legend(['data', 'anomaly'])
    plt.title(f'{k}, freq:{freq}')
    plt.tight_layout()
    plt.savefig(f'{file_dir}/figs/anomalychecked - {k.replace("/", " - ")[:-4]}.png')
    plt.close()

keys_nd = np.array(list(labels.keys())).reshape((len(labels.keys()),1))
values_nd = np.array(list(labels.values())).reshape((len(labels.values()),1))
k = keys_nd[29][0]