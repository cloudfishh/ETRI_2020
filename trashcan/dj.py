# -*- coding: utf-8 -*-
"""
Created on Tue Jun  14 21:05:11 2020

@author: Dongju
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


Data_path = 'D:/2020_ETRI/data/'

# Daejeon_path = Data_path + "SG_data_대전_비식별화/"
Gwangju_path = Data_path + "SG_data_광주_비식별화/"
# Incheon_path = Data_path + "SG_data_인천_비식별화/"
Naju_path = Data_path + "SG_data_나주_비식별화/"
# Seoul_path = Data_path + "SG_data_서울_비식별화/"


def nan_distribution(load_vector):
    isnan_vec = np.isnan(load_vector)
    first_idx = np.argwhere(~np.isnan(load_vector))[0][0]

    nan_length_list = []

    i = first_idx
    while i < len(load_vector):

        if np.isnan(load_vector[i]):
            start = i
            while i < len(load_vector):
                if ~np.isnan(load_vector[i]):
                    end = i
                    break
                i += 1
            nan_length_list.append(end - start)
            continue
        i += 1

    return np.array(nan_length_list)


if __name__ == '__main__':
    List = os.listdir(Gwangju_path)  # List of .csv

    number_of_bldg = np.zeros(len(List))
    for i in range(len(List)):  # for one csv
        df = pd.read_csv(Gwangju_path + List[i], index_col=0)
        bldg_idx = df.columns[3:]
        load = df[bldg_idx].to_numpy()
        break

    nan_length_list = np.array([])
    for h in range(load.shape[1]):
        temp = nan_distribution(load[:, h])
        nan_length_list = np.append(nan_length_list, temp)

    plt.figure(figsize=(10, 3))
    plt.hist(nan_length_list)

    plt.figure(figsize=(10, 3))
    plt.hist(nan_length_list[nan_length_list > 1])

    plt.figure(figsize=(10, 3))
    plt.hist(nan_length_list[nan_length_list < 400], bins=100)

    plt.figure(figsize=(10, 3))
    plt.hist(nan_length_list[(nan_length_list > 2) * (nan_length_list < 48)], bins=20)

    plt.figure(figsize=(10, 3))
    plt.hist(nan_length_list[(nan_length_list > 2) * (nan_length_list < 80)], bins=100)
    # plt.xlim([])
