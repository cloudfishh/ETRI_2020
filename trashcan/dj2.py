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
    value_before = []
    value_after = []

    i = first_idx
    while i < len(load_vector):

        if np.isnan(load_vector[i]):
            start = i
            value_before.append(float(load_vector[i - 1]))
            while i < len(load_vector):
                if ~np.isnan(load_vector[i]):
                    end = i
                    value_after.append(float(load_vector[i]))
                    break
                i += 1
            nan_length_list.append(end - start)
            continue
        i += 1

    return np.array(nan_length_list), np.array(value_before), np.array(value_after)


if __name__ == '__main__':

    List = os.listdir(Gwangju_path)  # List of .csv
    plt.rcParams.update({'font.size': 15})
    number_of_bldg = np.zeros(len(List))
    # for i in tqdm(range(len(List))): # for one csv
    for i in [0]:
        df = pd.read_csv(Gwangju_path + List[i], index_col=0)
        bldg_idx = df.columns[2:]
        load = df[bldg_idx].to_numpy()

        nan_length_list, value_before, value_after = nan_distribution(load[:, 0])

        plt.figure(figsize=(10, 3))
        plt.hist(nan_length_list)
        plt.title('Distribution of NaN length - ' + List[i])
        plt.savefig(List[i] + " dist.png", dpi=800)

        plt.figure(figsize=(10, 3))
        plt.hist(nan_length_list[nan_length_list > 1])
        plt.title('Distribution of NaN length(>1) - ' + List[i])
        plt.savefig(List[i] + " dist_g1.png", dpi=800)

        plt.figure(figsize=(10, 3))
        plt.hist(nan_length_list[(nan_length_list < 25)], bins=24, range=(1, 25))
        plt.title('Distribution of NaN length(<25) - ' + List[i])
        plt.savefig(List[i] + " dist_1to24.png", dpi=800)
        plt.xticks([x for x in range(1, 25)])

        plt.figure(figsize=(10, 3))
        plt.scatter(nan_length_list, value_before)
        plt.title('Nan and the value before - ' + List[i] +
                  "(%.2f)" % np.corrcoef(nan_length_list, value_before)[0, 1])
        plt.savefig(List[i] + " corr_before.png", dpi=800)

        plt.figure(figsize=(10, 3))
        plt.scatter(nan_length_list, value_after)
        plt.title('Nan and the value after - ' + List[i] +
                  "(%.2f)" % np.corrcoef(nan_length_list, value_after)[0, 1])
        plt.savefig(List[i] + " corr_after.png", dpi=800)

        plt.figure(figsize=(10, 3))
        plt.scatter(nan_length_list[(nan_length_list < 25)], value_before[(nan_length_list < 25)])
        plt.title('Nan and the value before(<25) - ' + List[i] +
                  "(%.2f)" % np.corrcoef(nan_length_list[(nan_length_list < 25)], value_before[(nan_length_list < 25)])[
                      0, 1])
        plt.savefig(List[i] + " corr_before_l25.png", dpi=800)

        plt.figure(figsize=(10, 3))
        plt.scatter(nan_length_list[(nan_length_list < 25)], value_after[(nan_length_list < 25)])
        plt.title('Nan and the value after(<25) - ' + List[i] +
                  "(%.2f)" % np.corrcoef(nan_length_list[(nan_length_list < 25)], value_after[(nan_length_list < 25)])[
                      0, 1])
        plt.savefig(List[i] + " corr_after_l25.png", dpi=800)