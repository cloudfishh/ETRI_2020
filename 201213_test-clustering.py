"""
Anomaly detection with clustering

2020. 12. 13. Sun.
Soyeong Park
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import os


nan_len = 5
df = pd.read_csv('D:/202010_energies/201207_result_aodsc+owa_spline-rev-again.csv', index_col=0)

house_list = np.unique(df['house'].values.astype('str'))

house = house_list[0]

for house in house_list:
    df_temp = df[df['house']==house]

