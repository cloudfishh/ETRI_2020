"""
Accumulation detection with similar days
 and fwd-bwd joint imputation with AR
- length of NaN = 5 test

2020. 11. 11. Wed.
Soyeong Park
"""
##############################
from funcs import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time


##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 5


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]


plt.figure(figsize=(12, 5), dpi=400)
# for test_house in ['68181c16', '1dcb5feb', '2ac64232']:
#     data_col = data[test_house]
for i in range(5,8):
    test_house = data.columns[i]
    data_col = data[test_house]
    plt.plot(data_col.values, alpha=.5)
plt.xticks(ticks=[x for x in range(0, len(data_col), 24*30*5)],
           labels=[data_col.index[x][:10] for x in range(0, len(data_col), 24*30*5)], rotation=45)
plt.xlim([0, len(data_col)])
plt.ylim([-0.4, 6])
plt.xlabel('Date')
plt.ylabel('Power [kW]')
plt.legend(['Observed data - house1', 'Observed data - house2', 'Observed data - house3'], loc='upper right')
plt.tight_layout()
plt.savefig('Fig_profile.pdf', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format='pdf',
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)