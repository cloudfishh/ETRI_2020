"""
Comparison between probabilistic forecast & nearest neighbor

2020. 10. 01. Thu.
Soyeong Park
"""
##############################
from funcs import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


##############################
'''
detection 할 때 기준을 세우는 방법은 2가지.
1. nearest neighbor 했을 때처럼 sigma 기준으로 자르기.
2. prob forecast 처럼 z-score 기준으로 자르기
원론적으로 둘이 같은 방식임. 그냥 계산 어케 하냐 차이죠.
z-score threshold로 통일하도록 합시다. 
nearest neighbor method를 z-score 구하는 방식으로 만드는거죠.
'''

##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 3
sigma = 4
# imputation_acc = True


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()
df['nan'] = chk_nan_bfaf(data_col)


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)

inj_mask = df['mask_inj'].copy()


##############################
# 3. get the sample with nearest neighbor method
idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]

sample_list = list()
for i in range(len(idx_list)):
    idx_target = idx_list[i]
    temp = nearest_neighbor()
    sample_list.append(temp)