"""
Accumulation detection with probabilistic forecast

2020. 09. 15. Tue.
Soyeong Park
"""
from funcs import *


'''
1. household 1개 데이터 불러오기
2. injection 하기
3. accumulation detection
    - probabilistic forecast로 판단하기?
    - candidate(before value)에 대해 prob. forecast를 하고 그 candidate가 위치한 interval에 따라 (z-score) 판단.
    - criteria? hmm?
    - 그럼 일단 다음주까진 candidate의 z-score 분석
4. imputation
5. result
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
# 1. 데이터 불러오기
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. accumulation detection
