from funcs import *


test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 3
sigma = 5
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)

p_nan, p_acc = 1, 0.5



