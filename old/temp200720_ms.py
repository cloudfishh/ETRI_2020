from matplotlib import pyplot as plt
from funcs import *

file_dir = 'D:/2020_ETRI/data/label_data.csv'
data_raw = pd.read_csv(file_dir, index_col=0)
data_raw = data_raw.drop(columns=['Season', 'Weekday'])
data, nan_data = clear_head(data_raw)
list_code = random.sample(list(data.columns), 10)

for house_code in list_code:
    # house_code = '1338c9d0'
    # house_code = '0583b3a6'
    # house_code = random.choice(data_raw.columns)

    data_col = data[house_code]
    injected, inj_mark = inject_nan_acc(data_col, p_nan=1, p_acc=0.3)
    saving = pd.concat([injected, inj_mark], axis=1)
    saving.columns = [house_code, 'mask']
    saving.to_csv(f'injected_{house_code}.csv')
    print(f'injected_{house_code}.csv SAVED SUCCESSFULLY\n')
