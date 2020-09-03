from matplotlib import pyplot as plt
from funcs import *


##################################################
# load data
# loc_list = ['Gwangju', 'Naju', 'Daejeon', 'Seoul', 'Incheon']

# for loc in range(1):
# loc = 0

# loc_str = loc_list[loc]
# list_apt = set_dir(loc)                   # 0=gwangju; 1=naju

file_dir = 'D:/2020_ETRI/label_data.csv'
data_raw = load_labeled()
data, nan_idx = clear_head(data_raw)
households = data.shape[1]

# for apt in range(len(list_apt)):
#     apt_dir = f'D:/2020_ETRI/fig_gwangju/{list_apt[apt][:10]}'
#     if not os.path.isdir(apt_dir):
#         os.mkdir(apt_dir)
#
#     data = load_household(list_apt, apt)
#     households = data.shape[1]

idx1 = 6100-24*11  # if apt != 6 else 18192 + 24*12
idx2 = idx1 + 24*21

house = '0098d3ee'
data_col = data[house].iloc[idx1:idx2]


for house in range(households):
    data_col = data.iloc[idx1:idx2, house]
    plt.figure(figsize=(18, 6))
    plt.plot(data_col)
    plt.xticks(ticks=[i for i in range(0, len(data_col), 24)], rotation=90)
    plt.xlim([data_col.index[0], data_col.index[-1]])
    plt.ylim([-0.4, 2.5])
    plt.tight_layout()
    plt.grid(alpha=0.3)

#     plt.savefig(f'{apt_dir}/{data.columns[house]}.png')
#     plt.close()
#     print(f'{house} ', end='')
# print(f'{list_apt[apt][:10]} SAVED SUCCESSFULLY')
