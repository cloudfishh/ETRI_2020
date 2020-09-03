from matplotlib import pyplot as plt
from funcs import *

'''
# # # # # SIMULATION HOUSE INFO # # # # #
# lowest nan rate (= # of nan / data length)
# loc = 0, Gwangju
# apt = 3, 2914011900
# hom = 20, 2004a00b, 마지막에 nan 안 붙은 걸로 고름
'''

'''
# # # # # SIMULATION HOUSE INFO - 2nd # # # # #
# loc = 3, Seoul
# apt = 0 
# hom = 96
'''

loc, apt, house = 3, 0, 96

##################################################
# load data
loc_list = ['Gwangju', 'Naju', 'Daejeon', 'Seoul', 'Incheon']
list_apt = set_dir(loc)                              # 0=gwangju; 1=naju
data_raw = load_household(list_apt, apt)
data, nan_data = clear_head(data_raw)
data_col = data.iloc[:, house]


##################################################
# injection
p_nan, p_acc = 0.5,  0.5
nan_count = count_nan_len(data_col)
nan_list = np.unique(nan_count, return_counts=True)

n_inj_list = nan_list[1][0:3] * p_nan
n1, n2, n3 = int(n_inj_list[0]), int(n_inj_list[1]), int(n_inj_list[2])
n_inj_sum = n1 + n2 + n3

# nan, bf, af 아닌 인덱스만 뽑아낸 list를 만들고
# 그 list의 index로 랜덤샘플링
# 간격 최소4 이어야 하니까 list의 index를 /4 해서 랜덤으로 뽑고 거기에 3을 곱해주면 되겟죵
nan_bool = chk_nan_bfaf(data_col)
candidates = np.array(np.where(nan_bool.values == 0))

random.seed(0)
rand = random.sample(range(1, int(len(candidates[0]) / 6)), k=n_inj_sum)
rand_rev = [6 * i for i in rand]
idx_inj = list(candidates[0][rand_rev])
idx_inj.sort()

random.seed(1)
idx_acc = random.sample(idx_inj, k=int(n_inj_sum * p_acc))
idx_acc = [i - 1 for i in idx_acc]
idx_acc.sort()

len_list = np.ones(n1)
len_list = np.append(len_list, np.ones(n2) * 2)
len_list = np.append(len_list, np.ones(n3) * 3)
random.seed(2)
random.shuffle(len_list)

inj_mask = nan_bool.copy()
for i in range(len(idx_inj)):
    idx = int(idx_inj[i])
    l_nan = int(len_list[i])
    inj_mask[idx:idx + l_nan] = 2
inj_mask[idx_acc] = 3

injected = data_col.copy()
s, k = 0, 0
for j in range(len(inj_mask)):
    if inj_mask[j] == 3:  # inj mask가 3이면
        while inj_mask[j + k] > 1:  # k는 이후의 mask, mask가 1 이하가 되기 전까지 돌린다
            s += injected[j + k]  # s는 temp sum
            k += 1  # k length니까 하나 더해주고
        injected[j] = s  # while 끝나면 injected에 accumulation 넣어주고
        s = 0  # s, k 초기화
        k = 0
injected[inj_mask == 2] = np.nan

plt.figure()
plt.xticks([])
plt.plot(data_col, linewidth=2)
plt.plot(injected, linewidth=1)
plt.plot(data_col[inj_mask == 2], '.', markersize=5)
plt.plot(injected[inj_mask == 3], 'x', markersize=8)
plt.legend(['original data', 'injected data', 'injected NaN', 'injected acc.'])
plt.xticks(ticks=[i for i in range(0, len(data_col), 24*7)], labels=[i for i in range(0, len(data_col), 24*7)])
plt.show()
