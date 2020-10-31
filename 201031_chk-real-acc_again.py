"""
Analyze the real accumulation
 - classify the value before NaNs by mean and sigma from same time points
2020. 09. 05. Sat.
Soyeong Park
"""
"""
1. 데이터 로드, nan 유무 저장
2. before 유무 저장, nan length 저장
3. before이면 해당 value의 시간대 앞뒤 1달치 mean & sigma detection
ps. 그냥 check accumulation function 적용하면 될 것 같은데...?
"""
from funcs import *
from matplotlib import pyplot as plt
import matplotlib.patches as patches


##############################
# 1. 데이터 로드, nan 유무 저장
# test_house = '68181c16'
# test_house = '1dcb5feb'
test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(6,4), dpi=400)


for test_house in test_house_list:
    data_raw = load_labeled()
    data, nan_data = clear_head(data_raw)
    data_col = data[test_house]

    idx_nan = np.where(np.isnan(data_col.values) == True)[0]

    df = pd.DataFrame([],
                      index=pd.DatetimeIndex(data_col.index),
                      columns=['values', 'nan', 'nan_bfaf', 'candidate', 'nan_len', 'result', 'mean', 'std', 'sample_len', 'z-score'])
    df['values'] = data_col.copy()
    df['nan'] = np.isnan(data_col.values)
    df['nan_bfaf'] = chk_nan_bfaf(data_col)


    ##############################
    # 2. candidate = before유무, nan_len = nan 길이
    num = 0
    cand, nan_len = np.zeros(df.shape[0]), np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        if df['nan'][i] == True:
            num += 1
        elif (df['nan'][i] == False) & (num != 0):
            cand[i-num-1] = 1
            nan_len[i-num-1] = num
            num = 0
    df['candidate'] = cand
    df['nan_len'] = nan_len


    # b = pd.DataFrame([])
    # b['idx'] = np.where(df['nan']==1)[0].astype('int')
    # b['time'] = df.index[b['idx'].values]


    ##############################
    # 3. candidate == 1이면 해당 시간대 mean/sigma 찾아서 유무 판별
    result, mean, std, sample_len = np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(df.shape[0])
    for idx in np.where(df['candidate'] == 1)[0]:
        if idx < 30*24:
            temp = df.iloc[:idx+30*24]
        elif idx > df.shape[0]-30*24:
            temp = df.iloc[idx-30*24:]
        else:
            temp = df.iloc[idx-30*24:idx+30*24]
        ttemp = temp['values'][(temp['nan_bfaf']==0)&(temp.index.hour==df.index[idx].hour)]
        m, s = ttemp.mean(), ttemp.std()
        # m = temp['values'][np.where(temp['nan_bfaf']==0)[0]].mean()
        # s = temp['values'][np.where(temp['nan_bfaf']==0)[0]].std()
        # if not m-3*s < df['values'][idx] < m+3*s:
        if not df['values'][idx] < m+3*s:
            result[idx], mean[idx], std[idx], sample_len[idx] = 1, m, s, ttemp.shape[0]
    df['result'], df['mean'], df['std'], df['sample_len'] = result, mean, std, sample_len


    ##############################
    # 3 another. adaptive z-score
    result, mean, std, sample_len, z_score = \
        np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        if np.isnan(df['values'][i]) == False:
            if i < 30*24:
                temp = df.iloc[:i+30*24]
            elif i > df.shape[0]-30*24:
                temp = df.iloc[i-30*24:]
            else:
                temp = df.iloc[i-30*24:i+30*24]
            ttemp = temp['values'][(temp['nan_bfaf']==0)&(temp.index.hour==df.index[i].hour)]
            m, s = ttemp.mean(), ttemp.std()
            z = (df['values'][i]-m)/s

            if not df['values'][i] < m+3*s:
                result[i], sample_len[i] = 1, len(ttemp)
        else:
            z, m, s = np.nan, np.nan, np.nan
        z_score[i], mean[i], std[i] = z, m, s


    df['result'], df['mean'], df['std'], df['sample_len'], df['z-score']\
        = result, mean, std, sample_len, z_score

    # print(f'    total: {sum(df["candidate"])}')
    # print(f'z-score 3: {sum( (df["z-score"]>3)&(df["candidate"]==1) )}')
    # print(f'z-score 4: {sum( (df["z-score"]>4)&(df["candidate"]==1) )}')
    print(f'***** {test_house}, z-score>3 depending on nan_len')
    for i in range(1, 21):
        if len(df["nan_len"][df["nan_len"]==i]) != 0:
            print(f'   nan_len={i:02}: { len(df["nan_len"][(df["nan_len"]==i)&(df["z-score"]>3)])/len(df["nan_len"][df["nan_len"]==i]) }')
        else:
            print(f'   nan_len={i:02}: len( z-score>3 ) = 0')
    print('\n')

    ##############################
    # 4. plot
    # 플롯 어케 해야할지?? y축을 standard score로 놓으라고 하셨음.
    # plt.figure(figsize=(7.5,5))
    plt.plot(df['nan_len'][df['nan_len']!=0], df['z-score'][df['nan_len']!=0], '.')
    plt.xlim([-1, 25])
    plt.xlabel('length of NaNs')
    plt.ylabel('z-score')
    # plt.title(f'{test_house}')
    # plt.tight_layout()
    # break


# fig, ax = plt.subplots(figsize=(6,4), dpi=100)
# plt.xlim([-1, 25])
plt.ylim([-13, 55])
plt.grid(alpha=0.2)
ell = patches.Ellipse(xy=(8.5, 25), width=60, height=10, angle=80, linewidth=1, facecolor='b', alpha=0.2)
ax.add_patch(ell)
plt.legend(['house 1', 'house 2', 'house 3', 'house 4', 'house 5'])
plt.tight_layout()
plt.savefig('Fig_observed.pdf')

# plt.plot(plot_df['num'], plot_df['val'], '.')
# plt.plot(plot_mean['num'], plot_mean['val'], color='tomato', linewidth=1)
# plt.plot(plot_hor['num'], plot_hor['val'], color='seagreen', linewidth=1)
# plt.fill_between(plot_hor['num'],
#                  # plot_mean['val'] - plot_df['val'][plot_df['num'] == 0].values.std()*3,
#                  plot_hor['val'] - plot_df['val'][plot_df['num'] == 0].values.std()*3,
#                  plot_hor['val'] + plot_df['val'][plot_df['num'] == 0].values.std()*3, color='seagreen', alpha=0.2)
# plt.xlim([-1, 25])
# # plt.ylim([-0.2, 3.7])
# # plt.xticks(ticks=[i for i in range(0, 25)])
# plt.xlabel('length of NaNs')
# plt.ylabel('Value (Power [kW])')
# plt.title(f'{test_house}')
# plt.tight_layout()
# # plt.savefig(f'D:/PycharmProjects/ETRI_2020/chkrealacc_{loc}_apt{apt}_{test_house}.png')
# plt.close()


##############################
# counting accumulation candidates
# count = dict()
# for i in range(plot_df.shape[0]):
#     if plot_df['val'][i] > plot_hor['val'][0] + plot_df['val'][plot_df['num'] == 0].values.std()*3:
#         try:
#             count[plot_df['num'][i]] += 1
#         except KeyError:
#             count[plot_df['num'][i]] = 1
#
# for keys, values in sorted(count.items()):
#     if keys != 0:
#         print(f'{keys}: {values}')
