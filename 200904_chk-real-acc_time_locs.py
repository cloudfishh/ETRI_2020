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


def analyze_real_accumulation(data_col):
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

    print(f'    total: {sum(df["candidate"])}')
    print(f'z-score 3: {sum( (df["z-score"]>3)&(df["candidate"]==1) )}')
    print(f'z-score 4: {sum( (df["z-score"]>4)&(df["candidate"]==1) )}')
    return sum(df["candidate"]), sum((df["z-score"]>3)&(df["candidate"]==1)), sum((df["z-score"]>4)&(df["candidate"]==1))


if __name__=="__main__":
    loc_list = ['Gwangju', 'Naju', 'Daejeon', 'Seoul', 'Incheon']
    for l in range(5):
        loc = loc_list[l]
        list_apt = set_dir(l)
        for apt in range(len(list_apt)):
            data_raw = load_household(list_apt, idx=apt)
            data, nan_data = clear_head(data_raw)

            list_ratio = pd.DataFrame([], index=data_raw.columns, columns=['total', 'chk_3', 'chk_4', 'ratio3'])
            list_test_house = data_raw.columns[:30] if len(data_raw.columns)>30 else data_raw.columns
            for test_house in list_test_house:
                data_col = data[test_house]
                t, z3, z4 = analyze_real_accumulation(data_col)
                list_ratio.loc[test_house] = [t, z3, z4, z3/t]
                print(f'* {np.where(list_test_house == test_house)[0][0]:02}_{test_house} DONE')
            list_ratio.to_csv(f'D:/PycharmProjects/ETRI_2020/csv/200907_realacc_z_{loc}_{apt}.csv')
            print(f'** APARTMENT {loc}_{apt} DONE')
        print(f'*** LOCATION {loc} DONE')
