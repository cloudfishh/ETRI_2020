import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# from pyinform.mutualinfo import mutual_info
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm



def make_date_index(y1, y2):
    # make datetime index
    # global date_index
    date_index_str = []
    day_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_list_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for y in range(y1, y2+1):
        if y % 4 == 0:
            d_list = day_list_leap
        else:
            d_list = day_list
        for m in range(12):
            for d in range(d_list[m]):
                for h in range(24):
                    date_index_str.append('%d-%02d-%02d %02d:00:00' % (y, m+1, d+1, h))
    date_index = pd.DatetimeIndex(date_index_str, freq='H')
    return date_index


def load_data():
    dir = 'D:/2020_GRI/200727'
    building = ['dasan', 'dorm_A', 'dorm_B', 'f_apt', 'GIST_A', 'GIST_B', 'GIST_C', 'plant', 'RISE', 'SU2']
    # feature = ['cloud', 'cloudlow', 'humid', 'rain', 'snow', 'solarirr', 'temp', 'windspeed']
    # feature = ['cloud', 'humid', 'rain', 'solarirr', 'temp', 'windspeed']
    feature = ['humid', 'rain', 'solarirr', 'temp', 'windspeed']

    load = pd.DataFrame()
    for b in building:
        temp = pd.read_csv(f'{dir}/2018GIST_{b}.csv', header=None)
        v = temp.transpose().stack(dropna=False).reset_index()[0]
        new = np.array([])
        for i in range(0, 96*365, 4):
            new = np.append(new, sum(v[i:i+4]))
        load[b] = new

    weather = pd.DataFrame()
    for f in feature:
        temp = pd.read_csv(f'{dir}/2018weather_{f}.csv', header=None)
        v = temp.transpose().stack(dropna=False).reset_index()[0]
        weather[f] = v
    # weather.columns = ['전운량', '중하층운량', '습도', '강수량', '이슬점온도', '적설량', '일사', '지면온도', '기온', '풍향', '풍속']
    return load, weather


##############################
# matplotlib font setting
font_dir = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_dir).get_name()
plt.rcParams.update({'font.size': 13,
                     'font.family': font_name})

bar_width = 0.5
# bar_width = 0.5


##############################
# get importance scores
# building = ['dasan', 'dorm_A', 'dorm_B', 'f_apt', 'GIST_A', 'GIST_B', 'GIST_C', 'plant', 'RISE', 'SU2']
# building = ['dasan', 'dorm_A', 'f_apt', 'RISE']
# building = ['dorm_A', 'dasan']

# building = ['GIST_A', 'GIST_B', 'plant', 'SU2']
building = ['plant', 'SU2']
# building = ['GIST_C', 'plant']
load, weather = load_data()
# test_bldg = 'dasan'


# weather_raw = weather_raw.fillna(0).values
weather_raw = weather.interpolate(method='linear').fillna(0).values
weather_nor = np.zeros([weather_raw.shape[0], weather_raw.shape[1]])
for col in range(weather_raw.shape[1]):
    # weather_nor[:, col] = (weather_raw[:, col] - weather_raw[:, col].mean())/weather_raw[:, col].std()
    weather_nor[:, col] = (weather_raw[:, col] - weather_raw[:, col].min())/(weather_raw[:, col].max() - weather_raw[:, col].min())


for t in range(len(building)):
    test_bldg = building[t]

    ##### CORRELATION
    # X, y = weather_raw, (load[test_bldg].fillna(0).values - load[test_bldg].fillna(0).values.mean())/load[test_bldg].fillna(0).values.std()
    X, y = weather_nor, load[test_bldg].fillna(0).values
    corr = np.array([])
    for i in range(X.shape[1]):
        c = np.corrcoef(X[:, i], y)
        corr = np.append(corr, c[0, 1])

    ##### GINI IMPORTANCE
    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_

    ##### MUTUAL INFORMATION
    # y_nor = (y-y.min())/(y.max()-y.min())
    # mutual = np.array([])
    # for i in range(X.shape[1]):
    #     if i != 6:
    #         m = mutual_info(weather_nor[:, i], y)
    #         mutual = np.append(mutual, m)
    #     else:
    #         mutual = np.append(mutual, 0)
    mutual = mutual_info_regression(weather_nor, y)

    # summarize feature importances
    # print(f'IMPORTANCE SCORE - {test_bldg}')
    # for i, v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    # print(f'CORRELATION COEFFICIENT - {test_bldg}')
    # for i, v in enumerate(corr):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    # print('\n')
    print(f'GINI IMPORTANCE RANK - {test_bldg}')
    print(pd.Series(importance).rank(ascending=False).values)
    print(f'CORRELATION COEFFICIENT RANK - {test_bldg}')
    print(pd.Series(corr).rank(ascending=False).values)
    print(f'MUTUAL INFORMATION RANK - {test_bldg}')
    print(pd.Series(mutual).rank(ascending=False).values)
    print('\n')


    # plot feature importance
    # plt.bar([x for x in range(len(importance))], importance, label='importance', width=0.5)
    plt.rcParams.update({'font.size': 13})
    plt.figure(1)
    plt.bar([x+(-0.2+0.4*t) for x in range(len(importance))], importance, label=test_bldg, width=0.4)
    # plt.bar([-len(building)*bar_width+t*bar_width+x for x in range(len(importance))], importance, label=test_bldg, width=bar_width)
    # plt.title('Importance Score')
    # plt.xticks(ticks=[x for x in range(weather_raw.shape[1])], labels=['전운량', '습도', '강수량', '일사', '기온', '풍속'], rotation=45)
    plt.xticks(ticks=[x for x in range(weather_raw.shape[1])], labels=['습도', '강수량', '일사', '기온', '풍속'], rotation=45)
    plt.ylim([0.05, 0.5])
    plt.ylabel('Gini importance')
    plt.legend()
    plt.tight_layout()

    plt.figure(2)
    plt.bar([x+(-0.2+0.4*t) for x in range(len(corr))], abs(corr), label=test_bldg, width=0.4)
    # plt.bar([-len(building)*bar_width+t*bar_width+x for x in range(len(corr))], abs(corr), label=test_bldg, width=bar_width)
    # plt.bar([-len(building)*bar_width+t*bar_width+x for x in range(len(corr))], corr, label=test_bldg, width=bar_width)
    # plt.title('Correlation Coefficient')
    # plt.xticks(ticks=[x for x in range(weather_raw.shape[1])], labels=['전운량', '습도', '강수량', '일사', '기온', '풍속'], rotation=45)
    plt.xticks(ticks=[x for x in range(weather_raw.shape[1])], labels=['습도', '강수량', '일사', '기온', '풍속'], rotation=45)
    plt.ylabel('Correlation coefficient')
    plt.legend()
    plt.tight_layout()

    plt.figure(3)
    plt.bar([x+(-0.2+0.4*t) for x in range(len(mutual))], mutual, label=test_bldg, width=0.4)
    # plt.bar([-len(building)*bar_width+t*bar_width+x for x in range(len(mutual))], mutual, label=test_bldg, width=bar_width)
    # plt.title('Correlation Coefficient')
    # plt.xticks(ticks=[x for x in range(weather_raw.shape[1])], labels=['전운량', '습도', '강수량', '일사', '기온', '풍속'], rotation=45)
    plt.xticks(ticks=[x for x in range(weather_raw.shape[1])], labels=['습도', '강수량', '일사', '기온', '풍속'], rotation=45)
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.tight_layout()

# plt.legend(['importance score', 'correlation coef.'])
# plt.ylim([0, 0.2])
plt.tight_layout()
plt.savefig(f'D:/2020_GRI/200727/ImportanceScore+correlation_{test_bldg}.png')
plt.show()


##############################
# correlation - temp & humid
# corr = np.corrcoef(homes.iloc[:,1][interest_point], load[interest_point])
test_bldg = 'dasan'
data_raw = pd.DataFrame([weather_nor.iloc[:, 8], load[test_bldg]]).transpose()
data = data_raw.dropna()
plt.figure()
plt.plot(data['temp'], data['dasan'], '.')
