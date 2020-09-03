import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt


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
    feature = ['cloud', 'cloudlow', 'humid', 'rain', 'setemp', 'snow', 'solarirr', 'surftemp', 'temp', 'windirec', 'windspeed']

    load = pd.DataFrame()
    for b in building:
        temp = pd.read_csv(f'{dir}/2018GIST_{b}.csv', header=None)
        v = temp.stack(dropna=False).reset_index()[0]
        new = np.array([])
        for i in range(0, 96*365, 4):
            new = np.append(new, sum(v[i:i+4]))
        load[b] = new

    weather = pd.DataFrame()
    for f in feature:
        temp = pd.read_csv(f'{dir}/2018weather_{f}.csv', header=None)
        v = temp.stack(dropna=False).reset_index()[0]
        weather[f] = v
    return load, weather


##############################
# get importance scores
# building = ['dasan', 'dorm_A', 'dorm_B', 'f_apt', 'GIST_A', 'GIST_B', 'GIST_C', 'plant', 'RISE', 'SU2']
# building = ['dasan', 'dorm_A', 'f_apt', 'RISE']
building = ['f_apt', 'RISE']
load, weather = load_data()
# test_bldg = 'dasan'

for test_bldg in building:
    ##############################
    # define dataset
    X, y = weather.fillna(0).values, load[test_bldg].fillna(0).values
    corr = np.array([])
    for i in range(X.shape[1]):
        c = np.corrcoef(X[:, i], y)
        corr = np.append(corr, c[0, 1])

    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    print(f'IMPORTANCE SCORE - {test_bldg}')
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    print(f'CORRELATION COEFFICIENT - {test_bldg}')
    for i, v in enumerate(corr):
        print('Feature: %0d, Score: %.5f' % (i, v))
    print('\n')

    # plot feature importance
    plt.figure()
    # plt.bar([x for x in range(len(importance))], importance, label='importance', width=0.5)
    plt.bar([x-0.2 for x in range(len(importance))], importance, label='importance', width=0.5)
    plt.bar([x+0.2 for x in range(len(corr))], corr, label='correlation', width=0.5)
    plt.title(f'importance score - {test_bldg}')
    plt.xticks(ticks=range(len(weather.columns)), labels=weather.columns.values, rotation=45)
    plt.legend(['importance score', 'correlation coef.'])
    # plt.ylim([0, 0.2])
    plt.tight_layout()
    plt.savefig(f'D:/2020_GRI/200727/ImportanceScore+correlation_{test_bldg}.png')
    plt.show()


##############################
# correlation - temp & humid
# corr = np.corrcoef(homes.iloc[:,1][interest_point], load[interest_point])
test_bldg = 'dasan'
data_raw = pd.DataFrame([weather.iloc[:, 8], load[test_bldg]]).transpose()
data = data_raw.dropna()
plt.figure()
plt.plot(data['temp'], data['dasan'], '.')
