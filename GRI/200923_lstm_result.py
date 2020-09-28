from scipy import io
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def make_date_index(y1, y2):
    # make datetime index
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
    date_index = pd.DatetimeIndex(date_index_str, freq='1H')
    return date_index


def make_date_index_q(y1, y2):
    # make datetime index
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
                    for q in range(4):
                        date_index_str.append('%d-%02d-%02d %02d:%02d:00' % (y, m+1, d+1, h, q*15))
    date_index = pd.DatetimeIndex(date_index_str, freq='15T')
    return date_index


def load_calendar(y1, y2):
    dir_cal = 'D:/PycharmProjects/ETRI_2020/GRI'
    cal = pd.DataFrame([])
    for y in range(y1, y2+1):
        cal = pd.concat([cal, pd.read_csv(f'{dir_cal}/calendar{y}.csv', header=None)])

    calendar = pd.DataFrame(np.empty([len(cal)*24, ]), index=make_date_index(y1, y2))
    for r in range(len(cal)):
        for rr in range(24):
            calendar.iloc[r*24+rr] = cal.iloc[r][4]
    calendar.columns = ['holiday']
    return calendar


def load_dataset(category='Apt'):
    # Apt(4), Complex(5), Dept(4), Hospital(4), Market(3), Office(4), School(15)
    make_date_index_q(2016, 2016)
    data_raw = io.loadmat(f'D:/PycharmProjects/ETRI_2020/GRI/Dat_{category}.mat')
    buildings = list(data_raw.keys())[3:]

    data = pd.DataFrame([], index=make_date_index_q(2016, 2016))
    data_1h = pd.DataFrame([], index=make_date_index(2016, 2016))
    for b in range(len(buildings)):
        s = np.empty([366*24,])
        for j in range(366*24):
            s[j] = sum(data_raw[buildings[b]].reshape(366*24*4,)[4*j:4*(j+1)])
        data[buildings[b]] = data_raw[buildings[b]].reshape(366*24*4, )
        data_1h[buildings[b]] = s

    data_1h['month'] = data_1h.index.month.to_numpy()
    data_1h['day'] = data_1h.index.day.to_numpy()
    data_1h['hour'] = data_1h.index.hour.to_numpy()
    data_1h['minute'] = data_1h.index.minute.to_numpy()
    data_1h['calendar'] = load_calendar(2016, 2016)

    data['month'] = data.index.month.to_numpy()
    data['day'] = data.index.day.to_numpy()
    data['hour'] = data.index.hour.to_numpy()
    data['minute'] = data.index.minute.to_numpy()

    cal_15m = load_calendar(2016, 2016)
    cal_15m.loc[pd.Timestamp('2017-01-01 00:00:00', freq='H')] = 1
    cal_15m = cal_15m.asfreq('15T')
    cal_15m = cal_15m.drop(index=pd.Timestamp('2017-01-01 00:00:00', freq='H'))
    cal_15m = cal_15m.fillna(method='ffill')

    data['calendar'] = cal_15m

    return data, data_1h


def mape(A, F):
    return 100 / len(A) * np.sum((np.abs(A - F))/A)


cat_list = ['Apt', 'Complex', 'Dept', 'Hospital', 'Office']
cat_list_num = [4, 5, 4, 4, 4]

n_input, n_out, timestep = 24*7, 24, 24
# cat, bldg = 'Complex', 0

dataset, dataset_1h = pd.DataFrame([]), pd.DataFrame([])
for cat in cat_list:
    dataset_temp, dataset_1h_temp = load_dataset(category=cat)
    dataset = pd.concat([dataset, dataset_temp.iloc[:, :-5]], axis=1)
    dataset_1h = pd.concat([dataset_1h, dataset_1h_temp.iloc[:, :-5]], axis=1)
dataset = pd.concat([dataset, dataset_temp.iloc[:, -5:]], axis=1)
dataset_1h = pd.concat([dataset_1h, dataset_1h_temp.iloc[:, -5:]], axis=1)

dataset_1h_temp = dataset_1h.iloc[-1560:, :]
result = pd.read_csv('GRI/200923_lstm_result.csv', index_col=0)


print('APT Total') # 0~3
y_true = dataset_1h_temp.iloc[:, 0:4].sum(axis=1)
y_pred = result.iloc[:, 0:4].sum(axis=1)
print(f'RMSE: {mean_squared_error(y_true, y_pred)**(1/2)}')
print(f' MSE: {mean_squared_error(y_true, y_pred)}')
print(f' MAE: {mean_absolute_error(y_true, y_pred)}')
print(f'MAPE: {mape(y_true, y_pred)}\n')

print('Complex + Dept') # 4~12
y_true = dataset_1h_temp.iloc[:, 4:13].sum(axis=1)
y_pred = result.iloc[:, 4:13].sum(axis=1)
print(f'RMSE: {mean_squared_error(y_true, y_pred)**(1/2)}')
print(f' MSE: {mean_squared_error(y_true, y_pred)}')
print(f' MAE: {mean_absolute_error(y_true, y_pred)}')
print(f'MAPE: {mape(y_true, y_pred)}\n')

print('Hospital + Office')  # 13~20
y_true = dataset_1h_temp.iloc[:, 13:21].sum(axis=1)
y_pred = result.iloc[:, 13:21].sum(axis=1)
print(f'RMSE: {mean_squared_error(y_true, y_pred)**(1/2)}')
print(f' MSE: {mean_squared_error(y_true, y_pred)}')
print(f' MAE: {mean_absolute_error(y_true, y_pred)}')
print(f'MAPE: {mape(y_true, y_pred)}\n')

