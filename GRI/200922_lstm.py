"""
load prediction with LSTM
2020. 09. 22. Tue
Soyeong Park
"""
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
# import keras.backend as K
from math import sqrt
import time


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
    dir_cal = 'GRI'
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
    data_raw = io.loadmat(f'GRI/Dat_{category}.mat')
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


def split_dataset(dataset, timestep):
    # split into standard weeks
    data_raw = dataset.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data_raw)

    # idx = dataset.columns.get_loc(target)
    # train, test = data[:28896], data[28896:]
    train, test = data[:7224], data[7224:]

    # restructure into windows of weekly data
    train = np.array(np.split(train, int(len(train) / timestep)))
    test = np.array(np.split(test, int(len(test) / timestep)))
    return train, test


def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step (x) 24steps(=1day)
        in_start += 24
    return np.array(X), np.array(y)


def lstm_model(train_x, n_out):
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(32))
    model.add(Dense(n_out))
    return model


def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def mape(A, F):
    return 100 / len(A) * np.sum((np.abs(A - F))/A)


##############################
if __name__=='__main__':
    # Apt(4), Complex(5), Dept(4), Hospital(4), Market(3), Office(4), School(15)
    # cat_list = ['Apt', 'Complex', 'Dept', 'Hospital', 'Market', 'Office', 'School']
    cat_list = ['Apt', 'Complex']

    n_input, n_out, timestep = 24*7, 24, 24
    # cat, bldg = 'Complex', 0

    dataset, dataset_1h = pd.DataFrame([]), pd.DataFrame([])
    for cat in cat_list:
        dataset_temp, dataset_1h_temp = load_dataset(category=cat)
        dataset = pd.concat([dataset, dataset_temp.iloc[:, :-5]], axis=1)
        dataset_1h = pd.concat([dataset_1h, dataset_1h_temp.iloc[:, :-5]], axis=1)
    dataset = pd.concat([dataset, dataset_temp.iloc[:, -5:]], axis=1)
    dataset_1h = pd.concat([dataset_1h, dataset_1h_temp.iloc[:, -5:]], axis=1)

    selected_col = input('합산할 컬럼 번호 입력 ( ex) 0 1 2 ): ', ).split()
    s_col = dataset_1h.iloc[:, int(selected_col[0])]
    for col in range(len(selected_col)-1):
        s_col = s_col + dataset_1h.iloc[:, int(selected_col[col+1])]

    df_temp = pd.concat([s_col, dataset_1h.iloc[:, -5:]], axis=1)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    _ = target_scaler.fit_transform(pd.DataFrame(df_temp.iloc[:, 0]))
    df_mean, df_std = df_temp.mean(), df_temp.std()

    train, test = split_dataset(df_temp, timestep=timestep)
    train_x, train_y = to_supervised(train, n_input=n_input, n_out=n_out)

    ##############################
    # parameters, fitting
    verbose, epochs, batch_size = 2, 50, 64
    model = lstm_model(train_x, n_out=n_out)
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt)
    start_time = time.time()
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, shuffle=False)
    print(f'    ** ELAPSED TIME {time.time()-start_time:.3f} secs')
    # prediction
    hist_data = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        d = np.array(hist_data)
        d = d.reshape((d.shape[0]*d.shape[1], d.shape[2]))
        input_x = d[-n_input:, :]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        yhat_sequence = model.predict(input_x, verbose=2)
        predictions.append(yhat_sequence)
        hist_data.append(test[i, :])

    pred = np.array(predictions)
    # score, scores = evaluate_forecasts(test[:, :, 0], pred)

    y_true = test[:, :, 0].reshape((test.shape[0]*test.shape[1],1))
    y_pred = pred.reshape((pred.shape[0]*pred.shape[2],1))
    y_true = target_scaler.inverse_transform(y_true)
    y_pred = target_scaler.inverse_transform(y_pred)

    # plot_index = dataset.index[28896:]
    plot_index = dataset_1h.index[7224:]

    result = pd.DataFrame(y_true, index=plot_index, columns=['true'])
    result['pred'] = y_pred
    # result.to_csv(f'{cat}-{selected_col}.csv')

    print(f'{cat} - {selected_col} COMPLETED')
    print(f'RMSE: {mean_squared_error(y_true, y_pred)**(1/2)}')
    print(f' MSE: {mean_squared_error(y_true, y_pred)}')
    print(f' MAE: {mean_absolute_error(y_true, y_pred)}')
    print(f'MAPE: {mape(y_true, y_pred)}\n')

    plt.figure(figsize=(15, 5))
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.xticks(ticks=[i for i in range(0, len(y_true), 120)],
               labels=[str(plot_index[i])[:-3] for i in range(0, len(y_true), 120)],
               rotation=45)
    plt.xlim([0, len(y_true)])
    plt.legend(['observation', 'prediction'])
    plt.title(f'{cat_list} - {selected_col}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{cat} - {dataset.columns[0]}.png')
    # plt.close()
