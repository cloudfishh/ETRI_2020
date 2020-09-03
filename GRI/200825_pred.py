from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import keras.backend as K
from math import sqrt


def make_date_index(y1, y2):
    # make datetime index
    global date_index
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


def load_dataset(type='Apt'):
    # Apt(4), Complex(5), Dept(4), Hospital(4), Market(3), Office(4), School(15)
    make_date_index(2016, 2016)
    data_raw = io.loadmat(f'GRI/Dat_{type}.mat')
    buildings = list(data_raw.keys())[3:]
    # for i in range(len(buildings)):
    #     plt.figure()
    #     plt.plot(data_raw[buildings[i]])
    #     plt.title(f'{i}')
    # load_bldg = input('로드할 빌딩 넘버: ')
    load_bldg = 0
    # plt.plot(data_raw[buildings[load_bldg]])
    # plt.title(f'{type} - {buildings[load_bldg]}')
    # plt.tight_layout()
    data = data_raw[buildings[int(load_bldg)]]
    data = pd.DataFrame(data, index=date_index, columns=[buildings[int(load_bldg)]])

    d_index = data.index
    f_month = d_index.month.to_frame(name='month')
    f_day = d_index.day.to_frame(name='day')
    f_hour = d_index.hour.to_frame(name='hour')
    f_min = d_index.minute.to_frame(name='minute')
    f_month.index = d_index
    f_day.index = d_index
    f_hour.index = d_index
    f_min.index = d_index

    dataset = pd.concat([data, f_month, f_day, f_hour, f_min], axis=1)
    return dataset


def split_dataset(dataset, timestep):
    # split into standard weeks
    data_raw = dataset.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data_raw)

    # idx = dataset.columns.get_loc(target)
    train, test = data[:28896], data[28896:]

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
# Apt(4), Complex(5), Dept(4), Hospital(4), Market(3), Office(4), School(15)
# type_list = ['Apt', 'Complex', 'Dept', 'Hospital', 'Market', 'Office', 'School']
type_list = ['Apt', 'Office']
type = type_list[0]
# type_list = ['Apt']
for type in type_list:
    # type = 'Apt'
    n_input, n_out, timestep = 24*4, 24*4, 24*4*7

    dataset = load_dataset(type=type)
    dataset = dataset.iloc[:364*96]

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    _ = target_scaler.fit_transform(pd.DataFrame(dataset.iloc[:, 0]))
    data_mean = dataset.mean()
    data_std = dataset.std()

    train, test = split_dataset(dataset, timestep=timestep)
    train_x, train_y = to_supervised(train, n_input=n_input, n_out=n_out)


    ##############################
    # parameters, fitting
    verbose, epochs, batch_size = 2, 50, 64
    model = lstm_model(train_x, n_out=n_out)
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt)
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, shuffle=False)
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

    model.save(f'lstm_{type}-{dataset.columns[0]}.h5')

    pred = np.array(predictions)
    # score, scores = evaluate_forecasts(test[:, :, 0], pred)

    y_true = test[:, :, 0].reshape((test.shape[0]*test.shape[1],))
    y_pred = pred.reshape((pred.shape[0]*pred.shape[2],))
    plot_index = dataset.index[28896:]

    result = pd.DataFrame(y_true, index=plot_index, columns=['true'])
    result['pred'] = y_pred
    result.to_csv(f'{type}-{dataset.columns[0]}.csv')

    print(f'{type} - {dataset.columns[0]} COMPLETED')
    print(f'RMSE: {mean_squared_error(y_true, y_pred)**(1/2)}')
    print(f' MSE: {mean_squared_error(y_true, y_pred)}')
    print(f' MAE: {mean_absolute_error(y_true, y_pred)}')
    print(f'MAPE: {mape(y_true, y_pred)}\n')


    plt.figure(figsize=(15, 5))
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.xticks(ticks=[i for i in range(0, len(y_true), 240*4)],
               labels=[str(plot_index[i])[:-3] for i in range(0, len(y_true), 240*4)],
               rotation=45)
    plt.xlim([0, len(y_true)])
    plt.legend(['observation', 'prediction'])
    plt.title(f'{type} - {dataset.columns[0]}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{type} - {dataset.columns[0]}.png')
    plt.close()
