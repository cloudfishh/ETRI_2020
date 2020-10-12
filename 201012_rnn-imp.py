'''
Accumulated detection with nearest neighbor,
imputation with RNN
 - forward-backward joint imputation (weighted sum)

2020. 10. 12. Mon.
Soyeong Park
'''
##############################
from funcs import *
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


##############################
def split_dataset(dataset, timestep):
    # split into standard weeks
    data_raw = dataset.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data_raw)

    # idx = dataset.columns.get_loc(target)
    train, test = data[:10752], data[10752:]

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


def pv_model(train_x, n_out):
    model = Sequential()
    model.add(LSTM(48, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(24, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(24))
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


##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 3


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()
df['nan'] = chk_nan_bfaf(data_col)
df['holiday'] = calendar.loc[pd.Timestamp(df.index[0]):pd.Timestamp(df.index[-1])]


##############################
# 2. injection
df['injected'], df['mask_inj'] = inject_nan_acc3(data_col, p_nan=1, p_acc=0.25)


##############################
# 3. get the sample with nearest neighbor method
idx_list = np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))[0]
nan_mask = df['nan'].copy()

sample_list, mean_list, std_list = list(), list(), list()
for i in range(len(idx_list)):
    idx_target = idx_list[i]
    sample, m, s = nearest_neighbor(data_col, df['nan'].copy(), idx_target, calendar)
    sample_list.append(sample)
    mean_list.append(m)
    std_list.append(s)
smlr_sample = pd.DataFrame(sample_list)
# smlr_sample.to_csv('result_nearest.csv')


# 3-2. z-score
# smlr_sample = pd.read_csv('result_nearest.csv', index_col=0)
cand = df[(df['mask_inj'] == 3) | (df['mask_inj'] == 4)].copy()
z_score = (cand['injected'].values-smlr_sample.mean(axis=1))/smlr_sample.std(axis=1)
cand['z_score'] = z_score.values

# df_z = df.copy()
df['z_score'] = np.nan
df['z_score'][(df['mask_inj'] == 3) | (df['mask_inj'] == 4)] = z_score.values

# 3-3. determine threshold for z-score
# x-axis) z-score threshold [0, 10], y-axis) # of detected acc.
# detection = list()
# for z in np.arange(0, 40, 0.1):
#     detection.append([z,
#                       sum((cand['mask_inj'] == 4) & (cand['z_score'] > z)),
#                       sum((cand['mask_inj'] == 3) & (cand['z_score'] > z)),     # false positive (true nor, detect acc)
#                       sum((cand['mask_inj'] == 4) & (cand['z_score'] < z))])    # false negative (true acc, detect nor)
# detection = pd.DataFrame(detection, columns=['z-score', 'detected_acc', 'false_positive', 'false_negative'])

# threshold = 7.5
threshold = 3.4
idx_detected_nor = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] < threshold))[0]
idx_detected_acc = np.where(((df['mask_inj'] == 3) | (df['mask_inj'] == 4)) & (df['z_score'] > threshold))[0]
detected = np.zeros(len(data_col))
detected[np.where((df['mask_inj'] == 3) | (df['mask_inj'] == 4))] = 3
detected[idx_detected_acc.astype('int')] = 4
df['mask_detected'] = detected



##############################
n_input, n_out, timestep = 24*7, 24*7, 24*7

dataset = dataset.iloc[:12768, :]       # 20180609 23:00, 532days, 76weeks, 19months

target_scaler = MinMaxScaler(feature_range=(0, 1))
_ = target_scaler.fit_transform(pd.DataFrame(dataset["PV_obs"]))
data_mean = dataset.mean()
data_std = dataset.std()

train, test = split_dataset(dataset, timestep=timestep)
train_x, train_y = to_supervised(train, n_input=n_input, n_out=n_out)


##############################
# parameters, fitting
verbose, epochs, batch_size = 2, 1, 16
model = pv_model(train_x, n_out=n_out)
model.compile(loss='mse', optimizer='adam')
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

# model.save('lstm_pv_model.h5')

pred = np.array(predictions)
# score, scores = evaluate_forecasts(test[:, :, 0], pred)

y_true = test[:, :, 0].reshape((test.shape[0]*test.shape[1],))
y_pred = pred.reshape((pred.shape[0]*pred.shape[2],))
plot_index = dataset.index[10752:]

plt.figure(figsize=(15, 5))
plt.plot(y_true)
plt.plot(y_pred)
plt.xticks(ticks=[i for i in range(0, len(y_true), 240)],
           labels=[plot_index[i] for i in range(0, len(y_true), 240)],
           rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()