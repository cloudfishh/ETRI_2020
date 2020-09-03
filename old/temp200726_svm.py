from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from funcs import *


##################################################
# set variables
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
# test_house = '1dcb5feb'

# sigma = 3


##################################################
# load data
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)
data_col_raw = data[test_house]
calendar = load_calendar(2017, 2019)

data_col, inj_mask = inject_nan_acc3(data_col_raw, p_nan=1, p_acc=0.5)

# idx_list_result, idx_dict = check_accumulation_injected(data_col, inj_mask, calendar, sigma=sigma)

# candidate idx list
idx_list_candidate = np.array([])
candidate = inj_mask[(inj_mask == 3) | (inj_mask == 4)]
for i in range(len(candidate)):
    idx_list_candidate = np.append(idx_list_candidate, inj_mask.index.get_loc(candidate.index[i]))

##################################################
# calculate result - false positive & false negative
# candidate idx list
idx_list_candidate = np.array([])
candidate = inj_mask[(inj_mask == 3) | (inj_mask == 4)]
for i in range(len(candidate)):
    idx_list_candidate = np.append(idx_list_candidate, inj_mask.index.get_loc(candidate.index[i]))

# idx_list_acc = np.array([])
# acc = inj_mask[inj_mask == 3]
# for i in range(len(acc)):
#     idx_list_acc = np.append(idx_list_acc, inj_mask.index.get_loc(acc.index[i]))
#
# idx_list_normal = np.array([])
# nor = inj_mask[inj_mask == 4]
# for i in range(len(nor)):
#     idx_list_normal = np.append(idx_list_normal, inj_mask.index.get_loc(nor.index[i]))


##################################################
# make SVM dataset
def make_svm_dataset():
    dataset = pd.DataFrame(data_col.copy())
    dataset['mask'] = inj_mask

    # load features
    weather = pd.read_csv('D:/2020_ETRI/data/2017_기상데이터.csv', encoding='cp949')
    weather = pd.concat([weather, pd.read_csv('D:/2020_ETRI/data/2018_기상데이터.csv', encoding='cp949')], axis=0)
    weather = pd.concat([weather, pd.read_csv('D:/2020_ETRI/data/2019_기상데이터.csv', encoding='cp949')], axis=0)
    temp = weather['기온(°C)']
    temp.index = make_date_index(2017, 2019)
    humi = weather['습도(%)']
    humi.index = make_date_index(2017, 2019)

    dataset['temperature'] = temp.loc[dataset.index[0]:dataset.index[-1]]
    dataset['humidity'] = humi.loc[dataset.index[0]:dataset.index[-1]]

    consumption = np.empty((dataset.shape[0],))
    for i in range(0, weather.shape[0], 24):
        consumption[i:i+24] = sum(dataset.iloc[i:i+24, 0].fillna(0))
    dataset['consumption'] = consumption
    dataset['before'] = dataset['68181c16'].shift(1)
    dataset['after'] = dataset['68181c16'].fillna(method='backfill').shift(-1)

    X_train = dataset.iloc[idx_list_candidate, :].drop(columns=['mask'])
    target = dataset.iloc[idx_list_candidate, 1]
    return X_train, target

##################################################
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


def plot_coefficients(classifier, feature_names, top_features=6):
    coef = classifier.coef_.ravel()
    # top_positive_coefficients = np.argsort(coef)[-top_features:]
    # top_negative_coefficients = np.argsort(coef)[:top_features]
    # top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    top_coefficients = np.argsort(coef)[:]
    # create plot
    # plt.figure(figsize=(15, 5))
    plt.figure()
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    # plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    plt.xticks(range(0, top_features), feature_names[top_coefficients], rotation=45, ha='right')
    plt.show()


X_train, target = make_svm_dataset()
# cv = CountVectorizer()
# cv.fit(data)
# print(len(cv.vocabulary_))
# print(cv.get_feature_names())
# X_train = cv.transform(data)

svm = LinearSVC()
svm.fit(X_train.fillna(0), target.astype('int'))
plt.rcParams.update({'font.size': 13})
plot_coefficients(svm, X_train.columns, top_features=len(X_train.columns))
plt.tight_layout()


for i in range(len(X_train.columns)):
    col = X_train.columns[i]
    print(f'CORRCOEF {col}: {np.corrcoef(X_train[test_house].fillna(0), X_train[col].fillna(0))}')
