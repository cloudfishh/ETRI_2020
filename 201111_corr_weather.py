"""
Accumulation detection with similar days
 and fwd-bwd joint imputation with AR
- length of NaN = 5 test

2020. 11. 11. Wed.
Soyeong Park
"""
##############################
from funcs import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


##############################
# matplotlib font setting
font_dir = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_dir).get_name()
plt.rcParams.update({'font.size': 13,
                     'font.family': font_name})


##############################
# 0. parameter setting
# test_house_list = ['68181c16', '1dcb5feb', '2ac64232', '3b218da6', '6a638b96']
test_house = '68181c16'
f_fwd, f_bwd = 24, 24
nan_len = 5


##############################
# 1. load dataset
data_raw = load_labeled()
data, nan_data = clear_head(data_raw)

print(f'********** TEST HOUSE {test_house} start - {np.where(data.columns == test_house)[0][0]}th')
data_col = data[test_house]
calendar = load_calendar(2017, 2019)
df = pd.DataFrame([], index=data_col.index)
df['values'] = data_col.copy()
df['nan'] = chk_nan_bfaf(data_col)
df['holiday'] = calendar[data_col.index[0]:data_col.index[-1]]
df['org_idx'] = np.arange(0, len(data_col))


##############################
# 날씨 로드해
weather_raw = load_weather('incheon', 2017, 2019)
weather = weather_raw[df.index[0][:16]:df.index[-1][:16]]
weather.index = df.index

df_corr = pd.concat([df['values'], weather[[col for col in weather.columns.values if col.endswith('QC플래그')==False]]], axis=1)

corr = df_corr.corr(method='pearson')
corr_na = df_corr.fillna(0).corr(method='pearson')

plt.figure(figsize=(15,7))
plt.plot(corr['values'][:-8].sort_values(ascending=False), 'r*')
plt.plot(corr_na['values'][:-8].sort_values(ascending=False), 'gX')
plt.axhline(y=0, color='k', linewidth=0.3, linestyle='--')
plt.xticks(rotation=90)
plt.ylim([-0.4, 0.6])
plt.legend(['raw data', 'NaN filled'], loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

corr['values'][:-8].sort_values(ascending=False)