import pandas as pd
from matplotlib import pyplot as plt

result = pd.read_csv('csv/AR_imputation_result.csv', header=0, index_col=0)
x_labels = [f'({f_fwd}, {f_bwd})' for f_fwd, f_bwd in zip(result['f_fwd'], result['f_bwd'])]


##############################
fig, ax1 = plt.subplots(figsize=(15, 5))
ax2 = ax1.twinx()
data_y10 = ax1.plot(result['avr_mse'], linewidth=0.5, color='blue', marker='x', markersize=4)
data_y11 = ax1.plot(result['avr_rmse'], linewidth=0.5, color='green', marker='o', markersize=4)
data_y2 = ax2.plot(result['avr_smape'], linewidth=0.5, color='orange', marker='*', markersize=4)

ax1.set_xlabel('filter length (forward, backward))')
ax1.set_xticks(ticks=[i+24 for i in range(0, 24*24, 24)])
ax1.set_xticklabels(labels=[x_labels[l+23] for l in range(0, 24*24, 24)], rotation=45)
ax1.set_ylabel('MSE, RMSE')
ax1.set_ylim([-0.1, 1])
ax2.set_ylabel('sMAPE')

plt.grid('both', alpha=0.3)
plt.legend(data_y10+data_y11+data_y2, ['MSE', 'RMSE', 'sMAPE'])
plt.tight_layout()
plt.show()


##############################
for j in range(0, 24, 6):
    plt.figure(figsize=(7.5, 5))
    plt.plot(result['avr_smape'].values.reshape((24, -1))[:, j:j+6], 'x-')
    # plt.ylim([-0.1, 1])
    plt.legend([i for i in range(j, j+6)])


sort_mse = result.copy().sort_values(by='avr_mse').drop(columns=['avr_rmse', 'avr_smape'])
sort_rmse = result.copy().sort_values(by='avr_rmse').drop(columns=['avr_mse', 'avr_smape'])
sort_smape = result.copy().sort_values(by='avr_smape').drop(columns=['avr_mse', 'avr_rmse'])

sort_mse.to_csv('flen_sort_mse.csv')
sort_rmse.to_csv('flen_sort_rmse.csv')
sort_smape.to_csv('flen_sort_smape.csv')

# (f_fwd, f_bwd) = (24, 24) is the best
