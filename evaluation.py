from DecisionTree import dt_main
from SVR import svr_main
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import seaborn as sns

def bench_model():
    df_daily = pd.read_pickle('data_clean_daily.pkl')
    df = df_daily.set_index(['id', 'time']).sort_index()

    df_bench = pd.DataFrame(columns=['y_pred', 'y_true'])
    df_bench['y_pred'] = df[('mood', 'mean')]
    df_bench['y_true'] = df[('mood', 'mean')].groupby(level='id').shift(periods=-1) # next day mean mood
    df_bench.dropna(inplace=True)

    # sc_y = MinMaxScaler(feature_range=(-1,1))
    #
    # y_true = sc_y.fit_transform(np.array(df_bench['y_true']).reshape(-1,1))
    # y_pred = sc_y.fit_transform(np.array(df_bench['y_pred']).reshape(-1,1))
    # mae = mean_absolute_error(y_true, y_pred)

    mae = mean_absolute_error(df_bench['y_true'], df_bench['y_pred'])

    return mae

final_df = pd.DataFrame()

dt_results = dt_main()
svr_results = svr_main()
bench = bench_model()

final_df = final_df.append([dt_results, svr_results]).transpose()
plt.figure(figsize=(6,4))

final_df.boxplot()
plt.ylabel('Mean Absolute Error')
plt.ylim(0,1.25)
plt.savefig('Plots/Non_temporal.pdf')
plt.show()

#test optimal window size
for i in [2,3,4,7]:
    dt_results = dt_main(window=i).transpose()
    print(dt_results.iloc[:, 0])
    mean = dt_results.iloc[:, 0].mean()
    std = dt_results.iloc[:, 0].std()
    print(i, np.round(mean, 3), np.round(std, 4))



test_df = pd.DataFrame()
arima_mae = pd.read_pickle('MAE_arima.pkl').iloc[:, 0].rename('ARIMA').to_frame()
dt_results_7 = dt_main(window=7)
dt_results_7 = dt_results_7.transpose()
dt_results_7 = dt_results_7.iloc[:, 0].rename('Regression tree \n(window=7)').to_frame()
test_df = test_df.append([dt_results_7, arima_mae])

plt.figure(figsize=(4,4))
test_df.boxplot()
plt.ylabel('Mean Absolute Error')
plt.axhline(bench, c='r', linestyle='--', label='Baseline')
plt.legend()
plt.savefig('Plots/final_models.pdf')
plt.show()

# regressiontree = pd.DataFrame()
# for window in [3,5,7]:
#     dt_results = dt_main(window)
#     print(dt_results.iloc[0].mean())
#     regressiontree = regressiontree.append(dt_results)
#
# regressiontree.transpose().boxplot()
# plt.ylabel('Mean Absolute Error')
# plt.ylim(0,0.2)
#
# plt.legend()
# plt.show()