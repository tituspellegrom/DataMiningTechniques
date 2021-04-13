import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

df_daily = pd.read_pickle('data_clean_daily.pkl')


def bench_model(df_daily):
    df = df_daily.set_index(['id', 'time']).sort_index()

    df_bench = pd.DataFrame(columns=['y_pred', 'y_true'])
    df_bench['y_pred'] = df[('mood', 'mean')]
    df_bench['y_true'] = df[('mood', 'mean')].groupby(level='id').shift(periods=-1) # next day mean mood
    df_bench.dropna(inplace=True)

    mse = mean_squared_error(df_bench['y_pred'], df_bench['y_true'])
    mae = mean_absolute_error(df_bench['y_pred'], df_bench['y_true'])

    print(f"Naive Bench Model:\n MSE: {mse}\n MAE: {mae}")
    return mse


bench_model(df_daily)

#
# ## Tabular aggregation
# df_temp = df_daily.copy().set_index(['id', 'time']).sort_index()
# lookback_days = 7
# df_tab = df_temp[cols_score+cols_count+cols_time].rolling(window=lookback_days).mean()
# df_tab = pd.concat([df_tab, df_temp[['mon', 'thue', 'wed', 'thu', 'fri', 'sat', 'sun']]], axis=1)   # DayOfWeek excluded from rolling mean
# df_tab['target'] = df_temp[('mood', 'mean')].shift(periods=-1) # next day mean mood
#
# display(df_tab)
#
# #
# # lookback_days = 7
# # df_tab = df_temp[cols_score+cols_count+cols_time].rolling(window=lookback_days).mean()
# # df_tab = pd.concat([df_tab, df_temp[['mon', 'thue', 'wed', 'thu', 'fri', 'sat', 'sun']]], axis=1)   # DayOfWeek excluded from rolling mean
# # df_tab['target'] = df_temp[('mood', 'mean')].shift(periods=-1) # next day mean mood
# #
# # display(df_tab)