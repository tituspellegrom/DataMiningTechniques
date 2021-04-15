from DecisionTree import dt_main
from SVR import svr_main
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def bench_model():
    df_daily = pd.read_pickle('data_clean_daily.pkl')
    df = df_daily.set_index(['id', 'time']).sort_index()

    df_bench = pd.DataFrame(columns=['y_pred', 'y_true'])
    df_bench['y_pred'] = df[('mood', 'mean')]
    df_bench['y_true'] = df[('mood', 'mean')].groupby(level='id').shift(periods=-1) # next day mean mood
    df_bench.dropna(inplace=True)

    mae = mean_absolute_error(df_bench['y_pred'], df_bench['y_true'])

    return mae

final_df = pd.DataFrame()

dt_results = dt_main()
svr_results = svr_main()
bench = bench_model()

final_df = final_df.append([dt_results, svr_results]).transpose()
print(final_df)

final_df.boxplot()
plt.axhline(bench, c='r', linestyle='--', label='Baseline')
plt.xticks(rotation='diagonal')

plt.legend()
plt.show()