import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

DF_DAILY = pd.read_pickle('data_clean_daily.pkl')


def bench_model():
    df_daily = pd.read_pickle('data_clean_daily.pkl')
    df = df_daily.set_index(['id', 'time']).sort_index()


    df_bench = pd.DataFrame(columns=['y_pred', 'y_true'])
    df_bench['y_pred'] = df[('mood', 'mean')]
    df_bench['y_true'] = df[('mood', 'mean')].groupby(level='id').shift(periods=-1) # next day mean mood
    df_bench.dropna(inplace=True)

    mse = mean_squared_error(df_bench['y_pred'], df_bench['y_true'])
    mae = mean_absolute_error(df_bench['y_pred'], df_bench['y_true'])

    print(f"Naive Bench Model:\n MSE: {mse}\n MAE: {mae}")
    return mae


def tabular_aggregation(lookback_days, min_period):
    df_temp = DF_DAILY.copy().set_index(['id', 'time']).sort_index()
    df_temp.reset_index('time', drop=True, inplace=True)    # Discard time variable after sort

    # columns lists
    cols = df_temp.columns.get_level_values(0).unique().to_list()
    day_cols = ['mon', 'thue', 'wed', 'thu', 'fri', 'sat', 'sun']
    sleep_cols = ['sleep']
    aggregate_cols = [c for c in cols if c not in day_cols+sleep_cols]

    # df splitted
    df_days = df_temp[day_cols].reset_index()
    df_tab = df_temp[aggregate_cols].groupby('id').rolling(window=lookback_days, min_period=min_period).mean().reset_index(drop=True)

    # Workaround: Pandas fucks up when applying more than 1 aggregation on a rolling window
    sleep_fn = {'max': np.max, 'min': np.min, 'median': np.median, 'mean': np.mean, 'sum': np.sum, 'std': np.std}
    dfs_sleep = [df_temp[sleep_cols].groupby('id').rolling(window=lookback_days, min_period=min_period).agg(fn) for fn in sleep_fn.values()]
    df_sleep = pd.concat(dfs_sleep, axis=1).reset_index(drop=True)
    df_sleep.columns = [('sleep', k) for k in sleep_fn]

    # next day mean mood
    df_target = df_temp[('mood', 'mean')].groupby(level='id').shift(periods=-1).reset_index(drop=True)

    df_tab2 = pd.concat([df_days, df_tab, df_sleep, df_target], axis=1)

    return df_tab2.dropna()


df_tab = tabular_aggregation(lookback_days=7, min_period=7)

X = df_tab.iloc[:, :-1].values
y = df_tab.iloc[:, -1].values.reshape(-1, 1)

print(X)
print(y)

# Encode user label
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_usr = enc.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

binary_cols = X_usr.shape[1]+7   # usr_labels and 7 days
X = np.hstack([X_usr, X[:, 1:]])


# Data split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=0)


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
sc_y = MinMaxScaler()

X_train[:, binary_cols:] = sc_X.fit_transform(X_train[:, binary_cols:])
X_val[:, binary_cols:] = sc_X.transform(X_val[:, binary_cols:])
X_test[:, binary_cols:] = sc_X.transform(X_test[:, binary_cols:])

y_train = sc_y.fit_transform(y_train)
y_val = sc_y.transform(y_val)
y_test = sc_y.transform(y_test)

# PCA => benefits from feature scaling
from sklearn.decomposition import PCA
min_dimension = min(*X_train.shape, *X_val.shape, *X_test.shape)
pca = PCA(n_components=min_dimension)

X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.fit_transform(X_val)
X_test_pca = pca.transform(X_test)


# Training the Support Vector Regression model on the Training set
from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')
# regressor = SVR(kernel = 'poly')

svr_regressor.fit(X_train_pca, y_train.ravel())

# Prediction evaluation
y_pred_sc = svr_regressor.predict(X_test_pca)
y_pred = sc_y.inverse_transform(y_pred_sc.reshape(-1, 1))
y_true = sc_y.inverse_transform(y_test)

mse = mean_squared_error(y_pred, y_true)
mae = mean_absolute_error(y_pred, y_true)
print(f"SVR MSE: {mse}")
print(f"SVR MAE: {mae}")

bench_model()


# Cross-validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=svr_regressor, X=X_train_pca, y=y_train.ravel(), cv=20, scoring='neg_mean_squared_error')

print(-accuracies.mean())   # TODO: Somehow scale back to compare with benchmark
# y_pred = np.array([y_train.mean() for _ in range(len(y_test))]).reshape(-1, 1)








