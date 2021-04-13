import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from tqdm import tqdm
import graphviz

import matplotlib.pyplot as plt
import wandb

# wandb.init(project='DMT1', entity='jaimierutgers')

def userEncode(X):
    enc = OneHotEncoder()
    X_usr = enc.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
    binary_cols = X_usr.shape[1] + 7  # usr_labels and 7 days
    X = np.hstack([X_usr, X[:, 1:]])

    return X, binary_cols

def scaleData(data_splits, binary_cols):
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']

    sc_X = MinMaxScaler(feature_range=(-1,1))
    sc_y = MinMaxScaler(feature_range=(-1,1))

    X_train[:, binary_cols:] = sc_X.fit_transform(X_train[:, binary_cols:])
    X_val[:, binary_cols:] = sc_X.transform(X_val[:, binary_cols:])
    X_test[:, binary_cols:] = sc_X.transform(X_test[:, binary_cols:])

    y_train = sc_y.fit_transform(y_train)
    y_val = sc_y.transform(y_val)
    y_test = sc_y.transform(y_test)

    data_splits['X_train'] = X_train
    data_splits['y_train'] = y_train
    data_splits['X_val'] =  X_val
    data_splits['y_val'] = y_val
    data_splits['X_test'] = X_test
    data_splits['y_test'] = y_test

    return data_splits


def PCAtransform(data_splits):
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']

    min_dimension = min(*X_train.shape, *X_val.shape, *X_test.shape)
    pca = PCA(n_components=min_dimension)
    X_train = pca.fit_transform(X_train)
    X_val = pca.fit_transform(X_val)
    X_test = pca.transform(X_test)

    return X_train, X_val, y_train, y_val, X_test, y_test

def tabular_aggregation(df_daily, lookback_days, min_period):
    df_temp = df_daily.copy().set_index(['id', 'time']).sort_index()
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

def regressionTree(data, parameters=None):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    X, binary_cols = userEncode(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=1)

    data_splits = {}
    data_splits['X_train'] = X_train
    data_splits['y_train'] = y_train
    data_splits['X_val'] =  X_val
    data_splits['y_val'] = y_val
    data_splits['X_test'] = X_test
    data_splits['y_test'] = y_test

    data_splits_scaled = scaleData(data_splits, binary_cols)
    X_train, X_val, y_train, y_val, X_test, y_test = PCAtransform(data_splits_scaled)


    metrics_df = pd.DataFrame(columns=['Depth', 'Samples split', 'Samples leaf', 'MAE', 'MSE'])
    for i in tqdm(range(1,50)):
        for j in range(2,40):
            for k in range(1,20):

                clf = DecisionTreeRegressor(random_state=1, max_depth=i, min_samples_split=j, min_samples_leaf=k)
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)

                # wandb.sklearn.plot_regressor(clf, X_train, X_val, y_train.flatten(), y_val, model_name='Decision tree')
                # wandb.sklearn.plot_feature_importances(clf, data.columns)

                MSE = metrics.mean_squared_error(y_val, y_pred)
                MAE = metrics.mean_absolute_error(y_val, y_pred)
                metrics_df = metrics_df.append({'Depth':i, 'Samples split':j, 'Samples leaf':k, 'MAE':MAE, 'MSE':MSE}, ignore_index=True)


    print(metrics_df)
    print(metrics_df.iloc[metrics_df['MAE'].argmin()])
    print(metrics_df.iloc[metrics_df['MSE'].argmin()])



    dot_data = tree.export_graphviz(clf, out_file='tree.dot')



data = pd.read_pickle('data_clean_daily.pkl')
df_tab = tabular_aggregation(data, 7, 7)
regressionTree(df_tab)

