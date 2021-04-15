import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
#
import matplotlib.pyplot as plt
import wandb

# wandb.init(project='DMT1', entity='jaimierutgers')

def getColumnNames(data):
    users = ['AS14.01', 'AS14.02','AS14.03','AS14.05','AS14.06','AS14.07','AS14.08',
 'AS14.09','AS14.12','AS14.13','AS14.14','AS14.15','AS14.16','AS14.17',
 'AS14.19','AS14.20','AS14.23','AS14.24','AS14.25','AS14.26','AS14.27',
 'AS14.28','AS14.29','AS14.30','AS14.31','AS14.32','AS14.33']

    columns = [' '.join(col).strip() for col in data.columns.values]
    columns_no_id = columns[1:-1].copy()
    for user in users:
        columns_no_id.append(user)

    return columns_no_id


def userEncode(X):
    enc = OneHotEncoder()
    X_usr = enc.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
    binary_cols = X_usr.shape[1] + 7  # usr_labels and 7 days
    X = np.hstack([X_usr, X[:, 1:]])

    return X, binary_cols

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

def randomForest(data, parameters=None):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    X, binary_cols = userEncode(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, random_state=1)

    regr = RandomForestRegressor(max_depth=3, random_state=1)
    fit = regr.fit(X, y.ravel())

    # plot graph of feature importances for better visualization
    fig, ax = plt.subplots()
    feat_importances = pd.Series(fit.feature_importances_, index=getColumnNames(data))
    feat_importances.nlargest(20).plot(kind='barh', xlabel='Feature')
    ax.set_xlabel('Gini performance')
    plt.show()

    model = ExtraTreesRegressor(random_state=1, max_depth=7)
    model.fit(X, y.ravel())
    fig, ax = plt.subplots()

    feat_importances = pd.Series(model.feature_importances_, index=getColumnNames(data))
    feat_importances.nlargest(20).plot(kind='barh', xlabel='Feature')
    ax.set_xlabel('Gini importance')
    plt.tight_layout()

    plt.savefig('ExtraTree.pdf')

    plt.show()


    # dot_data = tree.export_graphviz(clf, out_file='tree.dot')

def k_best(data):
    # data.columns = [' '.join(col).strip() for col in data.columns.values]
    # data = data.drop('id', axis=1)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    X, binary_cols = userEncode(X)

    bestfeatures = SelectKBest(score_func=f_regression, k=20)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(getColumnNames(data))
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(20, 'Score'))  # print 20 best features
    print(featureScores.nlargest(20, 'Score').to_latex(float_format="%.2f"))


def correlation_matrix(df_tab):
    #import plotly.graph_objects as go

    # rename the last column as target
    df_tab.columns = list(map(lambda x: ' '.join(x), df_tab.columns))
    df_tab.columns = df_tab.columns.tolist()[:-1] + ['target']

    df_corr = df_tab.corr()
    df_target_cor = df_corr['target'].sort_values(key=abs, ascending=False)

    # df_target_cor = df_target_cor[df_target_cor.abs() >= 0.15]
    df_target_cor = df_target_cor.iloc[1:21]

    for col in df_target_cor.index:
        print(col)

    fig, ax = plt.subplots()
    df_target_cor.plot(kind='barh', xlabel='Feature')
    ax.set_xlabel('Pearson Coefficient')
    plt.tight_layout()
    plt.xlim([-1, 1])
    plt.savefig('pearson.png')
    #
    # fig = go.Figure(data=go.Bar(x=list(map(lambda x: ', '.join(x), df_target_cor.index)),
    #                             y=df_target_cor.values))
    #
    # fig.update_layout(
    #     title="Pearson Correlation",
    #     xaxis_title="Explanatory Variable",
    #     yaxis_title="Pearson Coefficient",
    #     yaxis_range=[-1, 1]
    # )
    # fig.show()
    # fig.write_html("plots/corr_target.html")

    return df_target_cor
    # fig = go.Figure(data=go.Heatmap(z=df_corr_sign.values,
    #                                 x=list(map(lambda c: '-'.join(c), df_corr_sign.axes[1])),
    #                                 y=list(map(lambda c: '-'.join(c), df_corr_sign.axes[0])),
    #                                 zmid=0
    #                                 ))
    # fig.show()
    # fig.write_html("plots/correlation2.html")
    # pass
    
    
data = pd.read_pickle('data_clean_daily.pkl')

df_tab = tabular_aggregation(data, 7,7)
randomForest(df_tab)
k_best(df_tab)
correlation_matrix(df_tab)





