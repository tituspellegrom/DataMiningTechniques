import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from DecisionTree import userEncode, tabular_aggregation, scaleData, PCAtransform, loadFeatures



def svr(data, method, features):
    data.columns = [' '.join(col).strip() for col in data.columns.values]
    if method =='feature selection':
        X=data[features].values
    else:
        X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    X, binary_cols = userEncode(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, random_state=1)

    data_splits = {}
    # data_splits['X_train'] = X_train
    # data_splits['y_train'] = y_train
    data_splits['X_train'] = X_train_val
    data_splits['y_train'] = y_train_val
    data_splits['X_val'] = X_val
    data_splits['y_val'] = y_val
    data_splits['X_test'] = X_test
    data_splits['y_test'] = y_test


    if method == 'PCA':
        data_splits_scaled = scaleData(data_splits, binary_cols)
        X_train, X_val, y_train, y_val, X_test, y_test = PCAtransform(data_splits_scaled)
    else:
        data_splits_scaled = scaleData(data_splits, binary_cols)
        X_train = data_splits_scaled['X_train']
        y_train = data_splits_scaled['y_train']
        X_val = data_splits_scaled['X_val']
        y_val = data_splits_scaled['y_val']
        X_test = data_splits_scaled['X_test']
        y_test = data_splits_scaled['y_test']

    metrics_df = pd.DataFrame(columns=['Kernel', 'MAE', 'MSE'])
    #kernels =

    # TODO: Optimize kernels
    score = cross_validate(SVR(kernel='rbf').fit(X_train, y_train.ravel()),
                           X_val,
                           y_val.ravel(),
                           cv=10,
                           scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'])

    # print(f'Scores for each fold: {score}')
    MAE = -score['test_neg_mean_absolute_error']
    MSE = -score['test_neg_mean_squared_error']

    MAE_series = pd.Series(-score['test_neg_mean_absolute_error'], name='SVR \n'+method)

    return MAE_series

    # regr = SVR(random_state=1, kernel='rbf')
    # regr.fit(X_train, y_train)
    # y_pred = regr.predict(X_val)
    # MSE_val = mean_squared_error(y_val, y_pred)
    # MAE_val = mean_absolute_error(y_val, y_pred)

def svr_main():
    data = pd.read_pickle('data_clean_daily.pkl')
    df_tab = tabular_aggregation(data, 7, 7)
    features = loadFeatures()
    methods = ['feature selection', 'PCA']
    results = pd.DataFrame()
    for method in methods:
        MAE_series = svr(df_tab, method, features)
        results = results.append(MAE_series)
    return results

if __name__ == '__main__':
    svr_main()
# TODO: optimize days

#
#
# X = df_tab.iloc[:, :-1].values
# y = df_tab.iloc[:, -1].values.reshape(-1, 1)
#
# print(X)
# print(y)
#
# # Encode user label
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# X_usr = enc.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
#
# binary_cols = X_usr.shape[1]+7   # usr_labels and 7 days
# X = np.hstack([X_usr, X[:, 1:]])
#
#
# # Data split
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=0)
#
#
# # Feature Scaling
# from sklearn.preprocessing import MinMaxScaler
# sc_X = MinMaxScaler()
# sc_y = MinMaxScaler()
#
# X_train[:, binary_cols:] = sc_X.fit_transform(X_train[:, binary_cols:])
# X_val[:, binary_cols:] = sc_X.transform(X_val[:, binary_cols:])
# X_test[:, binary_cols:] = sc_X.transform(X_test[:, binary_cols:])
#
# y_train = sc_y.fit_transform(y_train)
# y_val = sc_y.transform(y_val)
# y_test = sc_y.transform(y_test)
#
# # PCA => benefits from feature scaling
# from sklearn.decomposition import PCA
# min_dimension = min(*X_train.shape, *X_val.shape, *X_test.shape)
# pca = PCA(n_components=min_dimension)
#
# X_train_pca = pca.fit_transform(X_train)
# X_val_pca = pca.fit_transform(X_val)
# X_test_pca = pca.transform(X_test)
#
#
# # Training the Support Vector Regression model on the Training set
# from sklearn.svm import SVR
# svr_regressor = SVR(kernel='rbf')
# # regressor = SVR(kernel = 'poly')
#
# svr_regressor.fit(X_train_pca, y_train.ravel())
#
# # Prediction evaluation
# y_pred_sc = svr_regressor.predict(X_test_pca)
# y_pred = sc_y.inverse_transform(y_pred_sc.reshape(-1, 1))
# y_true = sc_y.inverse_transform(y_test)
#
# mse = mean_squared_error(y_pred, y_true)
# mae = mean_absolute_error(y_pred, y_true)
# print(f"SVR MSE: {mse}")
# print(f"SVR MAE: {mae}")
#
# bench_model()
#
#
# # Cross-validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator=svr_regressor, X=X_train_pca, y=y_train.ravel(), cv=20, scoring='neg_mean_squared_error')
#
# print(-accuracies.mean())   # TODO: Somehow scale back to compare with benchmark
# # y_pred = np.array([y_train.mean() for _ in range(len(y_test))]).reshape(-1, 1)
#
