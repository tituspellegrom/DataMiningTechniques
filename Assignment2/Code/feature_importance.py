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


def randomForest(data, parameters=None):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    regr = RandomForestRegressor(max_depth=3, random_state=1, verbose=1, n_jobs=6)
    fit = regr.fit(X, y.ravel())

    # plot graph of feature importances for better visualization
    fig, ax = plt.subplots()
    feat_importances = pd.Series(fit.feature_importances_, index=data.columns[:-1])
    feat_importances.nlargest(20).plot(kind='barh', xlabel='Feature')
    ax.set_xlabel('Gini performance')
    plt.show()

    model = ExtraTreesRegressor(random_state=1, max_depth=3,  verbose=1, n_jobs=6)
    model.fit(X, y.ravel())
    fig, ax = plt.subplots()

    feat_importances = pd.Series(model.feature_importances_, index=data.columns[:-1])
    feat_importances.nlargest(20).plot(kind='barh', xlabel='Feature')
    ax.set_xlabel('Gini importance')
    plt.tight_layout()

    plt.savefig('ExtraTree.pdf')

    plt.show()
    
data = pd.read_pickle('training_set_VU_DM.pkl')
randomForest(data)