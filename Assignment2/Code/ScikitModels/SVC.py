import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_validate
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import scipy.sparse as sp
import dask.array as da
from dask_ml.wrappers import Incremental
from dask.diagnostics import ProgressBar
import gc
import time

arr = sp.load_npz('../df_features_data_merged.npz').tocsr()
# arr = da.from_array(arr)
X, y = arr[:, :-1], arr[:, -1]
del arr
gc.collect()

time.sleep(5)   # give gc some time

embedding_dims = np.genfromtxt(f'../df_features_embedding_dims.csv', delimiter=',')
non_embedded = np.sum(embedding_dims[:-1] == 1)

# TODO: keep searches together
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, random_state=1)

sc_X = MaxAbsScaler()

X_train[:, :non_embedded] = sc_X.fit_transform(X_train[:, :non_embedded])
X_test[:, :non_embedded] = sc_X.transform(X_test[:, :non_embedded])

# svc = SVC()
#
# mdl = svc.fit(X, y)











# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]
# parallel_classifiers = [Incremental(cls) for cls in classifiers]
#
# for parallel_cls in tqdm(parallel_classifiers):
#     try:
#         with ProgressBar():
#             parallel_cls.fit(X, y)
#     except ValueError as e:
#         print(e)
#         continue




