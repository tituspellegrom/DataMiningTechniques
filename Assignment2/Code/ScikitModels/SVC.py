import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.svm import SVC
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
from dask_ml.model_selection import GridSearchCV
from dask.distributed import Client
import gc
import time
import dill
from sklearn.utils import all_estimators
from collections import namedtuple


def save_model(model, model_name=None):
    if model_name is None:
        model_name = type(model).__name__

    if hasattr(model, 'cv_results_'):
        cv_result = pd.DataFrame(model.cv_results_)
        cv_result.to_excel(f"cv_output/{model_name}.xlsx", index=False)

    with open(f"models_saved/{model_name}.pkl", 'wb') as file:
        dill.dump(model, file)


Classifier = namedtuple('Classifier', 'name scikit_class sparse_support dask_support proba_support')


def obtain_scikit_classifiers(verbose=True):
    estimators = all_estimators(type_filter='classifier')

    classifiers = []
    for name, ClassifierClass in estimators:
        docstring = getattr(ClassifierClass, 'fit').__doc__
        sparse_support = docstring is not None and "X : {array-like, sparse matrix}" in docstring
        dask_support = hasattr(ClassifierClass, 'partial_fit')
        proba_support = hasattr(ClassifierClass, 'predict_proba')

        classifier = Classifier(name, ClassifierClass, sparse_support, dask_support, proba_support)
        classifiers.append(classifier)

        if verbose:
            print(f"{name}:\n"
                  f"\tsupport Sparsity: {sparse_support}\n"
                  f"\tsupports Dask training: {dask_support}\n"
                  f"\tsupports predict_proba(): {proba_support}")

    return classifiers


classifiers = obtain_scikit_classifiers()

arr = sp.load_npz('../df_temporary_data_merged.npz').tocsr()
# arr = da.from_array(arr)
X = arr[:, :-1]
y = arr[:, -1].todense().reshape(-1, 1)
del arr
gc.collect()

time.sleep(5)   # give gc some time

embedding_dims = np.genfromtxt(f'../df_features_embedding_dims.csv', delimiter=',')
non_embedded = np.sum(embedding_dims[:-1] == 1)

# TODO: keep searches together
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.999, random_state=1)

sc_X = MaxAbsScaler()

X_train[:, :non_embedded] = sc_X.fit_transform(X_train[:, :non_embedded])
X_test[:, :non_embedded] = sc_X.transform(X_test[:, :non_embedded])


for classifier in tqdm(classifiers):
    print(f"Fitting {classifier.name}")

    if not classifier.sparse_support:
        # TODO: use data without one-hot encoding & dense format
        continue

    # TODO: map all params with for init => and if probability in kwargs set to True

    cl = classifier.scikit_class()

    if classifier.dask_support:
        parallel_X_train = da.from_array(X_train)
        parallel_y_train = y_train
        cl = Incremental(cl)
        cl.fit(parallel_X_train, parallel_y_train)
    else:
        cl.fit(X_train, y_train)

    # TODO: wrap CalibratedClassifierCV if needed and activate predict_proba()
    # y_prob = cl.predict_proba(X_test[:1000, :])
    score = cl.score(X_test[:1_000, :], y_test[:1000])

    save_model(cl, model_name=classifier.name)


# svc = SVC(probability=True)
#
# parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [.5, 1., 2.]}
#
# tuned_svc = GridSearchCV(svc, parameters)
#
# with ProgressBar():
#     tuned_svc.fit(X_train, y_train)
#
#     y_pred = tuned_svc.predict(X_test[:1000, :])
#
#     y_prob = tuned_svc.predict_proba(X_test[:1000, :])
#     score = tuned_svc.score(X_test[:1_000, :], y_test[:1000])
#
#     save_model(tuned_svc, model_name=type(svc).__name__)
#
# print(score)
#

# parallel_svc = Incremental(svc)
#
# with ProgressBar():
#     parallel_svc.fit(X_train, y_train, classes=[0, 1, 5])
#
#     y_pred = parallel_svc.predict(X_test[:1000, :])
#
#     score = parallel_svc.score(X_test[:1_000, :], y_test[:1000])


# if parallel supported
# parallel_svc = Incremental(svc)
#
# with ProgressBar():
#     parallel_svc.fit(X_train, y_train, classes=[0, 1, 5])
#
#     y_pred = parallel_svc.predict(X_test[:1000, :])
#
#     score = parallel_svc.score(X_test[:1_000, :], y_test[:1000])













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




