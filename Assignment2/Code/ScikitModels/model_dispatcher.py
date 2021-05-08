import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_validate, GroupShuffleSplit

import scipy.sparse as sp
import dask.array as da
from dask_ml.wrappers import Incremental
from dask_ml.model_selection import GridSearchCV
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import gc
import time
import dill
from collections import ChainMap
import joblib

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


def obtain_scikit_classifiers(verbose=True, skip_classifiers={'ClassifierChain', 'MultiOutputClassifier',
                                                              'StackingClassifier', 'VotingClassifier',
                                                              'CategoricalNB', 'CalibratedClassifierCV',
                                                              'MultinomialNB', 'OneVsOneClassifier',
                                                              'OneVsRestClassifier', 'OutputCodeClassifier',
                                                              },
                              scikit_inconsistent_sparse={'CategoricalNB'}):

    estimators = all_estimators(type_filter='classifier')

    classifiers = []
    for name, ClassifierClass in estimators:
        if name in skip_classifiers:
            continue

        docstring = getattr(ClassifierClass, 'fit').__doc__

        sparse_support = docstring is not None and "X : {array-like, sparse matrix}" in docstring
        # Scikit is not consistent with docstrings & true support
        if name in scikit_inconsistent_sparse:
            sparse_support = False

        dask_support = hasattr(ClassifierClass, 'partial_fit')
        proba_support = hasattr(ClassifierClass, 'predict_proba')

        classifier = Classifier(name, ClassifierClass, sparse_support, dask_support, proba_support)
        classifiers.append(classifier)

        if verbose:
            print(f"{name}:\n"
                  f"\tsupports sparsity: {sparse_support}\n"
                  f"\tsupports dask training: {dask_support}\n"
                  f"\tsupports predict_proba(): {proba_support}")

    return classifiers


def prepare_data(dataset_name):
    arr = sp.load_npz(f"../{dataset_name}_data_merged.npz").tocsr()
    groups = np.load(f"../{dataset_name}_groups.npy")
    embedding_dims = np.genfromtxt(f'../{dataset_name}_embedding_dims.csv', delimiter=',')
    non_embedded = np.sum(embedding_dims[:-1] == 1)

    arr_train, arr_test, idx_train, idx_test = split_groups(arr, groups)
    del arr
    gc.collect()

    X_train, X_test = arr_train[:, :-1], arr_test[:, :-1]
    y_train = np.ravel(arr_train[:, -1].todense(), 'C')
    y_test = np.ravel(arr_test[:, -1].todense(), 'C')

    sc_X = MaxAbsScaler()

    X_train[:, :non_embedded] = sc_X.fit_transform(X_train[:, :non_embedded])
    X_test[:, :non_embedded] = sc_X.transform(X_test[:, :non_embedded])

    return X_train, X_test, y_train, y_test


def split_groups(arr, groups):
    gss = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7).split(arr, groups=groups)

    idx_train, idx_test = next(gss)

    data_train, data_test = arr[idx_train, :], arr[idx_test, :]
    return data_train, data_test, idx_train, idx_test


def prepare_data_nonhot(dataset_name):
    arr = np.load(f"../{dataset_name}_nonhot.npy")
    groups = np.load(f"../{dataset_name}_groups.npy")

    arr_train, arr_test, idx_train, idx_test = split_groups(arr, groups)
    del arr
    gc.collect()

    X_train, X_test = arr_train[:, :-1], arr_test[:, :-1]
    y_train = np.ravel(arr_train[:, -1], 'C')
    y_test = np.ravel(arr_test[:, -1], 'C')

    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train, X_test, y_train, y_test


def prepare_classifier(classifier, model_params, hyper_opt=False):
    print(f"Preparing {classifier.name}")

    default_params = model_params.get('default_params', dict())
    hyper_params = model_params.get('hyper_params', dict())

    clf = classifier.scikit_class(**default_params)

    if hyper_opt:
        hyper_params_incl_default = [{**default_params, **hyper_dict}
                                     for hyper_dict in hyper_params]
        clf = GridSearchCV(clf, hyper_params_incl_default)

    return clf


def train_models(X_train, y_train, clfs, parallel=False):
    if parallel:
        with joblib.parallel_backend('dask'):
            for clf in tqdm(clfs):
                print(f"Fitting: {type(clf)}")
                clf.fit(X_train, y_train)
    else:
        for clf in tqdm(clfs):
            print(f"Fitting: {type(clf)}")
            clf.fit(X_train, y_train)


def main():
    from model_parameters import params

    DATASET_NAME = 'df_temporary'

    client = Client(processes=False)    # processes = False ensures shared memory

    classifiers = obtain_scikit_classifiers()
    slow_models = ['AdaBoostClassifier', 'ExtraTreesClassifier', 'MLPClassifier']
    intractable_models = ['GaussianProcessClassifier', 'LabelPropagation', 'LabelSpreading']
    clfs_nt = list(filter(lambda clf_nt: clf_nt.name not in slow_models + intractable_models,
                          classifiers))

    dense_clfs_nt = list(filter(lambda clf_nt: clf_nt.sparse_support is False, clfs_nt))
    sparse_clfs_nt = list(filter(lambda clf_nt: clf_nt.sparse_support is True, clfs_nt))

    dense_clfs = [prepare_classifier(clf_nt, model_params=params.get(clf_nt.name, dict()), hyper_opt=False)
                  for clf_nt in dense_clfs_nt]

    sparse_clfs = [prepare_classifier(clf_nt, model_params=params.get(clf_nt.name, dict()), hyper_opt=False)
                   for clf_nt in sparse_clfs_nt]

    # TODO: create function
    X_train, X_test, y_train, y_test = prepare_data_nonhot(DATASET_NAME)
    train_models(X_train, y_train, dense_clfs, parallel=False)

    # QuadraticDiscriminanAnalysis predicts NaNs => because of collinear?
    for clf_nt, clf in zip(dense_clfs_nt, dense_clfs):
        # TODO: wrap CalibratedClassifierCV if needed and activate predict_proba()
        if clf_nt.proba_support:
            y_prob = clf.predict_proba(X_test)
        score = clf.score(X_test, y_test)
        save_model(clf, model_name=clf_nt.name)

    X_train, X_test, y_train, y_test = prepare_data(DATASET_NAME)
    train_models(X_train, y_train, sparse_clfs, parallel=False)
    for clf_nt, clf in zip(sparse_clfs_nt, sparse_clfs):
        # TODO: wrap CalibratedClassifierCV if needed and activate predict_proba()
        if clf_nt.proba_support:
            y_prob = clf.predict_proba(X_test)
        score = clf.score(X_test, y_test)
        save_model(clf, model_name=clf_nt.name)


    # TODO: use some voting ensemble for best models


if __name__ == '__main__':
    main()


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



    # TODO: use truly chunked
    # if classifier.dask_support:
    #     # parallel_X_train = da.from_array(X_train, (1_000, X_train.shape[1]))
    #     # parallel_y_train = da.from_array(y_train, (1_000, y_train.shape[1]))
    #     cl = Incremental(cl)
    #     with ProgressBar():
    #         cl.fit(X_train, y_train, classes=(0, 1, 5))    # dask needs classes explicit
    # else:
    #     cl.fit(X_train, y_train)

